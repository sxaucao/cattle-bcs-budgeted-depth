import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Label-free local batch inference and timed parity audit for the final candidate."""
import argparse, json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
fd = import_module('19_foundation_pooling')
u = fd.u
ROOT = fd.ROOT
rep = import_module('03_grouped_validation')
pr = import_module('05_score_profile_ablation')
ml = rep.ml

def encode_frames(net, records):
    result = {}
    grid = np.arange(16) * 14 + 7
    gx, gy = np.meshgrid(grid, grid)
    gx = gx.ravel()
    gy = gy.ravel()
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda')[None, :, None, None]
    with torch.inference_mode():
        for st in range(0, len(records), 8):
            xx = []
            masks = []
            for r in records[st:st + 8]:
                z, p = fd.image_input(Path(r['path']), r['depth'])
                xx.append(z)
                x = (gx - p['pad_left']) / p['resized_w']
                y = (gy - p['pad_top']) / p['resized_h']
                masks.append((x >= 0) & (x < 1) & (y >= 0) & (y < 1))
            x = torch.from_numpy(np.stack(xx)).permute(0, 3, 1, 2).to('cuda', dtype=torch.float32)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ff = net.forward_features((x - mean) / std)
            cls = ff['x_norm_clstoken'].float().cpu().numpy()
            tt = ff['x_norm_patchtokens'].float().cpu().numpy().astype('float16').astype('float32')
            for j, r in enumerate(records[st:st + 8]):
                result[r['item']] = (fd.norm(cls[j]), fd.norm(fd.norm(tt[j])[masks[j]].mean(0)))
    return result

def pool(cache, records):
    return np.concatenate([fd.norm(np.mean([cache[r['item']][0] for r in records], 0)), fd.norm(np.mean([cache[r['item']][1] for r in records], 0))]) / np.sqrt(2)

def inputs(manifest):
    rows = u.csv_in(manifest)
    by = defaultdict(list)
    audit = []
    for r in rows:
        for key in ['episode_id', 'frame_index', 'rgb_path', 'depth_path']:
            assert key in r, key
        path = Path(r['rgb_path'])
        path = path if path.is_absolute() else ROOT / path
        with Image.open(path) as im:
            a = np.asarray(im.convert('RGB'))
            digest = hashlib.sha256(a.tobytes()).hexdigest()
        dp = Path(r['depth_path'])
        dp = dp if dp.is_absolute() else ROOT / dp
        assert dp.exists(), 'Missing depth file: ' + str(dp)
        by[r['episode_id']].append({'episode_id': r['episode_id'], 'frame_index': int(r['frame_index']), 'rgb_path': relative_record(path), 'depth_path': relative_record(dp), 'digest': digest})
    final = {}
    for ep, rr in sorted(by.items()):
        unique = {}
        for r in sorted(rr, key=lambda r: r['frame_index']):
            unique.setdefault(r['digest'], r)
        ordered = sorted(unique.values(), key=lambda r: r['frame_index'])
        ix = np.linspace(0, len(ordered) - 1, min(12, len(ordered))).astype(int)
        selected = [ordered[i] for i in ix]
        for i, r in enumerate(selected):
            r['item'] = ep + '__' + str(r['frame_index'])
        final[ep] = selected
        audit.append({'episode_id': ep, 'input_pairs': len(rr), 'decoded_RGB_representatives': len(ordered), 'evaluated_unique_frames': len(selected), 'excluded_redundant_RGB': len(rr) - len(ordered), 'selected_frame_indices': ' '.join((str(r['frame_index']) for r in selected))})
    return (final, audit)

def infer(net, by, fits, gate, cap, policy):
    torch.cuda.synchronize()
    start = time.perf_counter()
    eps = list(by)
    n = len(eps)
    cost = np.array([len(by[e]) for e in eps])
    rgbrecords = [{'episode_id': e, 'item': r['item'], 'path': r['rgb_path'], 'depth': False} for e in eps for r in by[e]]
    rc = encode_frames(net, rgbrecords)
    R = np.stack([pool(rc, by[e]) for e in eps]).astype('float64')
    p0 = rep.predict(fits[0], R)
    prediction = p0.copy()
    level = np.zeros(n, int)
    paid = np.zeros(n, int)
    dc = {}
    probe_count = 0
    extra_count = 0
    B = int(np.floor(cap * cost.sum()))
    orders = {e: ml.order_frames(np.zeros((len(by[e]), 1)), 'temporal_spread') for e in eps}
    p1 = np.full(n, np.nan)
    gate_seconds = 0.0

    def drec(e, r):
        return {'episode_id': e, 'item': r['item'], 'path': r['depth_path'], 'depth': True}
    if policy == 'profile' and B >= n:
        probes = [drec(e, by[e][orders[e][0]]) for e in eps]
        dc = encode_frames(net, probes)
        probe_count = n
        D = np.stack([pool(dc, [by[e][orders[e][0]]]) for e in eps])
        p1 = rep.predict(fits[1], np.concatenate([R, D], 1) / np.sqrt(2))
        s = time.perf_counter()
        obs = pr.profile(np.column_stack([p0, p1, np.zeros((n, 3))]), cost)
        z = (obs - gate['feature_mean']) / gate['feature_sd']
        values = np.exp(-fd.dist(z, gate['train_features']) / float(gate['bandwidth'])) @ gate['coefficients']
        cm = np.column_stack([np.zeros(n, int)] + [np.minimum(k, cost) for k in [1, 3, 6, 12]])
        action, paid, _ = rep.probe.decisions(values, cm, B)
        level = np.array([0, 1, 3, 6, 12])[action]
        gate_seconds = time.perf_counter() - s
    elif policy == 'uniform':
        k = max((k for k in range(13) if np.minimum(k, cost).sum() <= B))
        level[:] = k
        paid = np.minimum(k, cost)
    elif policy == 'full':
        level[:] = 12
        paid = cost.copy()
    extra = []
    for i, e in enumerate(eps):
        if level[i]:
            for j in orders[e][:min(level[i], cost[i])]:
                r = by[e][j]
                if r['item'] not in dc:
                    extra.append(drec(e, r))
    if extra:
        dc.update(encode_frames(net, extra))
        extra_count = len(extra)
    for k in sorted(set(level) - {0}):
        ii = np.flatnonzero(level == k)
        D = np.stack([pool(dc, [by[eps[i]][j] for j in orders[eps[i]][:min(k, cost[i])]]) for i in ii])
        joint = np.concatenate([R[ii], D], 1) / np.sqrt(2)
        prediction[ii] = rep.predict(fits[int(k)], joint)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert int(paid.sum()) == probe_count + extra_count
    if policy != 'full':
        assert paid.sum() <= B
    out = [{'episode_id': e, 'RGB_score': float(p0[i]), 'probe_score': float(p1[i]) if np.isfinite(p1[i]) else '', 'prediction': float(prediction[i]), 'chosen_depth_level': int(level[i]), 'paid_depth_frames': int(paid[i]), 'available_selected_frames': int(cost[i]), 'probe_paid_even_if_RGB_fallback': bool(probe_count and level[i] == 0)} for i, e in enumerate(eps)]
    return (out, {'policy': policy, 'budget_cap': cap, 'events': n, 'RGB_frames': len(rgbrecords), 'depth_frames': probe_count + extra_count, 'probe_frames': probe_count, 'additional_depth_frames': extra_count, 'end_to_end_seconds': elapsed, 'gate_and_DP_seconds': gate_seconds, 'peak_GPU_allocated_MiB': torch.cuda.max_memory_allocated() / 2 ** 20})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--cap', type=float, default=0.25)
    parser.add_argument('--policy', choices=['profile', 'uniform', 'RGB', 'full'], default='profile')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.cap <= 1
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import_module('00_archive_sources').archive()
    repeated = ROOT / project_path('provenance/latest_repeated_validation.txt', ROOT).read_text().strip()
    matched = ROOT / project_path('provenance/latest_matched_controls.txt', ROOT).read_text().strip()
    decision = ROOT / project_path('provenance/latest_decision_profile.txt', ROOT).read_text().strip()
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    out = args.output or project_path('native_runs', ROOT) / ('native_profile_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    if args.manifest is None:
        assert args.benchmark, 'Provide an input manifest for inference'
        test = {r['episode_id'] for r in u.csv_in(repeated / 'group_splits.csv') if int(r['repeat']) == 0 and int(r['fold']) == 0 and (r['role'] == 'test')}
        r = u.csv_in(parent / 'source_RGB_tokens/input_manifest.csv')
        d = u.csv_in(parent / 'source_Depth_tokens/input_manifest.csv')
        manifest = []
        for j, (a, b) in enumerate(zip(r, d)):
            if a['episode_id'] in test:
                manifest.append({'episode_id': a['episode_id'], 'frame_index': j, 'rgb_path': a['path'], 'depth_path': b['path']})
        args.manifest = out / 'input_manifest.csv'
        u.csv_out(args.manifest, manifest, list(manifest[0]))
    by, audit = inputs(args.manifest)
    u.csv_out(out / 'input_audit.csv', audit, list(audit[0]))
    modeldir = args.model_dir or repeated / 'repeat0_fold0'
    gatepath = args.gate or decision / 'repeat0_fold0/Score_profile_RBF.npz'
    fits = {}
    for k in range(13):
        path = modeldir / f'scorer_frames{k}.npz'
        if not path.exists():
            path = matched / 'repeat0_fold0' / f'scorer_frames{k}.npz'
        fits[k] = np.load(path)
    gate = np.load(gatepath)
    net, ac = fd.get_model()
    warm = [{'item': r['item'], 'episode_id': e, 'path': r['rgb_path'], 'depth': False} for e in by for r in by[e]][:8]
    encode_frames(net, warm)
    logs = []
    allpred = []
    parity = []
    controls = [('RGB', 0), ('profile', 0.25), ('profile', 0.5), ('uniform', 0.25), ('uniform', 0.5), ('full', 1)] if args.benchmark else [(args.policy, args.cap)]
    for repeat in range(3 if args.benchmark else 1):
        order = list(controls)
        np.random.default_rng(331 + repeat).shuffle(order)
        for policy, cap in order:
            torch.cuda.reset_peak_memory_stats()
            pred, log = infer(net, by, fits, gate, cap, policy)
            log['timing_repeat'] = repeat
            logs.append(log)
            for r in pred:
                r.update({'policy': policy, 'cap': cap, 'timing_repeat': repeat})
                allpred.append(r)
            print('NATIVE PROFILE', log, flush=True)
    if args.benchmark:
        prr = u.csv_in(decision / 'test_predictions.csv')
        urr = u.csv_in(matched / 'test_predictions.csv')
        brr = u.csv_in(repeated / 'test_predictions.csv')
        for r in allpred:
            method = 'Score_profile_RBF' if r['policy'] == 'profile' else 'Uniform_max_affordable' if r['policy'] == 'uniform' else 'RGB_only' if r['policy'] == 'RGB' else 'RGBD_all'
            source = prr if r['policy'] == 'profile' else urr if r['policy'] == 'uniform' else brr
            old = next((z for z in source if int(z['repeat']) == 0 and int(z['fold']) == 0 and (z['method'] == method) and (float(z['budget_cap']) == r['cap']) and (z['episode_id'] == r['episode_id'])))
            parity.append({'policy': r['policy'], 'cap': r['cap'], 'episode_id': r['episode_id'], 'repeat': r['timing_repeat'], 'abs_prediction_difference': abs(r['prediction'] - float(old['prediction'])), 'same_depth_level': r['chosen_depth_level'] == int(old['depth_level']), 'same_paid_frames': r['paid_depth_frames'] == int(old['paid_depth_frames'])})
        u.csv_out(out / 'prediction_parity.csv', parity, list(parity[0]))
        assert max((r['abs_prediction_difference'] for r in parity)) < 0.005 and all((r['same_depth_level'] and r['same_paid_frames'] for r in parity))
    u.csv_out(out / 'predictions.csv', allpred, list(allpred[0]))
    u.csv_out(out / 'timings.csv', logs, list(logs[0]))
    info = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'benchmark': args.benchmark, 'GPU': torch.cuda.get_device_name(), 'native_files_reloaded': True, 'feature_cache_in_timed_path': False, 'indexing_and_model_loading_excluded': True, 'file_cache': 'warm OS cache', 'events': len(by), 'all_predictions_from_native_input': True, 'max_archive_prediction_difference': max([z['abs_prediction_difference'] for z in parity], default=None), 'archive_action_parity': all((z['same_depth_level'] and z['same_paid_frames'] for z in parity)), 'models': 'Exploratory repeat0 fold0 fitted scorers; no claim of prospective calibration', 'script_sha256': u.sha256(Path(__file__))}
    (out / 'COMPLETE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    if args.benchmark:
        project_path('provenance/latest_native_score_profile.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    print('RUN', out, flush=True)
if __name__ == '__main__':
    main()
