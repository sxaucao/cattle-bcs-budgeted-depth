import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Frozen-model depth-loss stress tests with native-pixel mask provenance."""
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
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
probe = rep.probe
ml = rep.ml

def extract(net, records, profile):
    by = defaultdict(list)
    audit = []
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
                path = Path(r['path'])
                d = np.load(path, allow_pickle=False).astype('float32')
                valid = d > 0
                before = int(valid.sum())
                seed = int(hashlib.sha256((r['item'] + profile).encode()).hexdigest()[:16], 16)
                if profile.startswith('random_drop'):
                    fraction = 0.1 if profile.endswith('10') else 0.3
                    mask = np.random.default_rng(seed).random(d.shape) < fraction
                    rect = ''
                else:
                    h, w = d.shape
                    top = (h - h // 2) // 2
                    left = (w - w // 2) // 2
                    mask = np.zeros(d.shape, bool)
                    mask[top:top + h // 2, left:left + w // 2] = 1
                    rect = f'{top},{left},{h // 2},{w // 2}'
                d[mask] = 0
                v = d > 0
                after = int(v.sum())
                assert after > 0
                lo, hi = np.percentile(d[v], [1, 99])
                x = np.clip((d - lo) / max(hi - lo, 1), 0, 1)
                x[~v] = 0
                ch = []
                for _ in range(3):
                    im, p = fd.prep.letterbox(x, (224, 224), Image.Resampling.BILINEAR)
                    ch.append(im)
                xx.append(np.stack(ch, 2).astype('float32'))
                ax = (gx - p['pad_left']) / p['resized_w']
                ay = (gy - p['pad_top']) / p['resized_h']
                masks.append((ax >= 0) & (ax < 1) & (ay >= 0) & (ay < 1))
                audit.append({'profile': profile, 'pair_id': r['item'], 'episode_id': r['episode_id'], 'native_path': str(path), 'native_sha256': u.sha256(path), 'height': d.shape[0], 'width': d.shape[1], 'mask_seed': seed, 'central_rectangle_top_left_h_w': rect, 'positive_before': before, 'positive_after': after, 'positive_loss_fraction': 1 - after / before, 'q1_after': float(lo), 'q99_after': float(hi), 'model_input_sha256': hashlib.sha256(xx[-1].tobytes()).hexdigest()})
            z = torch.from_numpy(np.stack(xx)).permute(0, 3, 1, 2).to('cuda', dtype=torch.float32)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ff = net.forward_features((z - mean) / std)
            cls = ff['x_norm_clstoken'].float().cpu().numpy()
            tt = ff['x_norm_patchtokens'].float().cpu().numpy().astype('float16').astype('float32')
            for i, r in enumerate(records[st:st + 8]):
                by[r['episode_id']].append((fd.norm(cls[i]), fd.norm(fd.norm(tt[i])[masks[i]].mean(0))))
            if st % 320 == 0:
                print('STRESS EXTRACT', profile, st, len(records), flush=True)
    return (by, audit)

def main():
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/depth_stress_v1.json', ROOT).read_text())
    repeat = ROOT / project_path('provenance/latest_repeated_validation.txt', ROOT).read_text().strip()
    matched = ROOT / project_path('provenance/latest_matched_controls.txt', ROOT).read_text().strip()
    dec = ROOT / project_path('provenance/latest_decision_profile.txt', ROOT).read_text().strip()
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    f = np.load(matched / 'all_temporal_levels_features.npz')
    epids = list(f['episode_ids'])
    pos = {e: i for i, e in enumerate(epids)}
    RGB = f['frames_0'].astype('float64')
    costs = f['costs']
    splits = u.csv_in(repeat / 'group_splits.csv')
    original = {}
    for ptr in [repeat, matched]:
        for r in u.csv_in(ptr / 'test_predictions.csv'):
            if r['method'] in cfg['methods']:
                original[int(r['repeat']), int(r['fold']), r['method'], float(r['budget_cap']), r['episode_id']] = r
    records = u.csv_in(parent / 'source_Depth_tokens/input_manifest.csv')
    run = project_path('stress_runs', ROOT) / ('depth_stress_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    net, ac = fd.get_model()
    allpred = []
    allaudit = []
    for profile in cfg['profiles']:
        pooled, audit = extract(net, records, profile)
        allaudit.extend(audit)
        features = {0: RGB}
        for k in range(1, 13):
            vv = []
            for ep in epids:
                seq = pooled[ep]
                order = ml.order_frames(np.zeros((len(seq), 1)), 'temporal_spread')
                chosen = [seq[i] for i in order[:min(k, len(seq))]]
                d = np.concatenate([fd.norm(np.mean([z[0] for z in chosen], 0)), fd.norm(np.mean([z[1] for z in chosen], 0))]) / np.sqrt(2)
                vv.append(np.concatenate([RGB[pos[ep]], d]) / np.sqrt(2))
            features[k] = np.asarray(vv, dtype='float64')
        np.savez_compressed(run / (profile + '_features.npz'), episode_ids=np.array(epids), **{'frames_' + str(k): v for k, v in features.items()})
        for repetition in range(3):
            for fold in range(3):
                rr = [r for r in splits if int(r['repeat']) == repetition and int(r['fold']) == fold and (r['role'] == 'test')]
                ii = [pos[r['episode_id']] for r in rr]
                y = np.array([int(r['reference']) for r in rr])
                cc = costs[ii]
                pp = np.zeros((len(ii), 13))
                folder = repeat / f'repeat{repetition}_fold{fold}'
                for k in range(13):
                    path = folder / f'scorer_frames{k}.npz'
                    if not path.exists():
                        path = matched / f'repeat{repetition}_fold{fold}' / f'scorer_frames{k}.npz'
                    fit = np.load(path)
                    pp[:, k] = rep.predict(fit, features[k][ii])
                for method in cfg['methods']:
                    for cap in cfg['budget_caps']:
                        B = int(np.floor(cap * cc[:, -1].sum()))
                        if method.startswith('Score_profile'):
                            gate = np.load(dec / f'repeat{repetition}_fold{fold}' / (method + '.npz'))
                            obs = pr.profile(pp[:, [0, 1, 3, 6, 12]], cc[:, -1])
                            z = (obs - gate['feature_mean']) / gate['feature_sd']
                            value = np.exp(-fd.dist(z, gate['train_features']) / float(gate['bandwidth'])) @ gate['coefficients']
                            action, paid, _ = probe.decisions(value, cc[:, [0, 1, 3, 6, 12]], B)
                            level = np.array([0, 1, 3, 6, 12])[action]
                        else:
                            old = [original[repetition, fold, method, cap, r['episode_id']] for r in rr]
                            level = np.array([int(r['depth_level']) for r in old])
                            paid = np.array([int(r['paid_depth_frames']) for r in old])
                        prediction = pp[np.arange(len(ii)), level]
                        for j, r in enumerate(rr):
                            allpred.append({'profile': profile, 'repeat': repetition, 'fold': fold, 'method': method, 'budget_cap': cap, 'episode_id': r['episode_id'], 'animal_group': r['animal_group'], 'reference': int(r['reference']), 'prediction': float(prediction[j]), 'absolute_error': float(abs(prediction[j] - y[j])), 'depth_level': int(level[j]), 'paid_depth_frames': int(paid[j]), 'full_depth_frames': int(cc[j, -1]), 'batch_budget': B, 'batch_paid_frames': int(paid.sum()), 'budget_feasible': bool(paid.sum() <= B), 'all_events_scored': True})
        print('STRESS PROFILE COMPLETE', profile, flush=True)
    u.csv_out(run / 'test_predictions.csv', allpred, list(allpred[0]))
    u.csv_out(run / 'corruption_audit.csv', allaudit, list(allaudit[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'profiles': cfg['profiles'], 'source_events': 151, 'source_frames_per_profile': 1245, 'all_existing_models_frozen': True, 'no_stress_specific_HPO': True, 'native_files_unchanged': True, 'independent_external_validation': False, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_depth_stress.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    for profile in cfg['profiles']:
        for method in cfg['methods']:
            zz = [z for z in allpred if z['profile'] == profile and z['method'] == method and (z['budget_cap'] == 0.25)]
            print('STRESS RESULT', profile, method, 'MAE', np.mean([z['absolute_error'] for z in zz]), flush=True)
    print('RUN', run, flush=True)
if __name__ == '__main__':
    main()
