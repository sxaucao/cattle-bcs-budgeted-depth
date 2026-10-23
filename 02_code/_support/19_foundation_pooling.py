from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Frozen DINOv2 region/episode pooling with same-encoder controls on two cohorts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, math, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.svm import SVR
u = import_module('01_audit_and_snapshot')
core = import_module('07_train_experiments')
prep = import_module('02_prepare_data')
ROOT = core.ROOT

def norm(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)

def get_model(device='cuda'):
    ac = json.loads(project_path('models/dinov2/acquisition.json', ROOT).read_text())
    assert u.sha256(ROOT / ac['weights']) == ac['weights_sha256']
    sys.path.insert(0, str(ROOT / ac['source_directory']))
    from dinov2.hub.backbones import dinov2_vits14
    net = dinov2_vits14(pretrained=False)
    net.load_state_dict(torch.load(ROOT / ac['weights'], map_location='cpu', weights_only=True), strict=True)
    return (net.to(device).eval().requires_grad_(False), ac)

def image_input(path, depth=False):
    if depth:
        d = np.load(path, allow_pickle=False).astype('float32')
        v = d > 0
        lo, hi = np.percentile(d[v], [1, 99])
        x = np.clip((d - lo) / max(hi - lo, 1), 0, 1)
        x[~v] = 0
        x = np.repeat(x[..., None], 3, 2)
    else:
        with Image.open(path) as im:
            x = np.asarray(im.convert('RGB')).astype('float32') / 255
    chans = []
    for k in range(3):
        y, params = prep.letterbox(x[..., k], (224, 224), Image.Resampling.BILINEAR)
        chans.append(y)
    return (np.stack(chans, 2).astype('float32'), params)

def extract(net, records, out, device='cuda'):
    n = len(records)
    tokens = np.lib.format.open_memmap(out / 'tokens.npy', mode='w+', dtype='float16', shape=(n, 256, 384))
    cls = np.zeros((n, 384), dtype='float32')
    valid = np.zeros((n, 256), bool)
    centre = np.zeros((n, 256), bool)
    log = []
    grid = np.arange(16) * 14 + 7
    gx, gy = np.meshgrid(grid, grid)
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.inference_mode():
        for st in range(0, n, 8):
            imgs = []
            for j, r in enumerate(records[st:st + 8], st):
                x, p = image_input(Path(r['path']), r.get('depth', False))
                imgs.append(x)
                xx = (gx - p['pad_left']) / p['resized_w']
                yy = (gy - p['pad_top']) / p['resized_h']
                vv = (xx >= 0) & (xx < 1) & (yy >= 0) & (yy < 1)
                cc = vv & (xx >= 0.35) & (xx <= 0.65) & (yy >= 0.15) & (yy <= 0.85)
                assert vv.any() and cc.any()
                valid[j] = vv
                centre[j] = cc
                log.append({'row': j, 'item': r['item'], 'episode_id': r['episode_id'], 'depth': r.get('depth', False), 'path': relative_record(r['path']), 'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(), **p, 'valid_tokens': int(vv.sum()), 'centre_tokens': int(cc.sum())})
            xx = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                ff = net.forward_features((xx - mean) / std)
            tokens[st:st + len(imgs)] = ff['x_norm_patchtokens'].float().cpu().numpy().astype('float16')
            cls[st:st + len(imgs)] = ff['x_norm_clstoken'].float().cpu().numpy()
            if st % 160 == 0:
                print(f'EXTRACT {out.name} {st}/{n}', flush=True)
    tokens.flush()
    np.savez_compressed(out / 'cls_and_masks.npz', cls=cls, valid=valid, centre=centre)
    u.csv_out(out / 'input_manifest.csv', log, list(log[0]))
    return (tokens, cls, valid, centre)

def pool(tokens, cls, valid, centre, records):
    by = defaultdict(list)
    for i, r in enumerate(records):
        by[r['episode_id']].append(i)
    features = {k: {} for k in ['CLS', 'global', 'centre', 'frame_local', 'consensus', 'low_similarity']}
    selector = []
    for ep, indices in sorted(by.items()):
        tt = [norm(np.asarray(tokens[i], dtype='float32')) for i in indices]
        anchors = [norm(t[centre[i]].mean(0)) for t, i in zip(tt, indices)]
        anchor = norm(np.mean(anchors, 0))
        fc = []
        pools = {k: [] for k in features if k != 'CLS'}
        for z, (t, i, local) in enumerate(zip(tt, indices, anchors)):
            ids = np.flatnonzero(valid[i])
            n = max(1, int(math.ceil(len(ids) * 0.25)))
            sim = t @ anchor
            loc = t @ local
            ix = ids[np.argsort(sim[ids], kind='stable')[-n:]]
            low = ids[np.argsort(sim[ids], kind='stable')[:n]]
            fr = ids[np.argsort(loc[ids], kind='stable')[-n:]]
            fc.append(norm(cls[i]))
            vectors = {'global': t[ids].mean(0), 'centre': t[centre[i]].mean(0), 'frame_local': t[fr].mean(0), 'consensus': t[ix].mean(0), 'low_similarity': t[low].mean(0)}
            for k, v in vectors.items():
                pools[k].append(norm(v))
            mask = np.zeros(256, dtype='uint8')
            mask[ix] = 1
            selector.append({'episode_id': ep, 'item': records[i]['item'], 'row': i, 'selected_patch_indices': ' '.join(map(str, ix.tolist())), 'low_similarity_patch_indices': ' '.join(map(str, low.tolist())), 'valid_tokens': len(ids), 'selected_tokens': n, 'centre_selected_fraction': float(centre[i][ix].mean()), 'selected_mean_cosine': float(sim[ix].mean()), 'low_mean_cosine': float(sim[low].mean())})
        c = norm(np.mean(fc, 0))
        features['CLS'][ep] = c
        for k, vs in pools.items():
            features[k][ep] = np.concatenate([c, norm(np.mean(vs, 0))]) / np.sqrt(2)
    return (features, selector)

def dist(x, z):
    return np.maximum(np.square(x).sum(1)[:, None] + np.square(z).sum(1)[None, :] - 2 * x @ z.T, 0)

def fit_families(run, features, epids, splits, references, scale, cohort):
    pos = {ep: i for i, ep in enumerate(epids)}
    preds = []
    fitlog = []
    selected = []
    for fold in range(3):
        sets = {role: sorted({r['episode_id'] for r in splits if int(r['test_fold']) == fold and r['role'] == role}) for role in ['train', 'validation', 'test']}
        ids = {r: [pos[e] for e in rr] for r, rr in sets.items()}
        y = {r: np.array([references[e]['reference'] for e in rr]) for r, rr in sets.items()}
        groups = {r: {references[e]['animal_group'] for e in rr} for r, rr in sets.items()}
        assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
        for arm, xx in features.items():
            x = np.asarray(xx, dtype='float64')
            xt = x[ids['train']]
            dd = dist(xt, xt)
            band = float(np.median(dd[dd > 1e-14]))
            best = None
            for s in [0.5, 1.0, 2.0]:
                K = np.exp(-dd / (band * s))
                Kv = np.exp(-dist(x[ids['validation']], xt) / (band * s))
                Kt = np.exp(-dist(x[ids['test']], xt) / (band * s))
                for a in [0.01, 0.1, 1.0, 10.0]:
                    model = SVR(kernel='precomputed', C=1 / a, epsilon=0.1).fit(K, y['train'])
                    v = np.clip(model.predict(Kv), *scale)
                    mae = float(abs(v - y['validation']).mean())
                    row = {'cohort': cohort, 'variant': arm, 'fold': fold, 'alpha': a, 'scale': s, 'validation_MAE': mae}
                    fitlog.append(row)
                    if best is None or mae < best[0]:
                        best = (mae, row.copy(), model, np.clip(model.predict(Kt), *scale))
            selected.append(best[1])
            model = best[2]
            np.savez_compressed(run / f'{cohort}_fold{fold}_{arm}_fit.npz', train_features=xt, support=model.support_, dual_coef=model.dual_coef_, intercept=model.intercept_, bandwidth=band * best[1]['scale'])
            for ep, p in zip(sets['test'], best[3]):
                ref = references[ep]
                preds.append({'variant': arm, 'fold': fold, 'seed': 'deterministic', 'role': 'test', 'episode_id': ep, 'animal_group': ref['animal_group'], 'reference_BCS': ref['reference'], 'predicted_BCS': float(np.clip(np.rint(p), *scale)), 'expected_BCS': float(p), **{k: ref[k] for k in ['BCS1', 'BCS2'] if k in ref}})
            print('FOUNDATION DONE', cohort, fold, arm, 'val', best[0], 'test', float(abs(best[3] - y['test']).mean()), flush=True)
    u.csv_out(run / (cohort + '_test_predictions.csv'), preds, list(preds[0]))
    u.csv_out(run / (cohort + '_validation_grid.csv'), fitlog, list(fitlog[0]))
    u.csv_out(run / (cohort + '_selected_parameters.csv'), selected, list(selected[0]))
    for arm in features:
        rr = [r for r in preds if r['variant'] == arm]
        print('POOLED FOUNDATION', cohort, arm, np.mean([abs(r['expected_BCS'] - r['reference_BCS']) for r in rr]), flush=True)

def main():
    import_module('00_archive_sources').archive()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(project_path('configs/foundation_pooling_v1.json', ROOT).read_text())
    bcfg = json.loads(project_path('configs/formal_experiment_v1.json', ROOT).read_text())
    run = project_path('runs', ROOT) / ('foundation_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    net, ac = get_model()
    bundle = core.Bundle(ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip(), bcfg)
    eps = sorted({r['episode_id']: r for r in bundle.splits}.values(), key=lambda r: r['episode_id'])
    records = {m: [] for m in ['RGB', 'Depth']}
    for e in eps:
        rs = bundle.by_ep[e['episode_id']]
        ix = np.linspace(0, len(rs) - 1, min(12, len(rs))).astype(int)
        for i in ix:
            r = rs[i]
            for m, key in [('RGB', 'rgb_relpath'), ('Depth', 'depth_relpath')]:
                records[m].append({'episode_id': e['episode_id'], 'item': r['pair_id'], 'path': str(ROOT / r[key]), 'depth': m == 'Depth'})
    allfeatures = {}
    epids = [e['episode_id'] for e in eps]
    for m in records:
        out = run / ('source_' + m + '_tokens')
        out.mkdir()
        tok, cl, v, c = extract(net, records[m], out)
        feats, selector = pool(tok, cl, v, c, records[m])
        u.csv_out(out / 'region_selector_audit.csv', selector, list(selector[0]))
        for k, vals in feats.items():
            allfeatures[f'DINO_{m}_{k}_SVR'] = np.stack([vals[ep] for ep in epids])
    for k in ['global', 'centre', 'consensus']:
        allfeatures[f'DINO_RGBD_{k}_SVR'] = np.concatenate([allfeatures[f'DINO_RGB_{k}_SVR'], allfeatures[f'DINO_Depth_{k}_SVR']], axis=1) / np.sqrt(2)
    np.savez_compressed(run / 'source_episode_features.npz', episode_ids=np.array(epids), **allfeatures)
    refs = {e['episode_id']: {'reference': int(e['bcs']), 'animal_group': e['animal_group']} for e in eps}
    fit_families(run, allfeatures, epids, bundle.splits, refs, (2, 6), 'source')
    external = ROOT / project_path('provenance/latest_external_data.txt', ROOT).read_text().strip()
    erows = sorted(u.csv_in(external / 'image_annotation_manifest.csv'), key=lambda r: r['image_name'])
    erecord = [{'episode_id': r['image_name'], 'item': r['image_name'], 'path': str(external / 'raw_images' / r['image_name']), 'depth': False} for r in erows]
    out = run / 'dairy_RGB_tokens'
    out.mkdir()
    tok, cl, v, c = extract(net, erecord, out)
    ef, selector = pool(tok, cl, v, c, erecord)
    u.csv_out(out / 'region_selector_audit.csv', selector, list(selector[0]))
    epids = [r['image_name'] for r in erows]
    efeatures = {f'DINO_RGB_{k}_SVR': np.stack([vals[ep] for ep in epids]) for k, vals in ef.items()}
    assert np.allclose(efeatures['DINO_RGB_frame_local_SVR'], efeatures['DINO_RGB_consensus_SVR'])
    np.savez_compressed(run / 'dairy_image_features.npz', episode_ids=np.array(epids), **efeatures)
    refs = {r['image_name']: {'reference': float(r['native_BCS_mean']), 'animal_group': r['animal_group'], 'BCS1': float(r['BCS1']), 'BCS2': float(r['BCS2'])} for r in erows}
    splits = [{**r, 'episode_id': r['image_name']} for r in u.csv_in(external / 'group_splits.csv')]
    fit_families(run, efeatures, epids, splits, refs, (1, 5), 'dairy')
    info = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_episodes': 151, 'source_selected_frames': len(records['RGB']), 'dairy_images': 207, 'source_arms': len(allfeatures), 'dairy_arms': len(efeatures), 'encoder_acquisition': ac, 'script_sha256': u.sha256(Path(__file__)), 'config_sha256': u.sha256(project_path('configs/foundation_pooling_v1.json', ROOT)), 'trained_BCS_encoder_parameters': 0, 'outer_tests_previously_observed_with_other_model_families': True, 'exploratory': True, 'dairy_consensus_equals_frame_local': True}
    (run / 'COMPLETE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    project_path('provenance/latest_foundation_experiment.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    (run / 'checksums.json').write_text(json.dumps([{'file': p.relative_to(run).as_posix(), 'sha256': u.sha256(p)} for p in sorted(run.rglob('*')) if p.is_file()], indent=2), encoding='utf-8')
    print('FOUNDATION EXPERIMENT COMPLETE ' + str(run), flush=True)
