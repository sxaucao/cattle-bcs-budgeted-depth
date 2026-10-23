from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Test depth quantity allocation, retaining all negative outcomes and audits."""
import json, itertools
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
from collections import defaultdict
import numpy as np
from sklearn.svm import SVR
core = import_module('07_train_experiments')
u = core.u
ROOT = core.ROOT
fd = import_module('19_foundation_pooling')
cf = import_module('21_crossfit_residual')
dist = cf.dist
svr_predict = cf.svr_predict

def allocate(values, costs, capacity):
    """One mutually exclusive action per event, including zero-cost zero benefit."""
    values = np.asarray(values, float)
    costs = np.asarray(costs, int)
    n, a = values.shape
    assert costs.shape == values.shape and np.all(costs[:, 0] == 0) and np.all(values[:, 0] == 0)
    dp = np.zeros(capacity + 1)
    choices = np.zeros((n, capacity + 1), dtype=np.int16)
    for i in range(n):
        previous = dp.copy()
        out = previous.copy()
        for j in range(1, a):
            c = int(costs[i, j])
            v = values[i, j]
            if c <= capacity and v > 0:
                candidate = previous[:capacity + 1 - c] + v
                better = candidate > out[c:] + 1e-12
                out[c:][better] = candidate[better]
                choices[i, c:][better] = j
        dp = out
    b = capacity
    action = np.zeros(n, int)
    for i in range(n - 1, -1, -1):
        j = int(choices[i, b])
        action[i] = j
        b -= int(costs[i, j])
    assert costs[np.arange(n), action].sum() <= capacity
    assert abs(values[np.arange(n), action].sum() - dp[capacity]) < 1e-08
    return action

def checks():
    rng = np.random.default_rng(9981)
    for _ in range(25):
        n = 5
        a = 4
        v = np.column_stack([np.zeros(n), rng.normal(size=(n, a - 1))])
        c = np.column_stack([np.zeros(n, int), np.tile([1, 3, 6], (n, 1))])
        B = int(rng.integers(1, 25))
        sol = allocate(v, c, B)
        best = max((sum((v[i, k] for i, k in enumerate(z))) for z in itertools.product(range(a), repeat=n) if sum((c[i, k] for i, k in enumerate(z))) <= B))
        assert abs(v[np.arange(n), sol].sum() - best) < 1e-08

def order_frames(x, policy):
    n = len(x)
    remaining = list(range(n))
    selected = []
    if policy == 'RGB_diversity':
        d = dist(x, x)
        anchor = fd.norm(x.mean(0))
        first = int(np.argmin(dist(x, anchor[None])[:, 0]))
    else:
        t = np.arange(n, dtype=float)
        d = abs(t[:, None] - t[None, :])
        first = int(np.argmin(abs(t - (n - 1) / 2)))
    selected.append(first)
    remaining.remove(first)
    while remaining:
        score = d[np.ix_(remaining, selected)].min(1)
        pick = remaining[int(np.argmax(score))]
        selected.append(pick)
        remaining.remove(pick)
    return np.array(selected)

def main():
    import_module('00_archive_sources').archive()
    checks()
    cfg = json.loads(project_path('configs/multilevel_depth_v1.json', ROOT).read_text())
    bcfg = json.loads(project_path('configs/formal_experiment_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    previous = ROOT / project_path('provenance/latest_selective_depth.txt', ROOT).read_text().strip()
    run = project_path('runs', ROOT) / ('multilevel_depth_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    cache = np.load(parent / 'source_episode_features.npz')
    epids = list(cache['episode_ids'])
    pos = {ep: i for i, ep in enumerate(epids)}
    Xr = cache['DINO_RGB_global_SVR'].astype('float64')
    manifest = u.csv_in(parent / 'source_RGB_tokens/input_manifest.csv')
    by = defaultdict(list)
    for i, r in enumerate(manifest):
        by[r['episode_id']].append(i)
    frame = {}
    for mod in ['RGB', 'Depth']:
        folder = parent / ('source_' + mod + '_tokens')
        tokens = np.load(folder / 'tokens.npy', mmap_mode='r')
        z = np.load(folder / 'cls_and_masks.npz')
        cl = fd.norm(z['cls'])
        po = []
        for i in range(len(tokens)):
            t = fd.norm(np.asarray(tokens[i], dtype='float32'))
            po.append(fd.norm(t[z['valid'][i]].mean(0)))
        frame[mod] = (cl, np.stack(po))
    bundle = core.Bundle(ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip(), bcfg)
    prior = u.csv_in(previous / 'training_benefit_audit.csv')
    sels = u.csv_in(parent / 'source_selected_parameters.csv')
    predictions = []
    audit = []
    baserows = []
    grid = []
    selected = []
    benefitaudit = []
    for policy in cfg['frame_orders']:
        features = {}
        costmatrix = np.zeros((len(epids), 5), int)
        for levelcol, k in enumerate(cfg['depth_levels'], 1):
            vals = []
            for ep in epids:
                rr = np.array(by[ep])
                rf = np.concatenate([frame['RGB'][0][rr], frame['RGB'][1][rr]], 1) / np.sqrt(2)
                order = order_frames(rf, policy)
                chosen = rr[order[:min(k, len(rr))]]
                d = np.concatenate([fd.norm(frame['Depth'][0][chosen].mean(0)), fd.norm(frame['Depth'][1][chosen].mean(0))]) / np.sqrt(2)
                vals.append(np.concatenate([Xr[pos[ep]], d]) / np.sqrt(2))
                costmatrix[pos[ep], levelcol] = len(chosen)
                audit.append({'policy': policy, 'episode_id': ep, 'level': k, 'depth_frames': len(chosen), 'token_rows': ' '.join(map(str, chosen.tolist())), 'pair_ids': ' '.join((manifest[j]['item'] for j in chosen)), 'reference_used': False, 'depth_content_used_for_selection': False})
            features[k] = np.asarray(vals, dtype='float64')
        assert np.max(abs(features[12] - cache['DINO_RGBD_global_SVR'])) < 2e-07
        np.savez_compressed(run / (policy + '_features.npz'), episode_ids=np.array(epids), costs=costmatrix, **{'frames_' + str(k): v for k, v in features.items()})
        for fold in range(3):
            roles = {r: bundle.split(fold, r) for r in ['train', 'validation', 'test']}
            ids = {r: [pos[e['episode_id']] for e in rr] for r, rr in roles.items()}
            y = {r: np.array([int(e['bcs']) for e in rr]) for r, rr in roles.items()}
            groups = np.array([e['animal_group'] for e in roles['train']])
            costs = {r: costmatrix[ids[r]] for r in roles}
            rgbfit = np.load(parent / f'source_fold{fold}_DINO_RGB_global_SVR_fit.npz')
            base = {r: np.clip(svr_predict(rgbfit, np.exp(-dist(Xr[ids[r]], rgbfit['train_features']) / float(rgbfit['bandwidth']))), 2, 6) for r in roles}
            pp = {r: np.column_stack([base[r], np.zeros((len(ids[r]), 4))]) for r in roles}
            audit_lookup = {r['episode_id']: r for r in prior if int(r['fold']) == fold}
            heldrgb = np.array([float(audit_lookup[e['episode_id']]['held_RGB']) for e in roles['train']])
            held = np.column_stack([heldrgb, np.zeros((len(heldrgb), 4))])
            for col, k in enumerate(cfg['depth_levels'], 1):
                xx = features[k][ids['train']]
                dd = dist(xx, xx)
                median = float(np.median(dd[dd > 1e-14]))
                best = None
                if k == 12:
                    s = [r for r in sels if int(r['fold']) == fold and r['variant'] == 'DINO_RGBD_global_SVR'][0]
                    params = {'scale': float(s['scale']), 'alpha': float(s['alpha'])}
                    fit = np.load(parent / f'source_fold{fold}_DINO_RGBD_global_SVR_fit.npz')
                    pred = {r: np.clip(svr_predict(fit, np.exp(-dist(features[k][ids[r]], xx) / float(fit['bandwidth']))), 2, 6) for r in roles}
                else:
                    for scale in [0.5, 1.0, 2.0]:
                        K = np.exp(-dd / (median * scale))
                        Kv = np.exp(-dist(features[k][ids['validation']], xx) / (median * scale))
                        for alpha in [0.01, 0.1, 1.0, 10.0]:
                            model = SVR(kernel='precomputed', C=1 / alpha, epsilon=0.1).fit(K, y['train'])
                            vp = np.clip(model.predict(Kv), 2, 6)
                            mae = float(abs(vp - y['validation']).mean())
                            if best is None or mae < best[0]:
                                best = (mae, scale, alpha, model)
                    params = {'scale': best[1], 'alpha': best[2]}
                    model = best[3]
                    pred = {r: np.clip(model.predict(np.exp(-dist(features[k][ids[r]], xx) / (median * params['scale']))), 2, 6) for r in roles}
                    np.savez_compressed(run / f'{policy}_fold{fold}_frames{k}_SVR.npz', train_features=xx, support=model.support_, dual_coef=model.dual_coef_, intercept=model.intercept_, bandwidth=median * params['scale'])
                for r in roles:
                    pp[r][:, col] = pred[r]
                for g in sorted(set(groups)):
                    tr = np.flatnonzero(groups != g)
                    ho = np.flatnonzero(groups == g)
                    sd = dist(xx[tr], xx[tr])
                    band = float(np.median(sd[sd > 1e-14])) * params['scale']
                    model = SVR(kernel='precomputed', C=1 / params['alpha'], epsilon=0.1).fit(np.exp(-sd / band), y['train'][tr])
                    held[ho, col] = np.clip(model.predict(np.exp(-dist(xx[ho], xx[tr]) / band)), 2, 6)
                    assert not set(groups[tr]) & set(groups[ho])
                for e, p in zip(roles['test'], pred['test']):
                    baserows.append({'policy': policy, 'fold': fold, 'level': k, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference': int(e['bcs']), 'prediction': float(p), 'absolute_error': float(abs(p - int(e['bcs']))), 'actual_frames': int(costmatrix[pos[e['episode_id']], col])})
            benefit = abs(held[:, 0, None] - y['train'][:, None]) - abs(held - y['train'][:, None])
            ins = abs(pp['train'][:, 0, None] - y['train'][:, None]) - abs(pp['train'] - y['train'][:, None])
            perm = np.random.default_rng(20261023 + fold * 701).permutation(len(held))
            for i, e in enumerate(roles['train']):
                for col, k in enumerate([0] + cfg['depth_levels']):
                    benefitaudit.append({'policy': policy, 'fold': fold, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'level': k, 'reference': int(e['bcs']), 'held_RGB': float(held[i, 0]), 'held_action_score': float(held[i, col]), 'benefit_target': float(benefit[i, col]), 'insample_benefit': float(ins[i, col]), 'shuffled_target_donor': roles['train'][perm[i]]['episode_id']})
            xt = Xr[ids['train']]
            dd = dist(xt, xt)
            band = float(np.median(dd[dd > 1e-14]))
            targets = {'multilevel_crossfit': benefit, 'binary_only_crossfit': benefit, 'multilevel_insample': ins, 'multilevel_shuffled': benefit[perm]}
            gate_scores = {}
            for gate, target in targets.items():
                best = None
                columns = [0, 4] if gate == 'binary_only_crossfit' else list(range(5))
                for scale in [0.5, 1.0, 2.0]:
                    K = np.exp(-dd / (band * scale))
                    Kv = np.exp(-dist(Xr[ids['validation']], xt) / (band * scale))
                    Kt = np.exp(-dist(Xr[ids['test']], xt) / (band * scale))
                    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                        coef = np.linalg.solve(K + alpha * np.eye(len(K)), target)
                        vp = Kv @ coef
                        maes = []
                        for cap in [0.25, 0.5, 0.75]:
                            ac = allocate(vp[:, columns], costs['validation'][:, columns], int(np.floor(cap * costs['validation'][:, -1].sum())))
                            action = np.array(columns)[ac]
                            p = pp['validation'][np.arange(len(ac)), action]
                            maes.append(float(abs(p - y['validation']).mean()))
                        key = float(np.mean(maes))
                        grid.append({'policy': policy, 'fold': fold, 'gate': gate, 'scale': scale, 'alpha': alpha, 'validation_MAE': key})
                        if best is None or key < best[0]:
                            best = (key, scale, alpha, coef, Kt @ coef)
                selected.append({'policy': policy, 'fold': fold, 'gate': gate, 'scale': best[1], 'alpha': best[2], 'validation_MAE': best[0]})
                gate_scores[gate] = (best[4], columns)
                np.savez_compressed(run / f'{policy}_fold{fold}_{gate}.npz', RGB_train=xt, coefficients=best[3], bandwidth=band * best[1])
            gate_scores['multilevel_oracle'] = (abs(pp['test'][:, 0, None] - y['test'][:, None]) - abs(pp['test'] - y['test'][:, None]), list(range(5)))
            for gate, (scores, columns) in gate_scores.items():
                for cap in cfg['budget_caps']:
                    B = int(np.floor(cap * costs['test'][:, -1].sum()))
                    ac = allocate(scores[:, columns], costs['test'][:, columns], B)
                    action = np.array(columns)[ac]
                    used = costs['test'][np.arange(len(ac)), action]
                    p = pp['test'][np.arange(len(ac)), action]
                    assert used.sum() <= B
                    for i, e in enumerate(roles['test']):
                        predictions.append({'policy': policy, 'gate': gate, 'fold': fold, 'budget_cap_fraction': cap, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference': int(e['bcs']), 'prediction': float(p[i]), 'absolute_error': float(abs(p[i] - int(e['bcs']))), 'depth_level': [0, 1, 3, 6, 12][action[i]], 'depth_frames': int(used[i]), 'full_depth_frames': int(costs['test'][i, -1]), 'batch_budget': B, 'batch_used_frames': int(used.sum()), 'predicted_benefit': float(scores[i, action[i]]), 'oracle': gate.endswith('oracle')})
            print('MULTILEVEL COMPLETE', policy, fold, flush=True)
    for name, rows in [('test_predictions.csv', predictions), ('frame_selection_audit.csv', audit), ('base_test_predictions.csv', baserows), ('gate_validation_grid.csv', grid), ('selected_gate_parameters.csv', selected), ('training_benefit_audit.csv', benefitaudit)]:
        u.csv_out(run / name, rows, list(rows[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'events': 151, 'animal_groups': 52, 'bruteforce_checks': 25, 'all_events_scored': True, 'exploratory': True, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_multilevel_depth.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    (run / 'checksums.json').write_text(json.dumps([{'file': p.name, 'sha256': u.sha256(p)} for p in sorted(run.iterdir()) if p.is_file()], indent=2), encoding='utf-8')
    for policy in cfg['frame_orders']:
        for gate in gate_scores:
            for cap in cfg['budget_caps']:
                rr = [r for r in predictions if r['policy'] == policy and r['gate'] == gate and (r['budget_cap_fraction'] == cap)]
                print('MULTILEVEL RESULT', policy, gate, cap, 'MAE', np.mean([r['absolute_error'] for r in rr]), 'depth_frames', sum((r['depth_frames'] for r in rr)), flush=True)
    print('RUN', run, flush=True)
