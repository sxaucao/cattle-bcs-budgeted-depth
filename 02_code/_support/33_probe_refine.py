from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""A depth probe buys information before allocating additional frame processing."""
import json
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module
import numpy as np
fd = import_module('19_foundation_pooling')
core = fd.core
u = fd.u
ROOT = fd.ROOT
rt = import_module('27_frame_cost_routing')
ml = import_module('28_multilevel_depth')

def decisions(values, costs, budget, greedy=False):
    n = len(costs)
    if budget < n:
        return (np.zeros(n, int), np.zeros(n, int), False)
    extra = np.maximum(costs - 1, 0)
    B = budget - n
    if greedy:
        action = np.zeros(n, int)
        spent = 0
        for i in range(n):
            free = np.flatnonzero(extra[i] == 0)
            action[i] = free[np.argmax(values[i, free])]
        while True:
            best = None
            for i in range(n):
                for j in range(1, 5):
                    dc = int(extra[i, j] - extra[i, action[i]])
                    dv = values[i, j] - values[i, action[i]]
                    if dc > 0 and dv > 0 and (spent + dc <= B):
                        key = dv / dc
                        if best is None or key > best[0]:
                            best = (key, i, j, dc)
            if best is None:
                break
            _, i, j, dc = best
            action[i] = j
            spent += dc
    else:
        action = ml.allocate(values, extra, B)
    paid = np.maximum(costs[np.arange(n), action], 1)
    assert paid.sum() <= budget
    return (action, paid, True)

def main():
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/probe_refine_v1.json', ROOT).read_text())
    bcfg = json.loads(project_path('configs/formal_experiment_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    mult = ROOT / project_path('provenance/latest_multilevel_depth.txt', ROOT).read_text().strip()
    f = np.load(mult / 'temporal_spread_features.npz')
    epids = list(f['episode_ids'])
    pos = {ep: i for i, ep in enumerate(epids)}
    rgb = np.load(parent / 'source_episode_features.npz')['DINO_RGB_global_SVR'].astype('float64')
    probe = f['frames_1'].astype('float64')
    features = {k: f['frames_' + str(k)].astype('float64') for k in [1, 3, 6, 12]}
    costs = f['costs']
    bundle = core.Bundle(ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip(), bcfg)
    heldrows = u.csv_in(mult / 'training_benefit_audit.csv')
    run = project_path('runs', ROOT) / ('probe_refine_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    predictions = []
    grid = []
    selected = []
    audits = []
    for fold in range(3):
        roles = {r: bundle.split(fold, r) for r in ['train', 'validation', 'test']}
        ids = {r: [pos[e['episode_id']] for e in rr] for r, rr in roles.items()}
        y = {r: np.array([int(e['bcs']) for e in rr]) for r, rr in roles.items()}
        cc = {r: costs[ids[r]] for r in roles}
        pp = {r: np.zeros((len(ids[r]), 5)) for r in roles}
        rfit = np.load(parent / f'source_fold{fold}_DINO_RGB_global_SVR_fit.npz')
        for r in roles:
            pp[r][:, 0] = np.clip(rt.svr_predict(rfit, np.exp(-fd.dist(rgb[ids[r]], rfit['train_features']) / float(rfit['bandwidth']))), 2, 6)
        for col, k in enumerate([1, 3, 6, 12], 1):
            fit = np.load(parent / f'source_fold{fold}_DINO_RGBD_global_SVR_fit.npz') if k == 12 else np.load(mult / f'temporal_spread_fold{fold}_frames{k}_SVR.npz')
            for r in roles:
                pp[r][:, col] = np.clip(rt.svr_predict(fit, np.exp(-fd.dist(features[k][ids[r]], fit['train_features']) / float(fit['bandwidth']))), 2, 6)
        lookup = {(r['episode_id'], int(r['level'])): r for r in heldrows if r['policy'] == 'temporal_spread' and int(r['fold']) == fold}
        benefit = np.array([[float(lookup[e['episode_id'], k]['benefit_target']) for k in [0, 1, 3, 6, 12]] for e in roles['train']])
        ins = abs(pp['train'][:, 0, None] - y['train'][:, None]) - abs(pp['train'] - y['train'][:, None])
        perm = np.random.default_rng(9901 + fold).permutation(len(benefit))
        scores = {}
        for method in cfg['methods']:
            if method == 'Probe_greedy_ratio':
                continue
            X = rgb if method == 'RGB_gate_with_probe_cost' else probe
            target = ins if method == 'Probe_insample' else benefit[perm] if method == 'Probe_shuffled' else benefit
            xt = X[ids['train']]
            dd = fd.dist(xt, xt)
            band = float(np.median(dd[dd > 1e-14]))
            best = None
            for scale in [0.5, 1.0, 2.0]:
                K = np.exp(-dd / (band * scale))
                Kv = np.exp(-fd.dist(X[ids['validation']], xt) / (band * scale))
                Kt = np.exp(-fd.dist(X[ids['test']], xt) / (band * scale))
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    coef = np.linalg.solve(K + alpha * np.eye(len(K)), target)
                    v = Kv @ coef
                    maes = []
                    for cap in [0.25, 0.5, 0.75]:
                        action, paid, observed = decisions(v, cc['validation'], int(np.floor(cap * cc['validation'][:, -1].sum())))
                        p = pp['validation'][np.arange(len(action)), action]
                        maes.append(float(abs(p - y['validation']).mean()))
                    key = float(np.mean(maes))
                    row = {'method': method, 'fold': fold, 'scale': scale, 'alpha': alpha, 'validation_MAE': key}
                    grid.append(row)
                    if best is None or key < best[0]:
                        best = (key, row.copy(), Kt @ coef, coef, band * scale)
            selected.append(best[1])
            scores[method] = best[2]
            np.savez_compressed(run / f'fold{fold}_{method}.npz', train_features=xt, coefficients=best[3], bandwidth=best[4])
        scores['Probe_greedy_ratio'] = scores['Probe_crossfit']
        for method, s in scores.items():
            for cap in cfg['budget_caps']:
                B = int(np.floor(cap * cc['test'][:, -1].sum()))
                action, paid, observed = decisions(s, cc['test'], B, method == 'Probe_greedy_ratio')
                p = pp['test'][np.arange(len(action)), action]
                for i, e in enumerate(roles['test']):
                    predictions.append({'method': method, 'fold': fold, 'budget_cap_fraction': cap, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference': int(e['bcs']), 'prediction': float(p[i]), 'absolute_error': float(abs(p[i] - int(e['bcs']))), 'chosen_depth_level': [0, 1, 3, 6, 12][action[i]], 'paid_depth_frames': int(paid[i]), 'full_depth_frames': int(cc['test'][i, -1]), 'batch_budget': B, 'batch_paid_frames': int(paid.sum()), 'mandatory_probe_observed': observed, 'probe_paid_despite_RGB_fallback': bool(observed and action[i] == 0)})
                audits.append({'method': method, 'fold': fold, 'budget_cap': cap, 'budget': B, 'frames_paid': int(paid.sum()), 'mandatory_probe_frames': len(action) if observed else 0, 'all_events_scored': True, 'budget_respected': bool(paid.sum() <= B)})
        print('PROBE REFINEMENT COMPLETE', fold, flush=True)
    for name, rr in [('test_predictions.csv', predictions), ('validation_grid.csv', grid), ('selected_parameters.csv', selected), ('budget_audit.csv', audits)]:
        u.csv_out(run / name, rr, list(rr[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'events': 151, 'all_budgets_respected': True, 'exploratory': True, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_probe_refine.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    for method in scores:
        for cap in cfg['budget_caps']:
            rr = [r for r in predictions if r['method'] == method and r['budget_cap_fraction'] == cap]
            print('PROBE RESULT', method, cap, 'MAE', np.mean([r['absolute_error'] for r in rr]), 'depth_frames', sum((r['paid_depth_frames'] for r in rr)), flush=True)
    print('RUN', run, flush=True)
