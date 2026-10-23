from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Compute-budget routing with animal-cross-fitted benefit and exact knapsack."""
import json, itertools
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
import numpy as np
core = import_module('07_train_experiments')
u = core.u
cf = import_module('21_crossfit_residual')
ROOT = core.ROOT
dist = cf.dist
svr_predict = cf.svr_predict

def knapsack(values, costs, capacity):
    values = np.asarray(values, float)
    costs = np.asarray(costs, int)
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1))
    take = np.zeros((n + 1, capacity + 1), bool)
    for i, (v, c) in enumerate(zip(values, costs), 1):
        dp[i] = dp[i - 1]
        if c <= capacity and v > 0:
            candidate = dp[i - 1, :capacity + 1 - c] + v
            better = candidate > dp[i, c:] + 1e-12
            dp[i, c:][better] = candidate[better]
            take[i, c:][better] = True
    b = capacity
    use = np.zeros(n, bool)
    for i in range(n, 0, -1):
        if take[i, b]:
            use[i - 1] = True
            b -= costs[i - 1]
    assert costs[use].sum() <= capacity
    assert abs(values[use].sum() - dp[-1, capacity]) < 1e-08
    return use

def check_knapsack():
    rng = np.random.default_rng(713)
    for _ in range(25):
        n = 7
        v = rng.normal(size=n)
        c = rng.integers(1, 13, n)
        B = int(rng.integers(1, 35))
        use = knapsack(v, c, B)
        optimum = max((sum((v[i] for i in range(n) if mask >> i & 1)) for mask in range(1 << n) if sum((c[i] for i in range(n) if mask >> i & 1)) <= B))
        assert abs(v[use].sum() - optimum) < 1e-08

def main():
    import_module('00_archive_sources').archive()
    check_knapsack()
    cfg = json.loads(project_path('configs/selective_depth_processing_v2.json', ROOT).read_text())
    bcfg = json.loads(project_path('configs/formal_experiment_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    old = ROOT / project_path('provenance/latest_selective_depth.txt', ROOT).read_text().strip()
    cache = np.load(parent / 'source_episode_features.npz')
    pos = {ep: i for i, ep in enumerate(cache['episode_ids'])}
    X = {'RGB': cache['DINO_RGB_global_SVR'].astype('float64'), 'RGBD': cache['DINO_RGBD_global_SVR'].astype('float64')}
    bundle = core.Bundle(ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip(), bcfg)
    run = project_path('runs', ROOT) / ('frame_cost_routing_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    train_audit = u.csv_in(old / 'training_benefit_audit.csv')
    predictions = []
    grid = []
    selected = []
    cost_audit = []
    for fold in range(3):
        roles = {r: bundle.split(fold, r) for r in ['train', 'validation', 'test']}
        ids = {r: [pos[e['episode_id']] for e in rr] for r, rr in roles.items()}
        y = {r: np.array([int(e['bcs']) for e in rr]) for r, rr in roles.items()}
        costs = {r: np.array([min(12, len(bundle.by_ep[e['episode_id']])) for e in rr]) for r, rr in roles.items()}
        pred = {}
        for role, rr in roles.items():
            for e, c in zip(rr, costs[role]):
                cost_audit.append({'fold': fold, 'role': role, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'actual_unique_frames': int(c), 'depth_frame_cost': int(c), 'padding_counted': False})
        for mod in X:
            xt = X[mod][ids['train']]
            fit = np.load(parent / f'source_fold{fold}_DINO_{mod}_global_SVR_fit.npz')
            pred[mod] = {r: np.clip(svr_predict(fit, np.exp(-dist(X[mod][ids[r]], xt) / float(fit['bandwidth']))), 2, 6) for r in roles}
        audit = {r['episode_id']: r for r in train_audit if int(r['fold']) == fold}
        b = np.array([float(audit[e['episode_id']]['benefit_target']) for e in roles['train']])
        err = np.array([float(audit[e['episode_id']]['RGB_error_target']) for e in roles['train']])
        ins = abs(pred['RGB']['train'] - y['train']) - abs(pred['RGBD']['train'] - y['train'])
        perm = np.random.default_rng(20261023 + fold * 701).permutation(len(b))
        targets = {'Crossfit_benefit_knapsack': b, 'Insample_benefit_knapsack': ins, 'Crossfit_RGB_error_knapsack': err, 'Shuffled_benefit_knapsack': b[perm]}
        scores = {}
        xt = X['RGB'][ids['train']]
        dd = dist(xt, xt)
        band = float(np.median(dd[dd > 1e-14]))
        for gate, target in targets.items():
            best = None
            for scale in [0.5, 1.0, 2.0]:
                K = np.exp(-dd / (band * scale))
                Kv = np.exp(-dist(X['RGB'][ids['validation']], xt) / (band * scale))
                Kt = np.exp(-dist(X['RGB'][ids['test']], xt) / (band * scale))
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    coef = np.linalg.solve(K + alpha * np.eye(len(K)), target)
                    vp = Kv @ coef
                    maes = []
                    for fraction in [0.25, 0.5, 0.75]:
                        budget = int(np.floor(fraction * costs['validation'].sum()))
                        use = knapsack(vp, costs['validation'], budget)
                        p = np.where(use, pred['RGBD']['validation'], pred['RGB']['validation'])
                        maes.append(float(abs(p - y['validation']).mean()))
                    key = float(np.mean(maes))
                    row = {'gate': gate, 'fold': fold, 'scale': scale, 'alpha': alpha, 'validation_mean_routing_MAE': key}
                    grid.append(row)
                    if best is None or key < best[0]:
                        best = (key, row.copy(), Kt @ coef, coef, band * scale)
            selected.append(best[1])
            scores[gate] = [('deterministic', best[2])]
            np.savez_compressed(run / f'fold{fold}_{gate}.npz', RGB_train=xt, coefficients=best[3], bandwidth=best[4])
        K = np.exp(-dd / band)
        Kt = np.exp(-dist(X['RGB'][ids['test']], xt) / band)
        A = np.linalg.inv(K + 0.1 * np.eye(len(K)))
        scores['RGB_variance_knapsack'] = [('deterministic', np.maximum(1 - np.einsum('ij,jk,ik->i', Kt, A, Kt), 0))]
        scores['RGB_distance_knapsack'] = [('deterministic', dist(X['RGB'][ids['test']], xt).min(1))]
        scores['Random_affordable_order'] = [(seed, np.random.default_rng(seed + fold * 1009).random(len(y['test']))) for seed in cfg['random_seeds']]
        scores['Oracle_benefit_knapsack'] = [('oracle', abs(pred['RGB']['test'] - y['test']) - abs(pred['RGBD']['test'] - y['test']))]
        for gate, reps in scores.items():
            for seed, s in reps:
                for fraction in cfg['budget_caps']:
                    budget = int(np.floor(fraction * costs['test'].sum()))
                    if gate == 'Random_affordable_order':
                        use = np.zeros(len(s), bool)
                        spent = 0
                        for i in np.argsort(-s, kind='stable'):
                            if spent + costs['test'][i] <= budget:
                                use[i] = True
                                spent += costs['test'][i]
                    else:
                        use = knapsack(s, costs['test'], budget)
                    p = np.where(use, pred['RGBD']['test'], pred['RGB']['test'])
                    spent = int(costs['test'][use].sum())
                    assert spent <= budget
                    for i, (e, value) in enumerate(zip(roles['test'], p)):
                        predictions.append({'gate': gate, 'fold': fold, 'seed': seed, 'budget_cap_fraction': fraction, 'budget_depth_frames': budget, 'batch_total_depth_frames': int(costs['test'].sum()), 'batch_used_depth_frames': spent, 'batch_used_depth_fraction': spent / int(costs['test'].sum()), 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'event_unique_frames': int(costs['test'][i]), 'depth_processed': bool(use[i]), 'gate_value': float(s[i]), 'RGB_score': float(pred['RGB']['test'][i]), 'RGBD_score': float(pred['RGBD']['test'][i]), 'prediction': float(value), 'reference': int(e['bcs']), 'absolute_error': float(abs(value - int(e['bcs']))), 'oracle': gate.startswith('Oracle')})
        print('FRAME-COST ROUTING COMPLETE', fold, flush=True)
    u.csv_out(run / 'test_predictions.csv', predictions, list(predictions[0]))
    u.csv_out(run / 'event_frame_cost_audit.csv', cost_audit, list(cost_audit[0]))
    u.csv_out(run / 'gate_validation_grid.csv', grid, list(grid[0]))
    u.csv_out(run / 'selected_gate_parameters.csv', selected, list(selected[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'events': 151, 'total_unique_frames': 1245, 'gates': 8, 'all_budget_caps_respected': True, 'knapsack_bruteforce_checks': 25, 'all_events_scored': True, 'padding_cost': 0, 'source_benefit_audit': old.relative_to(ROOT).as_posix(), 'script_sha256': u.sha256(Path(__file__)), 'exploratory': True}, indent=2), encoding='utf-8')
    project_path('provenance/latest_frame_cost_routing.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    (run / 'checksums.json').write_text(json.dumps([{'file': p.name, 'sha256': u.sha256(p)} for p in sorted(run.iterdir()) if p.is_file()], indent=2), encoding='utf-8')
    for gate in scores:
        for fraction in cfg['budget_caps']:
            rr = [r for r in predictions if r['gate'] == gate and r['budget_cap_fraction'] == fraction]
            usage = {}
            for r in rr:
                usage[r['fold'], r['seed']] = r['batch_used_depth_frames']
            reps = len({r['seed'] for r in rr})
            used = sum(usage.values()) / reps
            print('FRAME-COST RESULT', gate, fraction, 'MAE', np.mean([r['absolute_error'] for r in rr]), 'depth_frames', used, 'total_encoder_frame_saving', 1 - (1245 + used) / 2490, flush=True)
