import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Refit the complete scoring and allocation pipeline under repeated animal splits."""
import json, math
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
fd = import_module('19_foundation_pooling')
u = fd.u
ROOT = fd.ROOT
prep = fd.prep
rt = import_module('27_frame_cost_routing')
ml = import_module('28_multilevel_depth')
probe = import_module('33_probe_refine')

def predict(fit, x):
    return np.clip(rt.svr_predict(fit, np.exp(-fd.dist(x, fit['train_features']) / float(fit['bandwidth']))), 2, 6)

def fit_base(x, y, v, vy, grid, log, tag):
    dd = fd.dist(x, x)
    median = float(np.median(dd[dd > 1e-14]))
    best = None
    vd = fd.dist(v, x)
    for scale in grid['bandwidth_scale']:
        K = np.exp(-dd / (median * scale))
        Kv = np.exp(-vd / (median * scale))
        for alpha in grid['alpha']:
            model = SVR(kernel='precomputed', C=1 / alpha, epsilon=0.1).fit(K, y)
            vp = np.clip(model.predict(Kv), 2, 6)
            mae = float(abs(vp - vy).mean())
            log.append({**tag, 'scale': scale, 'alpha': alpha, 'validation_MAE': mae})
            if best is None or mae < best[0]:
                best = (mae, scale, alpha, model)
    model = best[3]
    return {'train_features': x, 'support': model.support_, 'dual_coef': model.dual_coef_, 'intercept': model.intercept_, 'bandwidth': np.array(median * best[1]), 'scale': np.array(best[1]), 'alpha': np.array(best[2])}

def frozen_teacher(x, y, held, fit):
    dd = fd.dist(x, x)
    b = float(np.median(dd[dd > 1e-14])) * float(fit['scale'])
    model = SVR(kernel='precomputed', C=1 / float(fit['alpha']), epsilon=0.1).fit(np.exp(-dd / b), y)
    return np.clip(model.predict(np.exp(-fd.dist(held, x) / b)), 2, 6)

def make_outer(eps, seed, candidates):
    labels = np.array([int(e['bcs']) for e in eps])
    groups = np.array([e['animal_group'] for e in eps])
    tot = np.array([(labels == c).sum() for c in range(2, 7)])
    best = None
    logs = []
    for j in range(candidates):
        ss = seed + j * 997
        folds = np.zeros(len(eps), int)
        for f, (_, test) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=ss).split(np.zeros((len(eps), 1)), labels, groups)):
            folds[test] = f
        counts = np.array([[sum((folds == f) & (labels == c)) for c in range(2, 7)] for f in range(3)])
        score = (int((counts == 0).sum()), float(np.square(counts / tot[None, :] - 1 / 3).sum()))
        logs.append({'repeat_seed': seed, 'candidate': j, 'candidate_seed': ss, 'missing_test_classes': score[0], 'class_imbalance': score[1]})
        if best is None or score < best[0]:
            best = (score, folds.copy(), ss)
    return (best[1], best[2], logs)

def allocate_binary(s, cost, B, greedy=False):
    if greedy:
        use = np.zeros(len(s), bool)
        spent = 0
        for i in np.argsort(-s, kind='stable'):
            if s[i] > 0 and spent + cost[i] <= B:
                use[i] = True
                spent += cost[i]
        return (use.astype(int) * 4, cost * use)
    use = rt.knapsack(s, cost, B)
    return (use.astype(int) * 4, cost * use)

def main():
    threadpool_limits(limits=2)
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/repeated_group_validation_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    mult = ROOT / project_path('provenance/latest_multilevel_depth.txt', ROOT).read_text().strip()
    mc = np.load(mult / 'temporal_spread_features.npz')
    epids = list(mc['episode_ids'])
    pos = {ep: i for i, ep in enumerate(epids)}
    RGB = np.load(parent / 'source_episode_features.npz')['DINO_RGB_global_SVR'].astype('float64')
    X = {0: RGB, **{k: mc['frames_' + str(k)].astype('float64') for k in [1, 3, 6, 12]}}
    allcost = mc['costs']
    prepared = ROOT / project_path('provenance/latest_prepared.txt', ROOT).read_text().strip()
    eps = sorted(u.csv_in(prepared / 'episodes.csv'), key=lambda e: e['episode_id'])
    assert [e['episode_id'] for e in eps] == epids
    run = project_path('repeated_runs', ROOT) / ('repeated_groups_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    splitrows = []
    splitlogs = []
    splits = {}
    for repeat, seed in enumerate(cfg['repeat_seeds']):
        ff, chosen, logs = make_outer(eps, seed, cfg['outer_candidates'])
        splitlogs.extend(logs)
        ee = [{**e, 'outer_fold': str(f)} for e, f in zip(eps, ff)]
        for fold in range(3):
            ic = {**cfg['inner_validation'], 'seed': cfg['inner_validation']['seed'] + repeat * 101}
            (score, vg), nval = prep.choose_inner_groups(ee, fold, ic)
            roles = {r: [] for r in ['train', 'validation', 'test']}
            for i, e in enumerate(ee):
                role = 'test' if int(e['outer_fold']) == fold else 'validation' if e['animal_group'] in vg else 'train'
                roles[role].append(i)
                splitrows.append({'repeat': repeat, 'repeat_seed': seed, 'selected_outer_seed': chosen, 'fold': fold, 'role': role, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference': int(e['bcs'])})
            gg = {r: {eps[i]['animal_group'] for i in ids} for r, ids in roles.items()}
            assert not (gg['train'] & gg['validation'] or gg['train'] & gg['test'] or gg['validation'] & gg['test'])
            splits[repeat, fold] = roles
    u.csv_out(run / 'group_splits.csv', splitrows, list(splitrows[0]))
    u.csv_out(run / 'split_candidates.csv', splitlogs, list(splitlogs[0]))
    predictions = []
    basegrid = []
    gategrid = []
    selected = []
    heldaudit = []
    for (repeat, fold), ids in splits.items():
        modeldir = run / f'repeat{repeat}_fold{fold}'
        modeldir.mkdir()
        y = {r: np.array([int(eps[i]['bcs']) for i in ix]) for r, ix in ids.items()}
        g = np.array([eps[i]['animal_group'] for i in ids['train']])
        cc = {r: allcost[ix] for r, ix in ids.items()}
        pp = {r: np.zeros((len(ix), 5)) for r, ix in ids.items()}
        held = np.zeros_like(pp['train'])
        frozen = np.zeros((len(g), 2))
        fits = {}
        for col, k in enumerate([0, 1, 3, 6, 12]):
            xx = X[k][ids['train']]
            fit = fit_base(xx, y['train'], X[k][ids['validation']], y['validation'], cfg['base_grid'], basegrid, {'repeat': repeat, 'fold': fold, 'depth_level': k, 'held_group': 'FULL_TRAIN'})
            fits[k] = fit
            np.savez_compressed(modeldir / f'scorer_frames{k}.npz', **fit)
            for r in ids:
                pp[r][:, col] = predict(fit, X[k][ids[r]])
            for group in sorted(set(g)):
                tr = np.flatnonzero(g != group)
                ho = np.flatnonzero(g == group)
                assert not set(g[tr]) & set(g[ho])
                sub = fit_base(xx[tr], y['train'][tr], X[k][ids['validation']], y['validation'], cfg['base_grid'], basegrid, {'repeat': repeat, 'fold': fold, 'depth_level': k, 'held_group': group})
                held[ho, col] = predict(sub, xx[ho])
                if k in [0, 12]:
                    frozen[ho, 0 if k == 0 else 1] = frozen_teacher(xx[tr], y['train'][tr], xx[ho], fit)
        benefit = abs(held[:, 0, None] - y['train'][:, None]) - abs(held - y['train'][:, None])
        ins = abs(pp['train'][:, 0, None] - y['train'][:, None]) - abs(pp['train'] - y['train'][:, None])
        frozenbenefit = abs(frozen[:, 0] - y['train']) - abs(frozen[:, 1] - y['train'])
        err = abs(held[:, 0] - y['train'])
        perm = np.random.default_rng(981 + repeat * 113 + fold).permutation(len(g))
        scoremodels = {}
        perrole = {}
        for i, index in enumerate(ids['train']):
            for col, k in enumerate([0, 1, 3, 6, 12]):
                heldaudit.append({'repeat': repeat, 'fold': fold, 'episode_id': eps[index]['episode_id'], 'animal_group': g[i], 'depth_level': k, 'held_animal_prediction': float(held[i, col]), 'benefit_target': float(benefit[i, col]), 'held_animal_excluded_from_fit_and_HPO': True})
        gates = ['Binary_strict_crossfit', 'Binary_frozen_teacher', 'Binary_insample', 'Binary_shuffled', 'Binary_RGB_error', 'Multilevel_strict_crossfit', 'Probe_strict_crossfit']
        for method in gates:
            multi = method.startswith(('Multilevel', 'Probe'))
            obs = X[1] if method.startswith('Probe') else RGB
            target = benefit if multi else ins[:, -1] if method == 'Binary_insample' else benefit[perm, -1] if method == 'Binary_shuffled' else err if method == 'Binary_RGB_error' else frozenbenefit if method == 'Binary_frozen_teacher' else benefit[:, -1]
            xt = obs[ids['train']]
            dd = fd.dist(xt, xt)
            band = float(np.median(dd[dd > 1e-14]))
            best = None
            for scale in cfg['gate_grid']['bandwidth_scale']:
                K = np.exp(-dd / (band * scale))
                Kv = np.exp(-fd.dist(obs[ids['validation']], xt) / (band * scale))
                Kt = np.exp(-fd.dist(obs[ids['test']], xt) / (band * scale))
                for alpha in cfg['gate_grid']['alpha']:
                    coef = np.linalg.solve(K + alpha * np.eye(len(K)), target)
                    v = Kv @ coef
                    maes = []
                    for cap in [0.25, 0.5, 0.75]:
                        B = int(np.floor(cap * cc['validation'][:, -1].sum()))
                        if method.startswith('Probe'):
                            ac, paid, _ = probe.decisions(v, cc['validation'], B)
                        elif multi:
                            ac = ml.allocate(v, cc['validation'], B)
                        else:
                            ac, paid = allocate_binary(v, cc['validation'][:, -1], B)
                        p = pp['validation'][np.arange(len(ac)), ac]
                        maes.append(float(abs(p - y['validation']).mean()))
                    key = float(np.mean(maes))
                    row = {'repeat': repeat, 'fold': fold, 'method': method, 'scale': scale, 'alpha': alpha, 'validation_MAE': key}
                    gategrid.append(row)
                    if best is None or key < best[0]:
                        best = (key, row.copy(), Kt @ coef, coef, band * scale)
            selected.append(best[1])
            scoremodels[method] = best[2]
            np.savez_compressed(modeldir / (method + '_gate.npz'), train_features=xt, coefficients=best[3], bandwidth=best[4])
        scoremodels['Binary_greedy'] = scoremodels['Binary_strict_crossfit']
        for method in cfg['methods']:
            for cap in cfg['budget_caps']:
                B = int(np.floor(cap * cc['test'][:, -1].sum()))
                n = len(ids['test'])
                if method in ['RGB_only', 'RGBD_all', 'Fixed_depth_1', 'Fixed_depth_3', 'Fixed_depth_6']:
                    col = {'RGB_only': 0, 'Fixed_depth_1': 1, 'Fixed_depth_3': 2, 'Fixed_depth_6': 3, 'RGBD_all': 4}[method]
                    ac = np.full(n, col, int)
                    paid = cc['test'][:, col]
                elif method.startswith('Probe'):
                    ac, paid, _ = probe.decisions(scoremodels[method], cc['test'], B)
                elif method.startswith('Multilevel'):
                    ac = ml.allocate(scoremodels[method], cc['test'], B)
                    paid = cc['test'][np.arange(n), ac]
                else:
                    ac, paid = allocate_binary(scoremodels[method], cc['test'][:, -1], B, method == 'Binary_greedy')
                feasible = bool(paid.sum() <= B)
                p = pp['test'][np.arange(n), ac]
                if method in scoremodels:
                    assert feasible
                for i, index in enumerate(ids['test']):
                    predictions.append({'repeat': repeat, 'repeat_seed': cfg['repeat_seeds'][repeat], 'fold': fold, 'method': method, 'budget_cap': cap, 'episode_id': eps[index]['episode_id'], 'animal_group': eps[index]['animal_group'], 'reference': int(eps[index]['bcs']), 'prediction': float(p[i]), 'absolute_error': float(abs(p[i] - int(eps[index]['bcs']))), 'depth_level': [0, 1, 3, 6, 12][ac[i]], 'paid_depth_frames': int(paid[i]), 'full_depth_frames': int(cc['test'][i, -1]), 'batch_budget': B, 'batch_paid_frames': int(paid.sum()), 'budget_feasible': feasible, 'all_events_scored': True})
        u.csv_out(run / 'test_predictions_partial.csv', predictions, list(predictions[0]))
        print('REPEATED VALIDATION COMPLETE', repeat, fold, flush=True)
    for name, rr in [('test_predictions.csv', predictions), ('base_validation_grid.csv', basegrid), ('gate_validation_grid.csv', gategrid), ('selected_gates.csv', selected), ('crossfit_teacher_audit.csv', heldaudit)]:
        u.csv_out(run / name, rr, list(rr[0]))
    info = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'repeats': 3, 'outer_fits': 9, 'source_events_per_repeat': 151, 'animals_per_repeat': 52, 'models_group_disjoint': True, 'strict_teacher_HPO_excludes_target_animal': True, 'all_methods_retained': True, 'exploratory_same_source_not_external': True, 'script_sha256': u.sha256(Path(__file__))}
    (run / 'COMPLETE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    project_path('provenance/latest_repeated_validation.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    print('RUN', run, flush=True)
    for method in cfg['methods']:
        for cap in cfg['budget_caps']:
            rr = [r for r in predictions if r['method'] == method and r['budget_cap'] == cap]
            print('REPEATED RESULT', method, cap, 'MAE', np.mean([r['absolute_error'] for r in rr]), 'depth_frames', sum((r['paid_depth_frames'] for r in rr)) / 3, flush=True)
if __name__ == '__main__':
    main()
