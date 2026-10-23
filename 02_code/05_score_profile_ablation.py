import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Low-dimensional, animal-cross-fitted decision profiles for depth refinement."""
import json
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
rep = import_module('03_grouped_validation')
u = rep.u
ROOT = rep.ROOT
fd = rep.fd
probe = rep.probe

def profile(p, n):
    a = p[:, 0]
    b = p[:, 1]
    d = b - a
    return np.column_stack([a, b, d, abs(d), d * d, np.minimum(a, b), np.maximum(a, b), n])

def main():
    threadpool_limits(limits=2)
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/decision_profile_v1.json', ROOT).read_text())
    source = ROOT / project_path('provenance/latest_repeated_validation.txt', ROOT).read_text().strip()
    mc = ROOT / project_path('provenance/latest_multilevel_depth.txt', ROOT).read_text().strip()
    f = np.load(mc / 'temporal_spread_features.npz')
    epids = list(f['episode_ids'])
    pos = {e: i for i, e in enumerate(epids)}
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    rgb = np.load(parent / 'source_episode_features.npz')['DINO_RGB_global_SVR'].astype('float64')
    X = {0: rgb, **{k: f['frames_' + str(k)].astype('float64') for k in [1, 3, 6, 12]}}
    split = u.csv_in(source / 'group_splits.csv')
    held = u.csv_in(source / 'crossfit_teacher_audit.csv')
    run = project_path('decision_runs', ROOT) / ('decision_profile_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    rows = []
    grids = []
    selected = []
    for repeat in range(3):
        for fold in range(3):
            roles = {r: [z for z in split if int(z['repeat']) == repeat and int(z['fold']) == fold and (z['role'] == r)] for r in ['train', 'validation', 'test']}
            ids = {r: [pos[z['episode_id']] for z in zz] for r, zz in roles.items()}
            y = {r: np.array([int(z['reference']) for z in zz]) for r, zz in roles.items()}
            cc = {r: f['costs'][ix] for r, ix in ids.items()}
            pp = {r: np.zeros((len(ix), 5)) for r, ix in ids.items()}
            modeldir = run / f'repeat{repeat}_fold{fold}'
            modeldir.mkdir()
            for col, k in enumerate([0, 1, 3, 6, 12]):
                fit = np.load(source / f'repeat{repeat}_fold{fold}' / f'scorer_frames{k}.npz')
                for r in roles:
                    pp[r][:, col] = rep.predict(fit, X[k][ids[r]])
            look = {(z['episode_id'], int(z['depth_level'])): float(z['held_animal_prediction']) for z in held if int(z['repeat']) == repeat and int(z['fold']) == fold}
            hp = np.array([[look[z['episode_id'], k] for k in [0, 1, 3, 6, 12]] for z in roles['train']])
            benefit = abs(hp[:, 0, None] - y['train'][:, None]) - abs(hp - y['train'][:, None])
            ins = abs(pp['train'][:, 0, None] - y['train'][:, None]) - abs(pp['train'] - y['train'][:, None])
            obs = {r: profile(pp[r], cc[r][:, -1]) for r in roles}
            oof = profile(hp, cc['train'][:, -1])
            perm = np.random.default_rng(231 + repeat * 17 + fold).permutation(len(hp))
            for method in cfg['variants']:
                xt = obs['train'].copy() if method == 'Score_profile_insample_features' else oof.copy()
                other = {r: obs[r].copy() for r in ['validation', 'test']}
                if method == 'Score_profile_no_count':
                    xt = xt[:, :-1]
                    other = {r: v[:, :-1] for r, v in other.items()}
                mu = xt.mean(0)
                sd = xt.std(0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                xt = (xt - mu) / sd
                other = {r: (v - mu) / sd for r, v in other.items()}
                target = ins if method == 'Score_profile_insample_targets' else benefit[perm] if method == 'Score_profile_shuffled' else benefit
                best = None
                for scale in [1.0] if method == 'Score_profile_linear' else [0.5, 1.0, 2.0]:
                    if method == 'Score_profile_linear':
                        Z = np.column_stack([np.ones(len(xt)), xt])
                        Zv = np.column_stack([np.ones(len(other['validation'])), other['validation']])
                        Zt = np.column_stack([np.ones(len(other['test'])), other['test']])
                    else:
                        dd = fd.dist(xt, xt)
                        band = float(np.median(dd[dd > 1e-14])) * scale
                        K = np.exp(-dd / band)
                        Kv = np.exp(-fd.dist(other['validation'], xt) / band)
                        Kt = np.exp(-fd.dist(other['test'], xt) / band)
                    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                        if method == 'Score_profile_linear':
                            penalty = alpha * np.eye(Z.shape[1])
                            penalty[0, 0] = 0
                            coef = np.linalg.solve(Z.T @ Z + penalty, Z.T @ target)
                            vp = Zv @ coef
                            tp = Zt @ coef
                        else:
                            coef = np.linalg.solve(K + alpha * np.eye(len(K)), target)
                            vp = Kv @ coef
                            tp = Kt @ coef
                        maes = []
                        for cap in [0.25, 0.5, 0.75]:
                            ac, paid, _ = probe.decisions(vp, cc['validation'], int(np.floor(cap * cc['validation'][:, -1].sum())))
                            p = pp['validation'][np.arange(len(ac)), ac]
                            maes.append(float(abs(p - y['validation']).mean()))
                        key = float(np.mean(maes))
                        record = {'repeat': repeat, 'fold': fold, 'method': method, 'scale': scale, 'alpha': alpha, 'validation_MAE': key}
                        grids.append(record)
                        if best is None or key < best[0]:
                            best = (key, record.copy(), tp.copy(), coef.copy(), None if method == 'Score_profile_linear' else band)
                selected.append(best[1])
                np.savez_compressed(modeldir / (method + '.npz'), train_features=xt, coefficients=best[3], bandwidth=np.nan if best[4] is None else best[4], feature_mean=mu, feature_sd=sd, linear=method == 'Score_profile_linear')
                for cap in cfg['budget_caps']:
                    B = int(np.floor(cap * cc['test'][:, -1].sum()))
                    ac, paid, observed = probe.decisions(best[2], cc['test'], B)
                    p = pp['test'][np.arange(len(ac)), ac]
                    for i, z in enumerate(roles['test']):
                        rows.append({'repeat': repeat, 'fold': fold, 'method': method, 'seed': 'deterministic', 'budget_cap': cap, 'episode_id': z['episode_id'], 'animal_group': z['animal_group'], 'reference': int(z['reference']), 'prediction': float(p[i]), 'absolute_error': float(abs(p[i] - y['test'][i])), 'depth_level': [0, 1, 3, 6, 12][ac[i]], 'paid_depth_frames': int(paid[i]), 'full_depth_frames': int(cc['test'][i, -1]), 'batch_budget': B, 'batch_paid_frames': int(paid.sum()), 'budget_feasible': True, 'all_events_scored': True})
            print('DECISION PROFILE COMPLETE', repeat, fold, flush=True)
    for name, rr in [('test_predictions.csv', rows), ('validation_grid.csv', grids), ('selected_parameters.csv', selected)]:
        u.csv_out(run / name, rr, list(rr[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_three_repeats_retained': True, 'all_budgets_respected': True, 'exploratory': True, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_decision_profile.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    for method in cfg['variants']:
        for cap in cfg['budget_caps']:
            zz = [z for z in rows if z['method'] == method and z['budget_cap'] == cap]
            print('PROFILE RESULT', method, cap, 'MAE', np.mean([z['absolute_error'] for z in zz]), 'depth_frames', sum((z['paid_depth_frames'] for z in zz)) / 3, flush=True)
    print('RUN', run, flush=True)
if __name__ == '__main__':
    main()
