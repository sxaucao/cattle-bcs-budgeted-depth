import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Cost-matched uniform frames, random, uncertainty and imputation controls."""
import json
import argparse
from datetime import datetime, timezone
from importlib import import_module
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
rep = import_module('03_grouped_validation')
fd = rep.fd
u = fd.u
ROOT = fd.ROOT
rt = rep.rt
ml = rep.ml

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precomputed-levels',type=Path,help='Repository-relative audited all-frame-level feature archive');args=parser.parse_args()
    threadpool_limits(limits=2)
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/matched_budget_controls_v1.json', ROOT).read_text())
    rcfg = json.loads(project_path('configs/repeated_group_validation_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    repeat = ROOT / project_path('provenance/latest_repeated_validation.txt', ROOT).read_text().strip()
    mult = ROOT / project_path('provenance/latest_multilevel_depth.txt', ROOT).read_text().strip()
    mc = np.load(mult / 'temporal_spread_features.npz')
    epids = list(mc['episode_ids'])
    pos = {e: i for i, e in enumerate(epids)}
    RGB = np.load(parent / 'source_episode_features.npz')['DINO_RGB_global_SVR'].astype('float64')
    if args.precomputed_levels is not None:
        feature_file=project_path(args.precomputed_levels)
        cached=np.load(feature_file,allow_pickle=False)
        assert list(cached['episode_ids'])==epids,'Feature episode order mismatch'
        costmatrix=cached['costs'].astype(int)
        features={k:cached['frames_'+str(k)].astype('float64') for k in range(13)}
        assert costmatrix.shape==(151,13) and np.all(costmatrix[:,0]==0)
        assert np.array_equal(features[0],RGB)
        for k in [1,3,6,12]:
            assert np.max(abs(features[k]-mc['frames_'+str(k)]))<2e-7
        cached_source={'path':relative_record(feature_file),'SHA256':u.sha256(feature_file)}
    else:
        manifest = u.csv_in(parent / 'source_Depth_tokens/input_manifest.csv')
        by = defaultdict(list)
        for i, r in enumerate(manifest):
            by[r['episode_id']].append(i)
        folder = parent / 'source_Depth_tokens'
        tokens = np.load(folder / 'tokens.npy', mmap_mode='r')
        aux = np.load(folder / 'cls_and_masks.npz')
        cls = fd.norm(aux['cls'])
        patch = np.stack([fd.norm(fd.norm(np.asarray(tokens[i], dtype='float32'))[aux['valid'][i]].mean(0)) for i in range(len(tokens))])
        features = {0: RGB}
        costmatrix = np.zeros((151, 13), int)
        for k in range(1, 13):
            vals = []
            for ep in epids:
                rr = np.array(by[ep])
                order = ml.order_frames(np.zeros((len(rr), 1)), 'temporal_spread')
                chosen = rr[order[:min(k, len(rr))]]
                d = np.concatenate([fd.norm(cls[chosen].mean(0)), fd.norm(patch[chosen].mean(0))]) / np.sqrt(2)
                vals.append(np.concatenate([RGB[pos[ep]], d]) / np.sqrt(2))
                costmatrix[pos[ep], k] = len(chosen)
            features[k] = np.asarray(vals, dtype='float64')
            if k in [1, 3, 6, 12]:
                assert np.max(abs(features[k] - mc['frames_' + str(k)])) < 2e-07
        cached_source=None
    run = project_path('matched_runs', ROOT) / ('matched_controls_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run/'feature_source.json').write_text(json.dumps({'precomputed':cached_source,'all_scorers_refitted_inside_grouped_splits':True},indent=2),encoding='utf-8')
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    np.savez_compressed(run / 'all_temporal_levels_features.npz', episode_ids=np.array(epids), costs=costmatrix, **{'frames_' + str(k): v for k, v in features.items()})
    splits = u.csv_in(repeat / 'group_splits.csv')
    rows = []
    grids = []
    fixedrows = []
    selected = []
    for repetition in range(3):
        for fold in range(3):
            roles = {r: [z for z in splits if int(z['repeat']) == repetition and int(z['fold']) == fold and (z['role'] == r)] for r in ['train', 'validation', 'test']}
            ids = {r: [pos[z['episode_id']] for z in rr] for r, rr in roles.items()}
            y = {r: np.array([int(z['reference']) for z in rr]) for r, rr in roles.items()}
            pp = {r: np.zeros((len(ix), 13)) for r, ix in ids.items()}
            cc = {r: costmatrix[ix] for r, ix in ids.items()}
            modeldir = run / f'repeat{repetition}_fold{fold}'
            modeldir.mkdir()
            fits = {}
            for k in range(13):
                old = repeat / f'repeat{repetition}_fold{fold}' / f'scorer_frames{k}.npz'
                if old.exists():
                    fit = np.load(old)
                else:
                    fit = rep.fit_base(features[k][ids['train']], y['train'], features[k][ids['validation']], y['validation'], rcfg['base_grid'], grids, {'repeat': repetition, 'fold': fold, 'depth_level': k, 'held_group': 'FULL_TRAIN'})
                    np.savez_compressed(modeldir / f'scorer_frames{k}.npz', **fit)
                fits[k] = fit
                for r in roles:
                    pp[r][:, k] = rep.predict(fit, features[k][ids[r]])
                for i, z in enumerate(roles['test']):
                    fixedrows.append({'repeat': repetition, 'fold': fold, 'depth_level': k, 'episode_id': z['episode_id'], 'animal_group': z['animal_group'], 'reference': int(z['reference']), 'prediction': float(pp['test'][i, k]), 'absolute_error': float(abs(pp['test'][i, k] - y['test'][i])), 'paid_depth_frames': int(cc['test'][i, k])})
            xt = RGB[ids['train']]
            dd = fd.dist(xt, xt)
            band = float(np.median(dd[dd > 1e-14]))
            K = np.exp(-dd / band)
            Kt = np.exp(-fd.dist(RGB[ids['test']], xt) / band)
            variance = np.maximum(1 - np.einsum('ij,jk,ik->i', Kt, np.linalg.inv(K + 0.1 * np.eye(len(K))), Kt), 0)
            best = None
            depth = features[12][:, 768:] * np.sqrt(2)
            for scale in [0.5, 1.0, 2.0]:
                K = np.exp(-dd / (band * scale))
                Kv = np.exp(-fd.dist(RGB[ids['validation']], xt) / (band * scale))
                Kt = np.exp(-fd.dist(RGB[ids['test']], xt) / (band * scale))
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    coef = np.linalg.solve(K + alpha * np.eye(len(K)), depth[ids['train']])
                    im = {r: (Kv if r == 'validation' else Kt) @ coef for r in ['validation', 'test']}
                    score = {}
                    for r in im:
                        xx = np.concatenate([RGB[ids[r]], fd.norm(im[r])], 1) / np.sqrt(2)
                        score[r] = abs(rep.predict(fits[12], xx) - pp[r][:, 0])
                    maes = []
                    for cap in [0.25, 0.5, 0.75]:
                        use = rt.knapsack(score['validation'], cc['validation'][:, -1], int(np.floor(cap * cc['validation'][:, -1].sum())))
                        p = np.where(use, pp['validation'][:, -1], pp['validation'][:, 0])
                        maes.append(float(abs(p - y['validation']).mean()))
                    key = float(np.mean(maes))
                    if best is None or key < best[0]:
                        best = (key, scale, alpha, coef.copy(), score['test'].copy())
            selected.append({'repeat': repetition, 'fold': fold, 'method': 'Imputed_mean_shift', 'scale': best[1], 'alpha': best[2], 'validation_MAE': best[0]})
            np.savez_compressed(modeldir / 'imputer.npz', RGB_train=xt, coefficients=best[3], bandwidth=band * best[1])
            for method in cfg['controls']:
                seeds = cfg['seeds'] if method == 'Random_affordable_order' else ['deterministic']
                for seed in seeds:
                    random = np.random.default_rng(int(seed) + fold * 1009 + repetition * 7).random(len(ids['test'])) if method == 'Random_affordable_order' else None
                    for cap in cfg['budget_caps']:
                        B = int(np.floor(cap * cc['test'][:, -1].sum()))
                        n = len(ids['test'])
                        if method.startswith(('Uniform', 'Validation')):
                            feasible = [k for k in range(13) if cc['test'][:, k].sum() <= B]
                            if method == 'Uniform_max_affordable':
                                k = max(feasible)
                            else:
                                V = int(np.floor(cap * cc['validation'][:, -1].sum()))
                                eligible = [k for k in range(13) if cc['validation'][:, k].sum() <= V]
                                pick = min(eligible, key=lambda k: float(abs(pp['validation'][:, k] - y['validation']).mean()))
                                k = max((k for k in feasible if k <= pick))
                            level = np.full(n, k, int)
                            p = pp['test'][:, k]
                            paid = cc['test'][:, k]
                        else:
                            s = random if method == 'Random_affordable_order' else variance if method == 'RGB_kernel_variance' else best[4]
                            if method == 'Random_affordable_order':
                                use = np.zeros(n, bool)
                                spent = 0
                                for i in np.argsort(-s, kind='stable'):
                                    if spent + cc['test'][i, -1] <= B:
                                        use[i] = True
                                        spent += int(cc['test'][i, -1])
                            else:
                                use = rt.knapsack(s, cc['test'][:, -1], B)
                            level = use.astype(int) * 12
                            paid = cc['test'][:, -1] * use
                            p = np.where(use, pp['test'][:, -1], pp['test'][:, 0])
                        assert paid.sum() <= B
                        for i, z in enumerate(roles['test']):
                            rows.append({'repeat': repetition, 'fold': fold, 'method': method, 'seed': seed, 'budget_cap': cap, 'episode_id': z['episode_id'], 'animal_group': z['animal_group'], 'reference': int(z['reference']), 'prediction': float(p[i]), 'absolute_error': float(abs(p[i] - y['test'][i])), 'depth_level': int(level[i]), 'paid_depth_frames': int(paid[i]), 'full_depth_frames': int(cc['test'][i, -1]), 'batch_budget': B, 'batch_paid_frames': int(paid.sum()), 'budget_feasible': True, 'all_events_scored': True})
            print('MATCHED CONTROLS COMPLETE', repetition, fold, flush=True)
    for name, rr in [('test_predictions.csv', rows), ('fixed_level_test_predictions.csv', fixedrows), ('base_validation_grid.csv', grids), ('selected_imputers.csv', selected)]:
        u.csv_out(run / name, rr, list(rr[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'repeats': 3, 'folds': 3, 'uniform_levels': 13, 'all_budgets_respected': True, 'exploratory': True, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_matched_controls.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    for method in cfg['controls']:
        for cap in cfg['budget_caps']:
            rr = [r for r in rows if r['method'] == method and r['budget_cap'] == cap]
            seeds = 10 if method == 'Random_affordable_order' else 1
            print('MATCHED RESULT', method, cap, 'MAE', np.mean([r['absolute_error'] for r in rr]), 'depth_frames', sum((r['paid_depth_frames'] for r in rr)) / 3 / seeds, flush=True)
    print('RUN', run, flush=True)
if __name__ == '__main__':
    main()
