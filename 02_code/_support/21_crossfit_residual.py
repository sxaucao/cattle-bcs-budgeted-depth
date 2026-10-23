from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Cross-animal residual fusion to test complementarity without in-sample errors."""
import json
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
import numpy as np
from sklearn.svm import SVR
core = import_module('07_train_experiments')
u = core.u
ROOT = core.ROOT

def dist(x, z):
    return np.maximum(np.square(x).sum(1)[:, None] + np.square(z).sum(1)[None, :] - 2 * x @ z.T, 0)

def svr_predict(fit, K):
    sv = fit['support'].astype(int)
    return K[:, sv] @ fit['dual_coef'].reshape(-1) + float(fit['intercept'].reshape(-1)[0])

def main():
    import_module('00_archive_sources').archive()
    cfg = json.loads(project_path('configs/crossfit_residual_v1.json', ROOT).read_text())
    bcfg = json.loads(project_path('configs/formal_experiment_v1.json', ROOT).read_text())
    parent = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    cache = np.load(parent / 'source_episode_features.npz')
    epids = list(cache['episode_ids'])
    pos = {ep: i for i, ep in enumerate(epids)}
    Xr = cache['DINO_RGB_global_SVR'].astype('float64')
    Xd = cache['DINO_Depth_global_SVR'].astype('float64')
    Xj = np.concatenate([Xr, Xd], 1) / np.sqrt(2)
    bundle = core.Bundle(ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip(), bcfg)
    run = project_path('runs', ROOT) / ('crossfit_residual_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    selections = u.csv_in(parent / 'source_selected_parameters.csv')
    preds = []
    grid = []
    selected = []
    residual_rows = []
    for fold in range(3):
        roles = {r: bundle.split(fold, r) for r in ['train', 'validation', 'test']}
        ids = {r: [pos[e['episode_id']] for e in rr] for r, rr in roles.items()}
        y = {r: np.array([int(e['bcs']) for e in rr]) for r, rr in roles.items()}
        groups = np.array([e['animal_group'] for e in roles['train']])
        sel = [r for r in selections if r['cohort'] == 'source' and r['variant'] == 'DINO_Depth_global_SVR' and (int(r['fold']) == fold)][0]
        basefit = np.load(parent / f'source_fold{fold}_DINO_Depth_global_SVR_fit.npz')
        xt = Xd[ids['train']]
        band = float(basefit['bandwidth'])
        base = {r: np.clip(svr_predict(basefit, np.exp(-dist(Xd[ids[r]], xt) / band)), 2, 6) for r in roles}
        held = np.zeros(len(xt))
        for g in sorted(set(groups)):
            tr = np.flatnonzero(groups != g)
            ho = np.flatnonzero(groups == g)
            dd = dist(xt[tr], xt[tr])
            subband = float(np.median(dd[dd > 1e-14])) * float(sel['scale'])
            K = np.exp(-dd / subband)
            Kh = np.exp(-dist(xt[ho], xt[tr]) / subband)
            model = SVR(kernel='precomputed', C=1 / float(sel['alpha']), epsilon=0.1).fit(K, y['train'][tr])
            held[ho] = np.clip(model.predict(Kh), 2, 6)
            assert not set(groups[tr]) & set(groups[ho])
        residual = y['train'] - held
        insample = y['train'] - base['train']
        perm = np.random.default_rng(20261023 + fold * 137).permutation(len(xt))
        for i, e in enumerate(roles['train']):
            residual_rows.append({'fold': fold, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference_BCS': int(e['bcs']), 'held_animal_base_prediction': float(held[i]), 'crossfit_residual': float(residual[i]), 'insample_base_prediction': float(base['train'][i]), 'insample_residual': float(insample[i]), 'shuffled_residual_donor': roles['train'][perm[i]]['episode_id']})
        for arm in cfg['arms']:
            if arm == 'Depth_validation_bias_calibration':
                shift = float(np.median(y['validation'] - base['validation']))
                prediction = np.clip(base['test'] + shift, 2, 6)
                record = {'variant': arm, 'fold': fold, 'alpha': '', 'scale': '', 'lambda': '', 'validation_MAE': float(abs(np.clip(base['validation'] + shift, 2, 6) - y['validation']).mean()), 'bias_shift': shift}
                selected.append(record)
            else:
                x = Xr if arm == 'Crossfit_RGB_residual_SVR' else Xd if arm == 'Crossfit_Depth_residual_SVR' else Xj
                xx = x[ids['train']]
                dd = dist(xx, xx)
                b = float(np.median(dd[dd > 1e-14]))
                best = None
                target = insample if arm == 'Insample_RGBD_residual_SVR' else residual[perm] if arm == 'Shuffled_crossfit_RGBD_residual_SVR' else residual
                for scale in cfg['correction_scale_grid']:
                    if arm == 'Crossfit_product_residual_SVR':
                        Ks = {r: np.ones((len(ids[r]), len(ids['train']))) for r in roles}
                        for fm in [Xr, Xd]:
                            tt = fm[ids['train']]
                            ds = dist(tt, tt)
                            bb = float(np.median(ds[ds > 1e-14]))
                            for r in roles:
                                Ks[r] *= np.exp(-0.5 * dist(fm[ids[r]], tt) / (bb * scale))
                    else:
                        Ks = {r: np.exp(-dist(x[ids[r]], xx) / (b * scale)) for r in roles}
                    for a in cfg['correction_alpha_grid']:
                        model = SVR(kernel='precomputed', C=1 / a, epsilon=0.1).fit(Ks['train'], target)
                        cor = {r: model.predict(Ks[r]) for r in ['validation', 'test']}
                        for lam in cfg['lambda_grid']:
                            v = np.clip(base['validation'] + lam * cor['validation'], 2, 6)
                            mae = float(abs(v - y['validation']).mean())
                            row = {'variant': arm, 'fold': fold, 'alpha': a, 'scale': scale, 'lambda': lam, 'validation_MAE': mae, 'bias_shift': ''}
                            grid.append(row)
                            if best is None or mae < best[0]:
                                best = (mae, row.copy(), model, np.clip(base['test'] + lam * cor['test'], 2, 6))
                selected.append(best[1])
                prediction = best[3]
                record = best[1]
                model = best[2]
                np.savez_compressed(run / f'fold{fold}_{arm}_correction_fit.npz', train_features=xx, support=model.support_, dual_coef=model.dual_coef_, intercept=model.intercept_, bandwidth=b * record['scale'], correction_lambda=record['lambda'], target=target)
            for e, p in zip(roles['test'], prediction):
                preds.append({'variant': arm, 'fold': fold, 'seed': 'deterministic', 'role': 'test', 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference_BCS': int(e['bcs']), 'predicted_BCS': int(np.clip(np.rint(p), 2, 6)), 'expected_BCS': float(p)})
            print('CROSSFIT DONE', fold, arm, record, 'testMAE', float(abs(prediction - y['test']).mean()), flush=True)
    u.csv_out(run / 'test_predictions.csv', preds, list(preds[0]))
    u.csv_out(run / 'validation_grid.csv', grid, list(grid[0]))
    u.csv_out(run / 'selected_parameters.csv', selected, list(selected[0]))
    u.csv_out(run / 'training_residual_audit.csv', residual_rows, list(residual_rows[0]))
    (run / 'COMPLETE.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'arms': len(cfg['arms']), 'folds': 3, 'all_residual_teachers_group_disjoint': True, 'outer_test_used_for_selection': False, 'exploratory': True, 'script_sha256': u.sha256(Path(__file__))}, indent=2), encoding='utf-8')
    project_path('provenance/latest_crossfit_residual.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    (run / 'checksums.json').write_text(json.dumps([{'file': p.name, 'sha256': u.sha256(p)} for p in sorted(run.iterdir()) if p.is_file()], indent=2), encoding='utf-8')
    for arm in cfg['arms']:
        rr = [r for r in preds if r['variant'] == arm]
        print('POOLED CROSSFIT', arm, np.mean([abs(r['expected_BCS'] - r['reference_BCS']) for r in rr]), flush=True)
