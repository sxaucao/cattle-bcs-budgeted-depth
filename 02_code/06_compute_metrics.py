import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Complete metrics, paired uncertainty and per-animal evidence for all main runs."""
import json
from datetime import datetime, timezone
from collections import defaultdict
from importlib import import_module
import numpy as np
from sklearn.metrics import r2_score, cohen_kappa_score, f1_score
from scipy.stats import spearmanr
syn = import_module('14_synthesise_results')
u = syn.u
ROOT = syn.ROOT

def measures(rr):
    y = np.array([float(r['reference']) for r in rr])
    p = np.array([float(r['prediction']) for r in rr])
    e = p - y
    a = abs(e)
    g = np.array([r['animal_group'] for r in rr])
    z = np.clip(np.rint(p), 2, 6)
    return {'MAE': float(a.mean()), 'RMSE': float(np.sqrt((e * e).mean())), 'bias': float(e.mean()), 'R2': float(r2_score(y, p)), 'Spearman': float(spearmanr(y, p).statistic), 'rounded_accuracy': float((z == y).mean()), 'rounded_macro_F1': float(f1_score(y, z, labels=[2, 3, 4, 5, 6], average='macro', zero_division=0)), 'rounded_QWK': float(cohen_kappa_score(y, z, labels=[2, 3, 4, 5, 6], weights='quadratic')), 'within_0.5': float((a <= 0.5).mean()), 'within_1.0': float((a <= 1).mean()), 'absolute_error_ge_2': float((a >= 2).mean()), 'animal_mean_MAE': float(np.mean([a[g == gg].mean() for gg in sorted(set(g))])), 'class_mean_MAE': float(np.mean([a[y == c].mean() for c in range(2, 7)])), 'depth_frames': float(sum((float(r['paid_depth_frames']) for r in rr))), 'all_event_output_fraction': len(rr) / 151, 'budget_feasible_fraction': float(np.mean([r['budget_feasible'] == 'True' for r in rr]))}

def main():
    import_module('00_archive_sources').archive()
    out = project_path('analysis', ROOT) / ('final_metrics_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir()
    rows = []
    sources = []
    for ptr in ['latest_repeated_validation.txt', 'latest_matched_controls.txt', 'latest_decision_profile.txt']:
        source = syn.pointer(ptr)
        assert (source / 'COMPLETE.json').exists()
        zz = u.csv_in(source / 'test_predictions.csv')
        sources.append(source.relative_to(ROOT).as_posix())
        for z in zz:
            z['seed'] = z.get('seed', 'deterministic')
            z['source_run'] = source.relative_to(ROOT).as_posix()
        rows.extend(zz)
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    u.csv_out(out / 'all_test_predictions.csv', rows, keys)
    by = defaultdict(list)
    for r in rows:
        by[r['method'], float(r['budget_cap'])].append(r)
    summary = []
    perrepeat = []
    perclass = []
    perfold = []
    vectors = {}
    costs = {}
    reference = None
    ids = None
    groups = None
    for (method, cap), zz in sorted(by.items()):
        reps = sorted({(int(r['repeat']), r['seed']) for r in zz})
        mms = []
        errors = defaultdict(list)
        paid = defaultdict(list)
        pred = defaultdict(list)
        bas = {}
        for repeat, seed in reps:
            rr = sorted([r for r in zz if int(r['repeat']) == repeat and r['seed'] == seed], key=lambda r: r['episode_id'])
            assert len(rr) == 151 and len({r['episode_id'] for r in rr}) == 151
            m = measures(rr)
            mms.append(m)
            perrepeat.append({'method': method, 'budget_cap': cap, 'repeat': repeat, 'seed': seed, **m})
            for r in rr:
                errors[r['episode_id']].append(float(r['absolute_error']))
                paid[r['episode_id']].append(float(r['paid_depth_frames']))
                pred[r['episode_id']].append(float(r['prediction']))
                bas[r['episode_id']] = r
                assert abs(float(r['absolute_error']) - abs(float(r['prediction']) - float(r['reference']))) < 1e-10
            for f in range(3):
                sub = [r for r in rr if int(r['fold']) == f]
                perfold.append({'method': method, 'budget_cap': cap, 'repeat': repeat, 'seed': seed, 'fold': f, 'events': len(sub), 'MAE': float(np.mean([float(r['absolute_error']) for r in sub])), 'depth_frames': sum((float(r['paid_depth_frames']) for r in sub))})
        ii = sorted(errors)
        if ids is None:
            ids = ii
            groups = np.array([bas[e]['animal_group'] for e in ids])
            reference = np.array([float(bas[e]['reference']) for e in ids])
        assert ids == ii and all((float(bas[e]['reference']) == reference[j] for j, e in enumerate(ids)))
        ev = np.array([np.mean(errors[e]) for e in ids])
        cv = np.array([np.mean(paid[e]) for e in ids])
        vectors[method, cap] = ev
        costs[method, cap] = cv
        lo, hi = syn.bootstrap(ev, groups)
        mm = {k: float(np.mean([m[k] for m in mms])) for k in mms[0]}
        repmae = [np.mean([m['MAE'] for r, m in zip(reps, mms) if r[0] == repeat]) for repeat in range(3)]
        summary.append({'method': method, 'budget_cap': cap, **mm, 'MAE_cluster_CI_low': lo, 'MAE_cluster_CI_high': hi, 'repeat_MAE_min': min(repmae), 'repeat_MAE_max': max(repmae), 'total_frame_calls': 1245 + mm['depth_frames'], 'frame_call_reduction': 1 - (1245 + mm['depth_frames']) / 2490, 'events': 151, 'groups': 52, 'unique_frames': 1245, 'exploratory': True})
        for c in range(2, 7):
            ix = reference == c
            perclass.append({'method': method, 'budget_cap': cap, 'BCS': c, 'events': int(ix.sum()), 'animal_groups': len(set(groups[ix])), 'MAE': float(ev[ix].mean()), 'depth_frames': float(cv[ix].sum())})
    pairs = []
    influence = []
    cand = 'Score_profile_RBF'
    comparators = ['RGBD_all', 'RGB_only', 'Uniform_max_affordable', 'Validation_fixed_level', 'Random_affordable_order', 'RGB_kernel_variance', 'Imputed_mean_shift', 'Binary_strict_crossfit', 'Multilevel_strict_crossfit', 'Probe_strict_crossfit', 'Score_profile_no_count', 'Score_profile_insample_features', 'Score_profile_insample_targets', 'Score_profile_shuffled', 'Score_profile_linear']
    endpoints = [('cap25', 0.25), ('cap50', 0.5), ('cap75', 0.75), ('mean_caps25_50_75', None)]
    for comparator in comparators:
        for endpoint, cap in endpoints:
            delta = np.mean([vectors[cand, c] - vectors[comparator, c] for c in [0.25, 0.5, 0.75]], axis=0) if cap is None else vectors[cand, cap] - vectors[comparator, cap]
            lo, hi = syn.bootstrap(delta, groups)
            vals = [float(delta[groups != g].mean()) for g in sorted(set(groups))]
            pairs.append({'candidate': cand, 'comparator': comparator, 'endpoint': endpoint, 'delta_MAE': float(delta.mean()), 'paired_cluster_CI_low': lo, 'paired_cluster_CI_high': hi, 'omit_BCS2_delta': float(delta[reference != 2].mean()), 'omit_one_animal_delta_min': min(vals), 'omit_one_animal_delta_max': max(vals), 'reps': 10000, 'exploratory_nominal_interval': True})
            if cap == 0.25:
                for g, v in zip(sorted(set(groups)), vals):
                    influence.append({'candidate': cand, 'comparator': comparator, 'omitted_animal': g, 'delta_remaining': v})
    for name, zz in [('summary.csv', summary), ('per_repeat.csv', perrepeat), ('per_class.csv', perclass), ('per_fold.csv', perfold), ('paired_comparisons.csv', pairs), ('animal_influence.csv', influence)]:
        u.csv_out(out / name, zz, list(zz[0]))
    vectorsave = {m + '__cap' + str(c): v for (m, c), v in vectors.items()}
    np.savez_compressed(out / 'mean_error_vectors.npz', episode_ids=np.array(ids), animal_groups=groups, reference=reference, **vectorsave)
    (out / 'metric_definitions.md').write_text('# Metric definitions\n\nEach fitted repeat predicts the same151 archived events once out of fold. Random allocation has10 policy seeds per repeat. Metrics are computed per repeat and policy seed, then averaged; predictions are not averaged before computing nonlinear metrics. MAE uncertainty uses paired per-event absolute errors averaged over repeats and policy seeds, resampling52 animal groups10000 times with fitted models held fixed. Folds/repeats/frames are not independent animals.\n\nRounded predictions use NumPy nearest-even rounding and clipping to observed labels2-6. Tolerances0.5 and1.0 apply to continuous score differences; they are descriptive, not validated management thresholds. Severe error is absolute continuous error>=2. Class MAE gives equal weight to5 represented classes; animal MAE gives equal weight to52 inherited identifier groups. R2 may be negative.\n\nDepth cost counts actual encoded frames. Probe models pay one frame for every probed event, including RGB fallback. RGB always costs1245 frames. Total reduction uses2490 full RGBD frame calls as denominator. Coverage means outputs/151 retained archived events, not all on-farm acquisition opportunities. Fixed-level baselines are flagged infeasible when they exceed the declared cap.\n\nAll nominal confidence intervals are exploratory; repeated reuse and method search are disclosed. No post-hoc noninferiority margin is used.\n', encoding='utf-8')
    (out / 'verification.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_runs': sources, 'methods': len({m for m, c in by}), 'all_events_per_repeat': 151, 'groups': 52, 'error_reconstruction_passed': True, 'all_repeat_seed_cohorts_unique': True, 'random_policy_replicates_not_counted_as_animals': True, 'bootstrap_replicates': 10000, 'exploratory': True}, indent=2), encoding='utf-8')
    project_path('provenance/latest_final_metrics.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    for r in pairs:
        if r['endpoint'] == 'mean_caps25_50_75':
            print('PRIMARY COMPARISON', r, flush=True)
    print('OUT', out, flush=True)
if __name__ == '__main__':
    main()
