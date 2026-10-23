from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Reconstruct every outcome from held-out predictions, with group uncertainty."""
import csv, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from importlib import import_module
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
u = import_module('01_audit_and_snapshot')
ROOT = project_root()

def pointer(name):
    return ROOT / (project_path('provenance', ROOT) / name).read_text().strip()

def metrics(rr):
    y = np.array([float(r['reference_BCS']) for r in rr])
    p = np.array([float(r['expected_BCS']) for r in rr])
    z = np.array([int(float(r['predicted_BCS'])) for r in rr])
    a = abs(p - y)
    gg = defaultdict(list)
    for r, e in zip(rr, a):
        gg[r['animal_group']].append(float(e))
    return {'MAE': float(a.mean()), 'RMSE': float(np.sqrt(np.square(p - y).mean())), 'bias': float((p - y).mean()), 'animal_mean_MAE': float(np.mean([np.mean(v) for v in gg.values()])), 'class_mean_MAE': float(np.mean([a[y == c].mean() for c in sorted(set(y))])), 'rounded_or_argmax_MAE': float(abs(z - y).mean()), 'accuracy': float((z == y).mean()), 'macro_F1': float(f1_score(y, z, labels=[2, 3, 4, 5, 6], average='macro', zero_division=0)), 'QWK': float(cohen_kappa_score(y, z, labels=[2, 3, 4, 5, 6], weights='quadratic')), 'n_episodes': len(y), 'animal_groups': len(gg)}

def bootstrap(vector, groups, reps=10000, seed=711):
    groups = np.array(groups)
    vector = np.array(vector)
    unique = sorted(set(groups))
    s = np.array([vector[groups == g].sum() for g in unique])
    n = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(unique), size=(reps, len(unique)))
    vals = s[ix].sum(1) / n[ix].sum(1)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

def collect_neural(run, expected):
    if not (run / 'COMPLETE.json').exists():
        raise RuntimeError('Incomplete run ' + str(run))
    rows = []
    audits = []
    for completion in sorted(run.glob('fold*_seed*/*/completion.json')):
        d = json.loads(completion.read_text())
        cfg = json.loads((run / 'experiment.json').read_text())
        if d['variant'] not in cfg['variants']:
            continue
        ck = run / d['checkpoint']
        attempt = run / d['attempt']
        r = u.csv_in(attempt / 'test_predictions.csv')
        assert ck.exists() and r
        assert all((x['role'] == 'test' for x in r))
        computed = metrics(r)
        original = d['metrics']['test']
        orig = original.get('expected_MAE', original.get('MAE'))
        assert abs(computed['MAE'] - orig) < 2e-06
        roles = u.csv_in(attempt / 'episode_roles.csv')
        sets = {role: {x['animal_group'] for x in roles if x['role'] == role} for role in ['train', 'validation', 'test']}
        assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
        audits.append({'variant': d['variant'], 'fold': d['fold'], 'seed': d['seed'], 'checkpoint': str(ck.relative_to(ROOT)), 'checkpoint_sha256': u.sha256(ck), 'prediction_file': str((attempt / 'test_predictions.csv').relative_to(ROOT)), 'MAE_recomputed': computed['MAE'], 'MAE_logged': orig, 'group_disjoint': True, 'epochs_trained': d['epochs_trained'], 'runtime_seconds': d['runtime_seconds']})
        for x in r:
            x['run'] = run.relative_to(ROOT).as_posix()
        rows.extend(r)
    assert len(audits) == expected, (run, len(audits), expected)
    return (rows, audits)

def main():
    import_module('00_archive_sources').archive()
    out = project_path('analysis', ROOT) / ('synthesis_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True)
    allrows = []
    audits = []
    for ptr, n in [('latest_formal_experiment.txt', 81), ('latest_development_experiment.txt', 72), ('latest_regression_experiment.txt', 72)]:
        rr, aa = collect_neural(pointer(ptr), n)
        allrows.extend(rr)
        audits.extend(aa)
    for ptr in ['latest_ridge_reference.txt', 'latest_graph_experiment.txt', 'latest_relief_experiment.txt', 'latest_distribution_experiment.txt', 'latest_ordinal_fusion.txt', 'latest_crossfit_residual.txt', 'latest_bidirectional_residual.txt', 'latest_complementary_kernel.txt']:
        run = pointer(ptr)
        rr = u.csv_in(run / 'test_predictions.csv')
        for x in rr:
            x['run'] = run.relative_to(ROOT).as_posix()
        allrows.extend(rr)
    foundation = pointer('latest_foundation_experiment.txt')
    assert (foundation / 'COMPLETE.json').exists()
    rr = u.csv_in(foundation / 'source_test_predictions.csv')
    for x in rr:
        x['run'] = foundation.relative_to(ROOT).as_posix()
    allrows.extend(rr)
    field = list(dict.fromkeys((k for r in allrows for k in r)))
    u.csv_out(out / 'source_all_test_predictions.csv', allrows, field)
    u.csv_out(out / 'trained_model_audit.csv', audits, list(audits[0]))
    by = defaultdict(list)
    for r in allrows:
        by[r['variant']].append(r)
    table = []
    perseed = []
    error_vectors = {}
    groups = None
    identifiers = None
    perclass = []
    for variant, rr in by.items():
        seeds = sorted({r['seed'] for r in rr})
        mm = []
        errs = defaultdict(list)
        base = {}
        for seed in seeds:
            r = [x for x in rr if x['seed'] == seed]
            assert len(r) == 151 and len({x['episode_id'] for x in r}) == 151
            m = metrics(r)
            perseed.append({'variant': variant, 'seed': seed, **m})
            mm.append(m)
            for x in r:
                errs[x['episode_id']].append(abs(float(x['expected_BCS']) - float(x['reference_BCS'])))
                base[x['episode_id']] = x
        ids = sorted(errs)
        if identifiers is None:
            identifiers = ids
            groups = [base[k]['animal_group'] for k in ids]
        assert ids == identifiers
        ev = np.array([np.mean(errs[k]) for k in ids])
        error_vectors[variant] = ev
        lo, hi = bootstrap(ev, groups)
        row = {'variant': variant, 'replicates': len(seeds), 'n_episodes_per_replicate': 151, 'animal_groups': 52, 'MAE_mean': float(np.mean([m['MAE'] for m in mm])), 'MAE_seed_SD': float(np.std([m['MAE'] for m in mm], ddof=1)) if len(seeds) > 1 else 0.0, 'MAE_cluster_CI_low': lo, 'MAE_cluster_CI_high': hi, **{k + '_mean': float(np.mean([m[k] for m in mm])) for k in ['RMSE', 'bias', 'animal_mean_MAE', 'class_mean_MAE', 'rounded_or_argmax_MAE', 'accuracy', 'macro_F1', 'QWK']}}
        table.append(row)
        for c in range(2, 7):
            ix = [i for i, k in enumerate(ids) if int(base[k]['reference_BCS']) == c]
            perclass.append({'variant': variant, 'BCS': c, 'n_episodes': len(ix), 'n_animal_groups': len({groups[i] for i in ix}), 'MAE': float(ev[ix].mean())})
    u.csv_out(out / 'source_summary.csv', table, list(table[0]))
    u.csv_out(out / 'source_per_seed_metrics.csv', perseed, list(perseed[0]))
    u.csv_out(out / 'source_per_class.csv', perclass, list(perclass[0]))
    comparisons = []
    candidate_names = ['RGB_depth_benefit', 'RGB_rgbd_prototype', 'RGB_reg_rgbd_feature', 'Depth_residual_graph', 'DINO_RGBD_global_SVR', 'DINO_ordinal_align_SVR', 'Crossfit_RGB_residual_SVR', 'Bidirectional_equal']
    baselines = ['RGB_CE', 'RGB_ordinal', 'RGB_depth_KD', 'RGB_gaussian', 'RGB_rgb_prototype', 'RGB_entropy_matched', 'RGB_mismatched_prototype', 'RGB_regression', 'RGB_reg_self_feature', 'RGB_reg_mismatched_feature', 'RGB_CORN', 'RGB_balanced_KRR', 'Label_residual_graph', 'RGB_residual_graph', 'Depth_smoothing_graph', 'Mismatched_depth_residual_graph', 'RGB_SVR', 'RGBD_CE', 'RGBD_KRR', 'RGBD_SVR', 'DINO_RGB_global_SVR', 'DINO_Depth_global_SVR', 'DINO_RGBD_global_SVR', 'DINO_RGBD_consensus_SVR', 'DINO_equal_kernel_SVR', 'DINO_shuffled_ordinal_align_SVR', 'Crossfit_Depth_residual_SVR', 'Insample_RGBD_residual_SVR', 'Shuffled_crossfit_RGBD_residual_SVR', 'Depth_SVR', 'Depth_validation_bias_calibration']
    for cand in candidate_names:
        for base in baselines:
            if cand == base:
                continue
            delta = error_vectors[cand] - error_vectors[base]
            lo, hi = bootstrap(delta, groups)
            comparisons.append({'candidate': cand, 'comparator': base, 'delta_MAE_candidate_minus_comparator': float(delta.mean()), 'paired_cluster_CI_low': lo, 'paired_cluster_CI_high': hi, 'clusters': 52, 'bootstrap_replicates': 10000, 'exploratory': True})
    u.csv_out(out / 'source_paired_comparisons.csv', comparisons, list(comparisons[0]))
    np.savez_compressed(out / 'source_error_vectors.npz', episode_ids=np.array(identifiers), animal_groups=np.array(groups), **error_vectors)
    ext = pointer('latest_external_transfer.txt')
    rr = u.csv_in(ext / 'test_predictions.csv')
    by = defaultdict(list)
    for r in rr:
        by[r['family']].append(r)
    et = []
    evec = {}
    eg = None
    eid = None
    sensitivity = []
    for fam, rows in by.items():
        errors = defaultdict(list)
        base = {}
        reps = {(r['source_fold'], r['source_seed']) for r in rows}
        assert len(rows) == 207 * len(reps)
        for r in rows:
            key = r['image_name']
            errors[key].append(abs(float(r['predicted_native_BCS']) - float(r['reference_BCS_mean'])))
            base[key] = r
        keys = sorted(errors)
        if eid is None:
            eid = keys
            eg = [base[k]['animal_group'] for k in keys]
        assert keys == eid
        ev = np.array([np.mean(errors[k]) for k in keys])
        evec[fam] = ev
        lo, hi = bootstrap(ev, eg)
        et.append({'family': fam, 'source_representations': len(reps), 'n_images': 207, 'n_groups': 29, 'MAE_native_scale': float(ev.mean()), 'cluster_CI_low': lo, 'cluster_CI_high': hi})
        for ref in ['BCS1', 'BCS2']:
            sensitivity.append({'family': fam, 'reference': ref, 'MAE': float(np.mean([abs(float(r['predicted_native_BCS']) - float(r[ref])) for r in rows]))})
    u.csv_out(out / 'external_summary.csv', et, list(et[0]))
    u.csv_out(out / 'external_rater_sensitivity.csv', sensitivity, list(sensitivity[0]))
    ec = []
    for base in ['ImageNet', 'RGB_CE', 'RGB_gaussian', 'RGB_rgb_prototype', 'RGB_entropy_matched', 'Target_training_mean']:
        d = evec['RGB_rgbd_prototype'] - evec[base]
        lo, hi = bootstrap(d, eg)
        ec.append({'candidate': 'RGB_rgbd_prototype', 'comparator': base, 'delta_MAE': float(d.mean()), 'paired_cluster_CI_low': lo, 'paired_cluster_CI_high': hi, 'clusters': 29})
    u.csv_out(out / 'external_paired_comparisons.csv', ec, list(ec[0]))
    np.savez_compressed(out / 'external_error_vectors.npz', image_names=np.array(eid), animal_groups=np.array(eg), **evec)
    dairy = u.csv_in(foundation / 'dairy_test_predictions.csv')
    dt = []
    dv = {}
    dg = None
    di = None
    for arm in sorted({r['variant'] for r in dairy}):
        rows = sorted([r for r in dairy if r['variant'] == arm], key=lambda r: r['episode_id'])
        assert len(rows) == 207
        keys = [r['episode_id'] for r in rows]
        groups2 = [r['animal_group'] for r in rows]
        if di is None:
            di = keys
            dg = groups2
        assert keys == di
        ev = np.array([abs(float(r['expected_BCS']) - float(r['reference_BCS'])) for r in rows])
        dv[arm] = ev
        lo, hi = bootstrap(ev, dg)
        dt.append({'variant': arm, 'native_MAE': float(ev.mean()), 'CI_low': lo, 'CI_high': hi, 'n_images': 207, 'n_groups': 29, 'calibration': 'Separately supervised in dairy cohort; no score-scale conversion'})
    dv['Target_training_mean'] = evec['Target_training_mean']
    assert di == eid
    dc = []
    for candidate in ['DINO_RGB_global_SVR', 'DINO_RGB_centre_SVR', 'DINO_RGB_consensus_SVR']:
        for base in ['Target_training_mean', 'DINO_RGB_global_SVR', 'DINO_RGB_centre_SVR']:
            if base == candidate:
                continue
            delta = dv[candidate] - dv[base]
            lo, hi = bootstrap(delta, dg)
            dc.append({'candidate': candidate, 'comparator': base, 'delta_MAE': float(delta.mean()), 'CI_low': lo, 'CI_high': hi, 'groups': 29})
    u.csv_out(out / 'dairy_foundation_summary.csv', dt, list(dt[0]))
    u.csv_out(out / 'dairy_foundation_comparisons.csv', dc, list(dc[0]))
    info = {'created_utc': datetime.now(timezone.utc).isoformat(), 'neural_models_audited': len(audits), 'source_episodes': 151, 'source_groups': 52, 'source_variants': len(table), 'source_metric': 'Each seed pools all three test folds into 151 out-of-fold episodes; table averages those complete-cohort metrics, not fold means or ensemble predictions', 'cluster_bootstrap': '10000 resamples of animal groups; episode absolute errors averaged across seed or source-model replicates before resampling; conditional on fitted models, no retraining', 'external': 'Supervised target-scale calibration on independent cattle; source encoders are not independent target animals; no zero-shot or native-scale conversion', 'source_development': 'All later candidates are exploratory because previous source outer-test outcomes were observed; paired confidence intervals are nominal and not adjusted for the complete adaptive development history', 'checked_training_group_separation': True, 'checked_logged_MAE_reconstruction': True, 'script_sha256': u.sha256(Path(__file__))}
    (out / 'verification.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    project_path('provenance/latest_synthesis.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    for r in sorted(table, key=lambda r: r['MAE_mean']):
        print(r['variant'], round(r['MAE_mean'], 4), 'CI', round(r['MAE_cluster_CI_low'], 4), round(r['MAE_cluster_CI_high'], 4))
    print('SYNTHESIS COMPLETE ' + str(out), flush=True)
