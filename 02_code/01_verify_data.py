import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Verify all native files, optionally replay the main pipeline in the release."""
import argparse, csv, json, hashlib, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
ROOT = project_root()

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def ptr(n):
    return ROOT / (project_path('provenance', ROOT) / n).read_text().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    out = project_path('reports', ROOT) / ('reproduction_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir()
    files = rows(ptr('latest_audit.txt') / 'files_manifest.csv')
    checks = []
    total = 0
    for i, r in enumerate(files):
        p = ROOT / r['snapshot_relpath']
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert p.stat().st_size == int(r['bytes'])
        assert sha(p) == r['sha256_snapshot']
        total += p.stat().st_size
        if i % 1000 == 0:
            print('RAW SHA VERIFIED', i, len(files), flush=True)
    checks.append({'check': 'All native file sizes and SHA256', 'passed': True, 'files': len(files), 'bytes': total})
    for sub in ['source_RGB_tokens', 'source_Depth_tokens']:
        rr = rows(ptr('latest_foundation_experiment.txt') / sub / 'input_manifest.csv')
        assert len(rr) == 1245 and all((Path(r['path']).resolve().is_relative_to(ROOT.resolve()) for r in rr))
        assert all((Path(r['path']).exists() for r in rr))
    checks.append({'check': 'All selected native paths are internal and present', 'passed': True, 'RGB_frames': 1245, 'Depth_frames': 1245})
    commands = ['02_extract_frozen_features.py', '03_grouped_validation.py', '04_matched_budget_baselines.py', '05_score_profile_ablation.py', '06_compute_metrics.py'] if args.full else []
    for name in commands:
        start = time.perf_counter()
        with (out / (name + '.log')).open('w', encoding='utf-8') as log:
            r = subprocess.run([sys.executable, str(project_path('scripts', ROOT) / name)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        checks.append({'check': name, 'passed': r.returncode == 0, 'seconds': time.perf_counter() - start})
        print('REPRODUCTION STAGE', name, r.returncode, flush=True)
        if r.returncode:
            raise RuntimeError('Replay failed; inspect ' + str(out / (name + '.log')))
    if args.full:
        expected = project_path('reports/expected_summary.csv', ROOT)
        assert expected.exists(), 'Missing internal expected summary'
        a = {(r['method'], r['budget_cap']): r for r in rows(expected)}
        b = {(r['method'], r['budget_cap']): r for r in rows(ptr('latest_final_metrics.txt') / 'summary.csv')}
        assert set(a) == set(b)
        maxdiff = max((abs(float(a[k]['MAE']) - float(b[k]['MAE'])) for k in a))
        assert maxdiff < 1e-06, maxdiff
        checks.append({'check': 'Replay final MAE versus original for all120 method-cap cells', 'passed': True, 'max_MAE_difference': maxdiff, 'cells': len(a)})
        native = out / 'native_inference'
        cmd = [sys.executable, str(project_path('scripts/08_native_inference.py', ROOT)), '--manifest', str(project_path('data/example_inference_manifest.csv', ROOT)), '--cap', '.25', '--output', str(native)]
        with (out / 'native_inference.log').open('w', encoding='utf-8') as log:
            r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert r.returncode == 0
        reference = rows(ptr('latest_decision_profile.txt') / 'test_predictions.csv')
        lookup = {r['episode_id']: r for r in reference if int(r['repeat']) == 0 and int(r['fold']) == 0 and (r['method'] == 'Score_profile_RBF') and (float(r['budget_cap']) == 0.25)}
        got = rows(native / 'predictions.csv')
        assert len(got) == len(lookup)
        maxdiff = max((abs(float(r['prediction']) - float(lookup[r['episode_id']]['prediction'])) for r in got))
        assert maxdiff < 0.005
        assert all((int(r['paid_depth_frames']) == int(lookup[r['episode_id']]['paid_depth_frames']) for r in got))
        checks.append({'check': 'Native release inference matches replayed held-out predictions and costs', 'passed': True, 'events': len(got), 'max_prediction_difference': maxdiff})
    info = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'full_replay': args.full, 'all_passed': all((r['passed'] for r in checks)), 'checks': checks, 'python': Path(sys.executable).name}
    (out / 'verification.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    project_path('provenance/latest_reproduction.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    print('REPRODUCTION VERIFIED', json.dumps(info), flush=True)
if __name__ == '__main__':
    main()
