import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Prepare publication tables and a single verified data source for all artifacts."""
import json, csv, statistics, re, html, shutil, argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from importlib import import_module
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=project_root())
    args = parser.parse_args()
    root = args.project.resolve()

    def read(p):
        with p.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def pointer(n):
        return root / (project_path('provenance', root) / n).read_text().strip()

    def write(p, rr):
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    analysis = pointer('latest_final_metrics.txt')
    summary = read(analysis / 'summary.csv')
    paired = read(analysis / 'paired_comparisons.csv')
    native = pointer('latest_native_score_profile.txt')
    timing = read(native / 'timings.csv')
    stress = read(pointer('latest_depth_stress.txt') / 'test_predictions.csv')
    by = defaultdict(list)
    for z in timing:
        by[z['policy'], float(z['budget_cap'])].append(z)
    timing_summary = []
    for (policy, cap), zz in by.items():
        vv = [float(z['end_to_end_seconds']) for z in zz]
        timing_summary.append({'policy': policy, 'budget_cap': cap, 'events': int(zz[0]['events']), 'RGB_frames': int(zz[0]['RGB_frames']), 'depth_frames': int(zz[0]['depth_frames']), 'median_seconds': statistics.median(vv), 'minimum_seconds': min(vv), 'maximum_seconds': max(vv), 'repeats': len(vv)})
    tbase = next((z['median_seconds'] for z in timing_summary if z['policy'] == 'full'))
    for z in timing_summary:
        z['reduction_vs_full'] = 1 - z['median_seconds'] / tbase
    by = defaultdict(list)
    for z in stress:
        by[z['profile'], z['method'], float(z['budget_cap'])].append(z)
    stress_summary = [{'profile': key[0], 'method': key[1], 'budget_cap': key[2], 'MAE': float(np.mean([float(z['absolute_error']) for z in zz])), 'mean_depth_frames': sum((int(z['paid_depth_frames']) for z in zz)) / 3, 'all_events_scored': len(zz) == 453} for key, zz in sorted(by.items())]
    methods = ['RGB_only', 'RGBD_all', 'Uniform_max_affordable', 'Random_affordable_order', 'RGB_kernel_variance', 'Imputed_mean_shift', 'Binary_strict_crossfit', 'Multilevel_strict_crossfit', 'Probe_strict_crossfit', 'Score_profile_RBF']
    mainrows = [z for method in methods for z in summary if z['method'] == method and float(z['budget_cap']) == 0.25]
    abl = [z for z in paired if z['endpoint'] == 'mean_caps25_50_75']
    tables = project_path('tables', root)
    tables.mkdir(exist_ok=True)
    write(tables / 'Table2_main_cap25.csv', mainrows)
    write(tables / 'Table3_paired_ablation.csv', abl)
    write(tables / 'TableS_stress.csv', stress_summary)
    write(tables / 'TableS_runtime.csv', timing_summary)
    for name in ['summary.csv', 'per_repeat.csv', 'per_class.csv', 'per_fold.csv', 'animal_influence.csv']:
        shutil.copy2(analysis / name, tables / ('Full_' + name))
    prepared = pointer('latest_prepared.txt')
    counts = read(prepared / 'class_counts.csv')
    data={'summary':summary,'paired':paired,'per_class':read(analysis/'per_class.csv'),'per_repeat':read(analysis/'per_repeat.csv'),'stress':stress_summary,'timing':timing_summary,'class_counts':counts}
    project_path('plot_data/research_data.json', root).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PLOT DATA', project_path('plot_data/research_data.json', root), flush=True)
if __name__ == '__main__':
    main()
