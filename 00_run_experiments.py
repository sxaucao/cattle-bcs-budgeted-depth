"""Numbered research pipeline; may be invoked from any working directory."""
from pathlib import Path
import argparse,csv,json,subprocess,sys

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'02_code'
STEPS={
 'check':('01_verify_data.py',[]),
 'full':('01_verify_data.py',['--full']),
 'fit_cached':(None,[]),
 'features':('02_extract_frozen_features.py',[]),
 'grouped':('03_grouped_validation.py',[]),
 'baselines':('04_matched_budget_baselines.py',[]),
 'ablation':('05_score_profile_ablation.py',[]),
 'metrics':('06_compute_metrics.py',[]),
 'stress':('07_depth_loss_stress.py',[]),
 'benchmark':('08_native_inference.py',['--benchmark']),
 'infer':('08_native_inference.py',[]),
 'tables':('09_prepare_paper_tables.py',[]),
 'figures':('10_make_figures.py',[]),
}
def fit_cached():
    """Refit all main predictors and gates from the distributed frozen features."""
    provenance=ROOT/'12_reproducibility'/'provenance'
    cached=ROOT/(provenance/'latest_matched_controls.txt').read_text().strip()/'all_temporal_levels_features.npz'
    if not cached.is_file():
        raise FileNotFoundError('Missing frozen feature archive: '+str(cached.relative_to(ROOT)))
    jobs=[('03_grouped_validation.py',[]),
          ('04_matched_budget_baselines.py',['--precomputed-levels',cached.relative_to(ROOT).as_posix()]),
          ('05_score_profile_ablation.py',[]),('06_compute_metrics.py',[])]
    for name,flags in jobs:
        subprocess.run([sys.executable,str(CODE/name),*flags],cwd=ROOT,check=True)
    def read_rows(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            return {(r['method'],r['budget_cap']):r for r in csv.DictReader(stream)}
    expected=read_rows(ROOT/'12_reproducibility/reports/expected_summary.csv')
    actual=read_rows(ROOT/(provenance/'latest_final_metrics.txt').read_text().strip()/'summary.csv')
    assert set(expected)==set(actual),'Method-budget cells differ'
    difference=max(abs(float(expected[k]['MAE'])-float(actual[k]['MAE'])) for k in expected)
    assert difference<1e-6,difference
    report={'all_passed':True,'cells':len(expected),'max_MAE_difference':difference,
            'raw_images_required':False,'encoder_weights_required':False,
            'frozen_features_reused':True,'predictors_and_gates_refitted':True}
    destination=ROOT/'12_reproducibility/reports/cached_refit_verification.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('CACHED REFIT VERIFIED',json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',choices=STEPS,default='check');args,extra=p.parse_known_args()
    if args.step=='fit_cached':
        if extra:p.error('fit_cached takes no additional arguments')
        fit_cached()
        return
    name,flags=STEPS[args.step];result=subprocess.run([sys.executable,str(CODE/name),*flags,*extra],cwd=ROOT)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
