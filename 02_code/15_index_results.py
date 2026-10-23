"""Index and copy recorded outputs in execution order without rerunning experiments."""
from pathlib import Path
import csv
import hashlib
import shutil
from project_paths import project_root, project_path

ROOT=project_root()
PROVENANCE=project_path('provenance')
STAGES=[
 ('01_data_provenance','latest_prepared.txt',[
  ('01_stage_ledger.csv','stage_ledger.csv'),('02_class_counts.csv','class_counts.csv'),
  ('03_episodes.csv','episodes.csv'),('04_frame_slots.csv','evaluation_frame_slots.csv')]),
 ('04_grouped_validation','latest_repeated_validation.txt',[
  ('01_test_predictions.csv','test_predictions.csv'),('02_group_splits.csv','group_splits.csv'),
  ('03_crossfit_teacher_audit.csv','crossfit_teacher_audit.csv'),
  ('04_scorer_validation_grid.csv','base_validation_grid.csv'),('05_selected_gates.csv','selected_gates.csv')]),
 ('05_budget_baselines','latest_matched_controls.txt',[
  ('01_test_predictions.csv','test_predictions.csv'),('02_fixed_level_predictions.csv','fixed_level_test_predictions.csv'),
  ('03_validation_grid.csv','base_validation_grid.csv'),('04_selected_imputers.csv','selected_imputers.csv')]),
 ('06_ablation','latest_decision_profile.txt',[
  ('01_test_predictions.csv','test_predictions.csv'),('02_selected_parameters.csv','selected_parameters.csv'),
  ('03_validation_grid.csv','validation_grid.csv')]),
 ('07_statistics','latest_final_metrics.txt',[
  ('01_summary.csv','summary.csv'),('02_paired_comparisons.csv','paired_comparisons.csv'),
  ('03_per_class.csv','per_class.csv'),('04_per_repeat.csv','per_repeat.csv'),
  ('05_animal_influence.csv','animal_influence.csv'),('06_per_fold.csv','per_fold.csv')]),
 ('08_depth_stress','latest_depth_stress.txt',[
  ('01_test_predictions.csv','test_predictions.csv'),('02_corruption_audit.csv','corruption_audit.csv')]),
 ('09_native_benchmark','latest_native_score_profile.txt',[
  ('01_timings.csv','timings.csv'),('02_predictions.csv','predictions.csv'),
  ('03_prediction_parity.csv','prediction_parity.csv'),('04_input_audit.csv','input_audit.csv')]),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    records=[]
    for folder,pointer,files in STAGES:
        source=ROOT/(PROVENANCE/pointer).read_text().strip()
        target=ROOT/folder/'results'
        target.mkdir(parents=True,exist_ok=True)
        lines=['# '+folder.replace('_',' ',1),'',
               'Numbered copies of recorded results. These are not additional experiments.',
               'The exact source paths and SHA-256 checksums are recorded in `12_reproducibility/result_index.csv`.','']
        for label,name in files:
            original=source/name
            if not original.is_file():
                raise FileNotFoundError(original)
            destination=target/label
            shutil.copy2(original,destination)
            checksum=sha(original)
            assert sha(destination)==checksum
            records.append({'ordered_file':destination.relative_to(ROOT).as_posix(),
                            'source_file':original.relative_to(ROOT).as_posix(),'SHA256':checksum})
            lines.append('- ['+label+'](results/'+label+')')
        (ROOT/folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    output=ROOT/'12_reproducibility/result_index.csv'
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['ordered_file','source_file','SHA256'])
        writer.writeheader();writer.writerows(records)
    print('Indexed',len(records),'result files:',output.relative_to(ROOT).as_posix())

if __name__=='__main__':
    main()
