# Cattle BCS: budgeted depth processing

Reproducible experiments for allocating depth processing to cattle body condition scoring (BCS) events. RGB features are always available; a paid depth probe precedes adaptive allocation. Animal-grouped validation separates predictor fitting, cross-fitting, hyperparameter selection, and held-out testing.

## Repository layout

| Order | Directory | Contents |
| --- | --- | --- |
| 01 | `01_data_provenance` | Source-file checksums, episode manifests, selection records, and data audit |
| 02 | `02_code` | Numbered executable scripts, configurations, and supporting modules |
| 03 | `03_models_features` | Frozen episode-level features and encoder acquisition metadata |
| 04 | `04_grouped_validation` | Animal-group splits, predictions, and scorer-selection records |
| 05 | `05_budget_baselines` | Matched-cost controls and cached features for all depth levels |
| 06 | `06_ablation` | Score-profile ablations, predictions, and selection records |
| 07 | `07_statistics` | Summary metrics, paired comparisons, class analyses, and plotting data |
| 08 | `08_depth_stress` | Depth-loss sensitivity experiments |
| 09 | `09_native_benchmark` | Native-image timing, predictions, and parity checks |
| 10 | `10_figures` | Generated scientific figures |
| 12 | `12_reproducibility` | Run pointers, expected metrics, checksums, and verification reports |

Raw images, patch-token caches, fitted model arrays, encoder weights, and manuscript/submission files are excluded from Git. Frozen episode-level features needed to refit the principal experiments are included. A complete local project can additionally audit the native data and replay image inference.

## Installation

Python 3.12 was used for the recorded experiments. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r 02_code/requirements.txt
```

The cached-feature experiments run on CPU. PyTorch and torchvision are imported by the shared implementation; CUDA is needed only to reproduce the recorded GPU timing conditions. The original package versions are recorded in `02_code/requirements_actual.txt`. Optional figure dependencies are in `02_code/requirements_figures.txt`. Times New Roman is discovered locally at runtime and is not redistributed.

## Reproduce the main experiments from a Git checkout

```bash
python 00_run_experiments.py --step fit_cached
```

This command refits the grouped scorers, matched-budget controls, score-profile gates, and ablations from the distributed frozen features, then compares all 120 method-budget MAE cells with the recorded reference. Native images, patch-token caches, and encoder weights are not required. New timestamped output folders are created under the numbered directories and run pointers are updated.

Paths are resolved relative to this repository. The launcher also works when called by absolute path from another working directory.

For the complete local project, including the excluded native assets:

```bash
python 00_run_experiments.py --step check
python 00_run_experiments.py --step full
```

`check` verifies source-file sizes and SHA-256 checksums. `full` also replays frozen-feature preparation, all main fitting stages, metric computation, and native-image inference parity. Individual stages are available through `python 00_run_experiments.py --help`; each numbered script also supports direct execution. Rebuild the numbered result index after running experiments with `python 02_code/15_index_results.py`.

## Data accounting and evaluation

The audited source contains 3,815 RGB/depth pairs (7,630 files), 151 scoring episodes, and 52 filename-defined animal groups. Consolidating 638 redundant decoded RGB frames leaves 3,177 representatives; selecting at most 12 frames per episode yields 1,245 input frames per modality. Detailed transitions and source-file hashes are retained in `01_data_provenance`.

Validation uses three repeats of three animal-grouped folds and five depth-budget caps. All RGB processing and the initial depth probe are charged; a probe remains charged when the final output uses the RGB fallback. The main candidate is `Score_profile_RBF`. Statistical comparisons retain the original paired units; point-estimate improvements do not imply superiority where uncertainty intervals include zero. The sparse BCS-2 stratum and the lack of an independent RGB/depth dataset limit generalisation.

## Results and provenance

Numbered result files and exact source paths are listed in `12_reproducibility/result_index.csv`; stage READMEs identify the corresponding outputs. The `latest_*.txt` pointers select the recorded run for each stage. Reference metrics are stored in `12_reproducibility/reports/expected_summary.csv`.

Filename-defined animal identifiers have not been externally verified. Raw data and pretrained weights are not redistributed here; their original access conditions and terms are separate from access to this code.
