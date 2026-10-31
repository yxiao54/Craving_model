# Public Release Code Bundle

This folder collects the core code needed to release the current modeling and statistical analysis pipeline in a cleaner structure.

## Layout

- `core/`
  - Shared package used by the main model and baseline training code.
- `main_model/`
  - `pretrain_stress_encoder.py`
  - `train_craving.py`
- `baselines/`
  - `train.py`
- `stat_analysis/`
  - `rq1.py` -- physiological modulation analysis used before the main evaluation
  - `rq2.py` -- language--recovery alignment analysis used before the main evaluation
- `feature_extraction/`
  - `extract_h5_window_features.py`
  - `filter_feature_parquets.py`
- `utils/`
  - `extract_hr_postgap_stat_embeddings.py`

## Intended scope

This bundle covers:

1. Main model training
2. Stress-encoder pretraining
3. RETRACE craving training and inference
4. Baseline training and inference
5. Pre-evaluation physiological modulation analysis
6. Pre-evaluation language--recovery alignment analysis
7. HR-AUC subject-embedding construction used by the current release bundle
8. Generic physiological feature extraction and feature-quality filtering

It does not include raw data due to IRB restrictions.

## Running the code

Run scripts from this folder root and set `PYTHONPATH` so the shared `core` package is visible:

```bash
cd public_release_code
PYTHONPATH=. python main_model/train_craving.py --help
PYTHONPATH=. python baselines/train.py --help
```

The public release scripts do not assume the original local workspace layout. Paths, split files, checkpoint directories, and output directories should be passed explicitly at runtime.

Typical entry points:

```bash
cd public_release_code
PYTHONPATH=. python stat_analysis/rq1.py \
  --oud-parquet data/oud_left.parquet \
  --control-parquet data/control_left.parquet \
  --checkpoint-dir checkpoints/stress_encoder \
  --resilience-groups-json resilience_groups.json
PYTHONPATH=. python stat_analysis/rq2.py --emb-npz emb_data.npz --labels-csv labels.csv
PYTHONPATH=. python baselines/train.py \
  --baseline cross_attention \
  --parquet-path data/oudlab_windows.parquet \
  --embedding-path data/subject_embeddings.pickle \
  --splits-path data/loso_splits.json \
  --output-dir runs/baselines/cross_attention
PYTHONPATH=. python feature_extraction/extract_h5_window_features.py \
  --input-dir converted_h5 \
  --output-path runs/window_features.parquet
PYTHONPATH=. python feature_extraction/filter_feature_parquets.py \
  --input-dir runs \
  --output-dir filtered \
  --files window_features.parquet
```



## Notes

- The current main model is the dual-path frozen-stress craving model in `main_model/train_craving.py`.
- The baseline entry point is `baselines/train.py`.
- The baseline entry point supports `mlp`, `irm`, `vrex`, `groupdro`, `concat`, `cross_attention`, `film_input`, `moe`, `rbf_svm`, `xgboost`, and `random_forest`.
- `stat_analysis/rq1.py` implements the physiological modulation analysis that motivates the subsequent craving-detection evaluation.
- `stat_analysis/rq2.py` implements the language--recovery alignment analysis that motivates semantic subject-level guidance. The embedding matrix is standardized once, and the one-component PLS model is refitted for each participant-level HR-AUC permutation.
- The `feature_extraction/` scripts are dataset-agnostic public-release versions of the feature extraction and filtering pipeline. They intentionally avoid hardcoded local paths and dataset-specific naming conventions.
- The shared modeling package released with this code bundle is named `core`.
