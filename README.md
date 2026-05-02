# Public Release Code Bundle

This folder collects the core code needed to release the current modeling and statistical analysis pipeline in a cleaner structure.

## Layout

- `core/`
  - Shared package used by the main model and baseline training code.
- `main_model/`
  - `pretrain_stress_encoder.py`
  - `optuna_craving_frozen_stress.py`
- `baselines/`
  - `optuna_craving_baselines.py`
- `stat_analysis/`
  - `rq1.py`
  - `rq2.py`
- `feature_extraction/`
  - `extract_h5_window_features.py`
  - `filter_feature_parquets.py`
- `utils/`
  - `extract_hr_postgap_stat_embeddings.py`

## Intended scope

This bundle covers:

1. Main model training
2. Stress-encoder pretraining
3. Baseline models
4. RQ1 statistical analysis
5. RQ2 representational alignment analysis
6. HR dynamic embedding construction
7. Generic physiological feature extraction and feature-quality filtering

It does not include raw data due to IRB restrictions.

## Running the code

Run scripts from this folder root and set `PYTHONPATH` so the shared `core` package is visible:

```bash
cd public_release_code
PYTHONPATH=. python main_model/optuna_craving_frozen_stress.py --help
```

The public release scripts do not assume the original local workspace layout. Paths, split files, checkpoint directories, and study names should be passed explicitly at runtime.

Typical entry points:

```bash
cd public_release_code
PYTHONPATH=. python baselines/optuna_craving_baselines.py --help
PYTHONPATH=. python stat_analysis/rq1.py \
  --oud-parquet data/oud_left.parquet \
  --control-parquet data/control_left.parquet \
  --checkpoint-dir checkpoints/stress_encoder \
  --resilience-groups-json resilience_groups.json
PYTHONPATH=. python stat_analysis/rq2.py --emb-npz emb_data.npz --labels-csv labels.csv
PYTHONPATH=. python feature_extraction/extract_h5_window_features.py \
  --input-dir converted_h5 \
  --output-path runs/window_features.parquet
PYTHONPATH=. python feature_extraction/filter_feature_parquets.py \
  --input-dir runs \
  --output-dir filtered \
  --files window_features.parquet
```



## Notes

- The current main model is the dual-path frozen-stress craving model in `main_model/optuna_craving_frozen_stress.py`.
- The baseline entry point is `baselines/optuna_craving_baselines.py`.
- The strict 303-feature RQ1 rerun is `stat_analysis/rq1.py`.
- The RQ2 script is a compact standalone implementation of the PLS-based representational alignment analysis discussed in the paper.
- The `feature_extraction/` scripts are dataset-agnostic public-release versions of the feature extraction and filtering pipeline. They intentionally avoid hardcoded local paths and dataset-specific naming conventions.
- The shared modeling package released with this code bundle is named `core`.
