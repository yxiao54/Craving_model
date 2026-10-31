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
  - `physiological_modulation.py`
  - `language_recovery_alignment.py`
- `feature_extraction/`
  - `extract_h5_window_features.py`
  - `filter_feature_parquets.py`
- `utils/`
  - `extract_hr_postgap_stat_embeddings.py`

## Running the code

Run scripts from this folder root and set `PYTHONPATH` so the shared `core` package is visible:

```bash
cd public_release_code
PYTHONPATH=. python main_model/train_craving.py --help
PYTHONPATH=. python baselines/train.py --help
```

Typical entry points:

```bash
cd public_release_code
PYTHONPATH=. python stat_analysis/physiological_modulation.py \
  --oud-parquet data/oud_left.parquet \
  --control-parquet data/control_left.parquet \
  --checkpoint-dir checkpoints/stress_encoder \
  --resilience-groups-json resilience_groups.json
PYTHONPATH=. python stat_analysis/language_recovery_alignment.py --emb-npz emb_data.npz --labels-csv labels.csv
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

- Run commands from this folder with `PYTHONPATH=.`.
- `pretrain_stress_encoder.py` prepares the stress encoder; `train_craving.py` trains RETRACE for craving detection; `baselines/train.py` trains comparison models.
- `stat_analysis/physiological_modulation.py` runs the physiological modulation analysis, and `stat_analysis/language_recovery_alignment.py` runs the language--recovery alignment analysis.
