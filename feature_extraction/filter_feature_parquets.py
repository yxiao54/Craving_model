#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLUMNS = {
    "participant_id",
    "global_subject_id",
    "group",
    "side",
    "block_id",
    "window_start_ns",
    "window_end_ns",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "task",
    "stress",
    "craving",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only common high-quality feature columns across window-feature parquets."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--max-nan-rate", type=float, default=0.2)
    parser.add_argument("--min-unique-nonnull", type=int, default=2)
    return parser.parse_args()


def summarize_feature_quality(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        s = df[col]
        nonnull = s.dropna()
        rows.append(
            {
                "column": col,
                "nan_rate": float(s.isna().mean()),
                "nonnull_count": int(nonnull.shape[0]),
                "unique_nonnull": int(nonnull.nunique(dropna=True)),
                "std": float(nonnull.std()) if nonnull.shape[0] > 1 else np.nan,
                "is_constant": bool(nonnull.nunique(dropna=True) < 2),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [input_dir / name for name in args.files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet files: {missing}")

    dfs = {p.name: pd.read_parquet(p) for p in paths}
    common_cols = set.intersection(*(set(df.columns) for df in dfs.values()))
    metadata_cols = [c for c in sorted(common_cols) if c in METADATA_COLUMNS]
    feature_cols = [c for c in sorted(common_cols) if c not in METADATA_COLUMNS]

    per_file_quality = {}
    kept_features = set(feature_cols)
    for name, df in dfs.items():
        quality = summarize_feature_quality(df, feature_cols)
        quality["keep_here"] = (
            (quality["nan_rate"] <= args.max_nan_rate)
            & (quality["unique_nonnull"] >= args.min_unique_nonnull)
            & (~quality["is_constant"])
        )
        per_file_quality[name] = quality
        kept_features &= set(quality.loc[quality["keep_here"], "column"])
        quality.to_csv(output_dir / f"{Path(name).stem}_quality.csv", index=False)

    kept_features = sorted(kept_features)
    dropped_features = sorted(set(feature_cols) - set(kept_features))

    for name, df in dfs.items():
        filtered = df[metadata_cols + kept_features].copy()
        filtered.to_parquet(output_dir / name, index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [p.name for p in paths],
        "n_common_columns": len(common_cols),
        "n_metadata_columns_kept": len(metadata_cols),
        "n_feature_columns_common": len(feature_cols),
        "n_feature_columns_kept": len(kept_features),
        "n_feature_columns_dropped": len(dropped_features),
        "max_nan_rate": args.max_nan_rate,
        "min_unique_nonnull": args.min_unique_nonnull,
        "kept_feature_columns": kept_features,
        "dropped_feature_columns": dropped_features,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        {
            "kept_feature_columns": pd.Series(kept_features),
            "dropped_feature_columns": pd.Series(dropped_features),
        }
    ).to_csv(output_dir / "kept_vs_dropped_columns.csv", index=False)

    print(f"wrote filtered parquets to {output_dir}")
    print(f"common feature columns={len(feature_cols)} kept={len(kept_features)} dropped={len(dropped_features)}")


if __name__ == "__main__":
    main()
