#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extraction.common import (
    choose_event_label,
    decode_if_bytes,
    extract_acc_features,
    extract_bvp_features,
    extract_eda_features,
    extract_hr_features,
    extract_temp_features,
    get_1d_signal,
    get_slice,
    load_events,
    sliding_windows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sliding-window physiological features from converted HDF5 files."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--step-sec", type=float, default=15.0)
    parser.add_argument("--min-window-sec", type=float, default=30.0)
    parser.add_argument("--label-mode", type=str, choices=["majority", "midpoint"], default="majority")
    parser.add_argument("--eda-process-sr", type=float, default=64.0)
    parser.add_argument("--include-partial-last-window", action="store_true")
    return parser.parse_args()


def infer_subject_id(h5_path: Path) -> str:
    parts = h5_path.stem.split("-")
    return parts[-1] if parts else h5_path.stem


def infer_group_name(h5_path: Path) -> str:
    parent = h5_path.parent.name.lower()
    if parent.startswith("oud") or "patient" in parent:
        return "group_a"
    if parent.startswith("control") or parent.startswith("hc"):
        return "group_b"
    return "unknown"


def infer_side(h5_path: Path) -> str:
    parent = h5_path.parent.name.lower()
    if parent.endswith("_left"):
        return "left"
    if parent.endswith("_right"):
        return "right"
    return "unknown"


def estimate_sampling_rate(n_samples: int, start_ns: int, end_ns: int) -> float | None:
    duration_sec = (end_ns - start_ns) / 1e9
    if n_samples <= 1 or duration_sec <= 0:
        return None
    return float(n_samples / duration_sec)


def get_block_times(h5: h5py.File, block_id: str) -> tuple[int, int]:
    block_index = h5["block_index"][:]
    for row in block_index:
        row_block_id = decode_if_bytes(row["block_id"])
        if row_block_id == block_id:
            return int(row["start_time_ns"]), int(row["end_time_ns"])
    raise KeyError(f"Missing block_index row for {block_id}")


def infer_sampling_rates(block_signals: h5py.Group, start_ns: int, end_ns: int) -> dict[str, float]:
    rates: dict[str, float] = {}
    for key in ["ACC", "BVP", "EDA", "HR", "TEMP"]:
        if key not in block_signals:
            continue
        sr = estimate_sampling_rate(int(block_signals[key].shape[0]), start_ns, end_ns)
        if sr is not None:
            rates[key] = sr
    return rates


def process_subject(h5_path: Path, args) -> pd.DataFrame | None:
    rows = []
    with h5py.File(h5_path, "r") as h5:
        participant_id = infer_subject_id(h5_path)
        global_subject_id = h5_path.stem
        group_name = infer_group_name(h5_path)
        side = infer_side(h5_path)
        events = load_events(h5)

        for block_id in sorted(h5["blocks"].keys()):
            block = h5["blocks"][block_id]
            sigs = block["signals"]
            start_ns, end_ns = get_block_times(h5, block_id)
            duration_sec = (end_ns - start_ns) / 1e9
            if duration_sec < args.min_window_sec:
                continue

            sr = infer_sampling_rates(sigs, start_ns, end_ns)
            if not sr:
                continue

            hr = get_1d_signal(sigs["HR"]) if "HR" in sigs else None
            bvp = get_1d_signal(sigs["BVP"]) if "BVP" in sigs else None
            eda = get_1d_signal(sigs["EDA"]) if "EDA" in sigs else None
            temp = get_1d_signal(sigs["TEMP"]) if "TEMP" in sigs else None
            acc = sigs["ACC"][:].astype(np.float32) if "ACC" in sigs else None

            windows = sliding_windows(
                duration_sec=duration_sec,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                min_window_sec=args.min_window_sec,
                include_partial_last=args.include_partial_last_window,
            )

            for win_start_sec, win_end_sec in tqdm(windows, desc=f"{global_subject_id}-{block_id}", leave=False):
                win_start_ns = start_ns + int(round(win_start_sec * 1e9))
                win_end_ns = start_ns + int(round(win_end_sec * 1e9))
                task, stress, craving = choose_event_label(win_start_ns, win_end_ns, events, args.label_mode)

                feat = {}
                if hr is not None and "HR" in sr:
                    feat.update(extract_hr_features(get_slice(hr, sr["HR"], win_start_sec, win_end_sec), sr["HR"]))
                if bvp is not None and "BVP" in sr:
                    feat.update(extract_bvp_features(get_slice(bvp, sr["BVP"], win_start_sec, win_end_sec), sr["BVP"]))
                if eda is not None and "EDA" in sr:
                    feat.update(
                        extract_eda_features(
                            get_slice(eda, sr["EDA"], win_start_sec, win_end_sec),
                            sr["EDA"],
                            args.eda_process_sr,
                        )
                    )
                if temp is not None and "TEMP" in sr:
                    feat.update(extract_temp_features(get_slice(temp, sr["TEMP"], win_start_sec, win_end_sec), sr["TEMP"]))
                if acc is not None and "ACC" in sr:
                    feat.update(extract_acc_features(get_slice(acc, sr["ACC"], win_start_sec, win_end_sec), sr["ACC"]))
                if not feat:
                    continue

                rows.append(
                    {
                        "participant_id": participant_id,
                        "global_subject_id": global_subject_id,
                        "group": group_name,
                        "side": side,
                        "block_id": block_id,
                        "window_start_ns": win_start_ns,
                        "window_end_ns": win_end_ns,
                        "window_start_sec": float(win_start_sec),
                        "window_end_sec": float(win_end_sec),
                        "window_duration_sec": float(win_end_sec - win_start_sec),
                        "task": task,
                        "stress": stress,
                        "craving": craving,
                        **feat,
                    }
                )

    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    frames = []
    for h5_path in sorted(input_dir.rglob("*.h5")):
        df = process_subject(h5_path, args)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No valid feature rows were extracted.")

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_parquet(output_path, index=False)
    print(f"wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
