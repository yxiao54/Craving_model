#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


VARIANTS = {
    "all_gap10_mean": {"window_sec": 10.0},
    "good_bad_gap10_mean": {"window_sec": 10.0},
    "stress_gap10": {"window_sec": 10.0},
    "postlast15": {"window_sec": 15.0},
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract post-gap HR stat embeddings from left-hand OUDLab HDF5.")
    parser.add_argument("--participants-dir", type=str, default=str(root / "converted_h5" / "oud_left"))
    parser.add_argument("--output-dir", type=str, default=str(root / "data"))
    parser.add_argument("--shape-points", type=int, default=0)
    return parser.parse_args()


def decode_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def safe_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float32)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / denom)


def compute_stat_embedding(curve: np.ndarray, dt_sec: float) -> np.ndarray | None:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if len(curve) == 0:
        return None
    pos = np.maximum(curve, 0.0)
    neg = np.minimum(curve, 0.0)
    abs_curve = np.abs(curve)
    peak_idx = int(np.argmax(curve))
    trough_idx = int(np.argmin(curve))
    zero_crossings = int(np.sum(np.diff(np.signbit(curve.astype(np.float64))) != 0))
    curvature = float(np.mean(np.abs(np.diff(curve, n=2)))) if len(curve) >= 3 else 0.0
    return np.asarray(
        [
            float(np.mean(curve)),
            float(np.std(curve)),
            float(np.max(curve)),
            float(np.min(curve)),
            float(np.trapz(curve, dx=dt_sec)),
            float(np.trapz(abs_curve, dx=dt_sec)),
            float(np.trapz(pos, dx=dt_sec)),
            float(np.trapz(neg, dx=dt_sec)),
            float(curve[-1] - curve[0]),
            safe_slope(curve),
            float(peak_idx * dt_sec),
            float(trough_idx * dt_sec),
            float(curve[peak_idx]),
            float(curve[trough_idx]),
            float(zero_crossings),
            curvature,
        ],
        dtype=np.float32,
    )


def slice_signal(hr: np.ndarray, block_start: int, block_end: int, seg_start: int, seg_end: int) -> np.ndarray:
    seg_start = max(seg_start, block_start)
    seg_end = min(seg_end, block_end)
    if seg_end <= seg_start:
        return np.zeros(0, dtype=np.float32)
    start_idx = int(np.floor((seg_start - block_start) / (block_end - block_start) * len(hr)))
    end_idx = int(np.ceil((seg_end - block_start) / (block_end - block_start) * len(hr)))
    start_idx = max(0, min(start_idx, len(hr) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(hr)))
    return hr[start_idx:end_idx]


def compute_baseline(events, hr, block_start, block_end) -> float | None:
    jelly = [slice_signal(hr, block_start, block_end, int(r["start_time_ns"]), int(r["end_time_ns"])) for r in events if decode_str(r["task"]) == "jelly"]
    jelly = [x for x in jelly if len(x) > 0]
    if not jelly:
        return None
    return float(np.mean(np.concatenate(jelly)))


def build_gap_segment(curr_end: int, next_start: int | None, window_sec: float, block_start: int, block_end: int) -> tuple[int, int]:
    target_ns = int(window_sec * 1e9)
    if next_start is None:
        seg_start = curr_end
        seg_end = min(curr_end + target_ns, block_end)
        if (seg_end - seg_start) < target_ns:
            seg_start = max(block_start, seg_end - target_ns)
        return seg_start, seg_end

    gap_end = min(next_start, block_end)
    seg_start = curr_end
    seg_end = min(curr_end + target_ns, gap_end)
    if (seg_end - seg_start) < target_ns:
        seg_start = max(block_start, seg_end - target_ns)
    return seg_start, seg_end


def extract_subject_variants(path: Path) -> tuple[str, dict[str, dict[str, object]] | None]:
    with h5py.File(path, "r") as f:
        participant_id = path.stem.split("-")[-1]
        block = f["block_index"][0]
        block_id = decode_str(block["block_id"])
        block_start = int(block["start_time_ns"])
        block_end = int(block["end_time_ns"])
        events = f["events"]["table"][:]
        hr = np.asarray(f[f"blocks/{block_id}/signals/HR"][:], dtype=np.float32).reshape(-1)
        if len(events) == 0 or len(hr) == 0 or block_end <= block_start:
            return participant_id, None

        baseline_hr = compute_baseline(events, hr, block_start, block_end)
        if baseline_hr is None:
            return participant_id, None
        dt_sec = (block_end - block_start) / 1e9 / float(len(hr))

        rows = []
        for i, row in enumerate(events):
            task = decode_str(row["task"])
            next_start = int(events[i + 1]["start_time_ns"]) if i + 1 < len(events) else None
            rows.append(
                {
                    "task": task,
                    "start_time_ns": int(row["start_time_ns"]),
                    "end_time_ns": int(row["end_time_ns"]),
                    "next_start_ns": next_start,
                }
            )

        variant_payload = {}
        for variant, cfg in VARIANTS.items():
            embeddings = []
            meta = {"segments": []}
            if variant == "postlast15":
                selected = [rows[-1]]
            elif variant == "stress_gap10":
                selected = [r for r in rows if r["task"] == "stress"]
            elif variant == "good_bad_gap10_mean":
                selected = [r for r in rows if r["task"] in {"good", "bad"}]
            else:
                selected = rows

            per_task_embeddings: dict[str, list[np.ndarray]] = {"good": [], "bad": []}

            for item in selected:
                seg_start, seg_end = build_gap_segment(
                    curr_end=item["end_time_ns"],
                    next_start=item["next_start_ns"],
                    window_sec=cfg["window_sec"],
                    block_start=block_start,
                    block_end=block_end,
                )
                segment_hr = slice_signal(hr, block_start, block_end, seg_start, seg_end)
                if len(segment_hr) == 0:
                    continue
                curve = segment_hr - baseline_hr
                emb = compute_stat_embedding(curve, dt_sec)
                if emb is None:
                    continue
                embeddings.append(emb)
                if variant == "good_bad_gap10_mean" and item["task"] in per_task_embeddings:
                    per_task_embeddings[item["task"]].append(emb)
                meta["segments"].append(
                    {
                        "task": item["task"],
                        "segment_duration_sec": float((seg_end - seg_start) / 1e9),
                        "start_offset_from_task_end_sec": float((seg_start - item["end_time_ns"]) / 1e9),
                        "end_offset_from_task_end_sec": float((seg_end - item["end_time_ns"]) / 1e9),
                    }
                )

            if not embeddings:
                continue
            if variant == "good_bad_gap10_mean":
                base_dim = int(embeddings[0].shape[0])
                task_means: dict[str, np.ndarray] = {}
                for task_name in ["good", "bad"]:
                    task_embs = per_task_embeddings[task_name]
                    if task_embs:
                        task_means[task_name] = np.mean(np.stack(task_embs, axis=0), axis=0).astype(np.float32)
                    else:
                        task_means[task_name] = np.zeros(base_dim, dtype=np.float32)
            else:
                mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            meta["n_segments"] = len(embeddings)
            meta["baseline_hr"] = baseline_hr
            if variant == "good_bad_gap10_mean":
                variant_payload[variant] = {
                    "good_embedding": task_means["good"],
                    "bad_embedding": task_means["bad"],
                    "meta": meta,
                }
            else:
                variant_payload[variant] = {
                    "embedding": mean_emb,
                    "meta": meta,
                }

        return participant_id, variant_payload or None


def main() -> None:
    args = parse_args()
    participants_dir = Path(args.participants_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = {name: {} for name in VARIANTS}
    summary = {
        "participants_dir": str(participants_dir),
        "n_subjects_total": 0,
        "variants": {name: {"n_subjects_written": 0, "subjects_skipped": [], "subjects": {}} for name in VARIANTS},
    }

    for path in sorted(participants_dir.glob("sub-OUDLab-*.h5")):
        summary["n_subjects_total"] += 1
        participant_id, variant_payload = extract_subject_variants(path)
        if variant_payload is None:
            for name in VARIANTS:
                summary["variants"][name]["subjects_skipped"].append(str(participant_id))
            continue
        for name in VARIANTS:
            if name not in variant_payload:
                summary["variants"][name]["subjects_skipped"].append(str(participant_id))
                continue
            if name == "good_bad_gap10_mean":
                payloads[name][str(participant_id)] = {
                    "good_text": "",
                    "bad_text": "",
                    "good_embedding": variant_payload[name]["good_embedding"],
                    "bad_embedding": variant_payload[name]["bad_embedding"],
                }
            else:
                vec = variant_payload[name]["embedding"]
                payloads[name][str(participant_id)] = {
                    "good_text": "",
                    "bad_text": "",
                    "good_embedding": vec,
                    "bad_embedding": vec.copy(),
                }
            summary["variants"][name]["n_subjects_written"] += 1
            summary["variants"][name]["subjects"][str(participant_id)] = variant_payload[name]["meta"]

    for name, payload in payloads.items():
        out = output_dir / f"hr_{name}_embeddings.pickle"
        with out.open("wb") as f:
            pickle.dump(payload, f)
        print(f"wrote {out}")

    summary_path = output_dir / "hr_postgap_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
