from __future__ import annotations

import ast
import json

import flirt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal


FLIRT_FREQ_NS = {
    "ACC": "31250000ns",
    "BVP": "15625000ns",
    "EDA": "250000000ns",
    "TEMP": "250000000ns",
    "HR": "1s",
}


def parse_sampling_rates(attr):
    if isinstance(attr, dict):
        return attr
    if isinstance(attr, bytes):
        attr = attr.decode("utf-8")
    if isinstance(attr, str):
        try:
            return json.loads(attr)
        except Exception:
            try:
                return ast.literal_eval(attr)
            except Exception:
                return None
    return None


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def get_1d_signal(ds):
    arr = ds[:]
    if arr.ndim == 1:
        return arr.astype(np.float32)
    return arr[:, 0].astype(np.float32)


def sliding_windows(duration_sec, window_sec, step_sec, min_window_sec, include_partial_last):
    t = 0.0
    while t + window_sec <= duration_sec:
        yield t, t + window_sec
        t += step_sec

    if not include_partial_last or duration_sec <= 0:
        return

    last_start = max(0.0, duration_sec - window_sec)
    if t < duration_sec and duration_sec - last_start >= min_window_sec and last_start + window_sec > duration_sec:
        yield last_start, duration_sec


def get_slice(sig, sr, start_sec, end_sec):
    i0 = max(int(round(start_sec * sr)), 0)
    i1 = min(int(round(end_sec * sr)), len(sig))
    return sig[i0:i1]


def maybe_resample_signal(x, orig_sr, target_sr):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x, orig_sr
    if target_sr is None or target_sr <= 0 or abs(float(target_sr) - float(orig_sr)) < 1e-8:
        return x, float(orig_sr)

    try:
        resampled = nk.signal_resample(
            x,
            sampling_rate=float(orig_sr),
            desired_sampling_rate=float(target_sr),
            method="poly",
        )
        return np.asarray(resampled, dtype=np.float32), float(target_sr)
    except Exception:
        return x, float(orig_sr)


def compute_stats(x, prefix):
    if len(x) == 0:
        return {}

    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}

    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-8:
        skew = 0.0
        kurt = 0.0
    else:
        centered = x - mean
        skew = float((centered ** 3).mean() / (std ** 3 + 1e-8))
        kurt = float((centered ** 4).mean() / (std ** 4 + 1e-8))

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": std,
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_range": float(x.max() - x.min()),
        f"{prefix}_p20": float(np.percentile(x, 20)),
        f"{prefix}_p80": float(np.percentile(x, 80)),
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
    }


def compute_shape_features(x, prefix):
    if len(x) == 0:
        return {}

    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}

    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    mean_abs = float(np.mean(abs_x))
    peak = float(np.max(abs_x))
    waveform_length = float(np.sum(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    zero_crossings = float(np.count_nonzero(np.diff(np.signbit(x)))) if len(x) > 1 else 0.0

    return {
        f"{prefix}_rms": rms,
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_mad": float(np.median(np.abs(x - np.median(x)))),
        f"{prefix}_peak_abs": peak,
        f"{prefix}_crest_factor": float(peak / (rms + 1e-8)),
        f"{prefix}_shape_factor": float(rms / (mean_abs + 1e-8)),
        f"{prefix}_impulse_factor": float(peak / (mean_abs + 1e-8)),
        f"{prefix}_waveform_length": waveform_length,
        f"{prefix}_zero_crossings": zero_crossings,
    }


def compute_spectral_features(x, sr, prefix):
    if len(x) < max(8, int(sr)):
        return {}

    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) < max(8, int(sr)):
        return {}

    try:
        freqs, psd = signal.welch(x, fs=sr, nperseg=min(len(x), max(256, int(sr * 4))))
    except Exception:
        return {}

    if len(freqs) == 0 or len(psd) == 0:
        return {}

    total_power = float(np.trapezoid(psd, freqs))
    psd_sum = float(np.sum(psd)) + 1e-12
    centroid = float(np.sum(freqs * psd) / psd_sum)
    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
    p_norm = psd / psd_sum
    spectral_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

    def band_power(low, high):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    return {
        f"{prefix}_spec_power": total_power,
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_spec_peak_freq": peak_freq,
        f"{prefix}_spec_bandwidth": bandwidth,
        f"{prefix}_spec_entropy": spectral_entropy,
        f"{prefix}_band_0_0.15": band_power(0.0, 0.15),
        f"{prefix}_band_0.15_0.5": band_power(0.15, 0.5),
        f"{prefix}_band_0.5_1.5": band_power(0.5, 1.5),
        f"{prefix}_band_1.5_4.0": band_power(1.5, 4.0),
    }


def frame_to_feature_dict(df, prefix):
    if df is None or df.empty:
        return {}
    row = df.reset_index(drop=True).iloc[0]
    out = {}
    for col, value in row.items():
        out[f"{prefix}_{col}"] = float(value) if pd.notna(value) else np.nan
    return out


def maybe_flirt_features(values, sr, column_name, prefix):
    if len(values) == 0:
        return {}

    freq_key = None
    for key in FLIRT_FREQ_NS:
        if prefix.startswith(key):
            freq_key = key
            break
    if freq_key is None:
        return {}

    try:
        df = pd.DataFrame({column_name: values})
        df = df.set_index(pd.date_range(start=0, periods=len(df), freq=FLIRT_FREQ_NS[freq_key]))
        df = df.fillna(0.0)
        feat_df = flirt.get_acc_features(
            df,
            window_length=len(df) / sr,
            window_step_size=len(df) / sr,
            data_frequency=sr,
        )
        return frame_to_feature_dict(feat_df, prefix)
    except Exception:
        return {}


def extract_bvp_features(bvp_win, bvp_sr):
    feat = {}
    feat.update(compute_stats(bvp_win, "BVP"))
    feat.update(compute_shape_features(bvp_win, "BVP"))
    feat.update(compute_spectral_features(bvp_win, bvp_sr, "BVP"))
    feat["BVP_energy"] = float(np.sum(bvp_win ** 2)) if len(bvp_win) > 0 else 0.0

    if len(bvp_win) == 0:
        return feat

    try:
        cleaned = nk.ppg_clean(bvp_win, sampling_rate=bvp_sr)
        peaks_dict, _ = nk.ppg_peaks(cleaned, sampling_rate=bvp_sr)
        peak_mask = np.asarray(peaks_dict["PPG_Peaks"])
        peak_idx = np.flatnonzero(peak_mask)
        quality = nk.ppg_quality(cleaned, sampling_rate=bvp_sr, method="disimilarity")
        quality = np.asarray(quality).reshape(-1)
        feat["PPG_quality_sum_abs"] = float(np.sum(np.abs(quality)))
        feat["PPG_quality_mean"] = float(np.mean(quality)) if len(quality) > 0 else np.nan
        feat["PPG_num_peaks"] = float(len(peak_idx))
        if len(peak_idx) >= 2:
            hrv_t = nk.hrv_time({"PPG_Peaks": peak_idx}, sampling_rate=bvp_sr)
            for k, v in hrv_t.iloc[0].items():
                feat[k] = float(v) if pd.notna(v) else np.nan
            hrv_f = nk.hrv_frequency({"PPG_Peaks": peak_idx}, sampling_rate=bvp_sr, psd_method="lomb")
            for k, v in hrv_f.iloc[0].items():
                feat[k] = float(v) if pd.notna(v) else np.nan
    except Exception:
        pass

    feat.update(maybe_flirt_features(bvp_win, bvp_sr, "BVP", "BVP"))
    return feat


def extract_eda_features(eda_win, eda_sr, eda_process_sr):
    feat = {}
    feat.update(compute_stats(eda_win, "EDA"))
    feat.update(compute_shape_features(eda_win, "EDA"))
    feat.update(compute_spectral_features(eda_win, eda_sr, "EDA"))
    feat["EDA_auc"] = float(np.trapezoid(eda_win)) if len(eda_win) > 0 else 0.0
    feat["EDA_slope"] = float((eda_win[-1] - eda_win[0]) / max(len(eda_win), 1)) if len(eda_win) > 1 else 0.0
    feat["EDA_native_sr"] = float(eda_sr)

    try:
        eda_proc, proc_sr = maybe_resample_signal(eda_win, eda_sr, eda_process_sr)
        feat["EDA_process_sr"] = float(proc_sr)
        cleaned = nk.eda_clean(eda_proc, sampling_rate=proc_sr, method="neurokit")
        processed, _ = nk.eda_process(cleaned, sampling_rate=proc_sr, method="neurokit")
        tonic = processed["EDA_Tonic"].to_numpy()
        phasic = processed["EDA_Phasic"].to_numpy()
        feat.update(compute_stats(tonic, "EDA_tonic"))
        feat.update(compute_shape_features(tonic, "EDA_tonic"))
        feat.update(compute_spectral_features(tonic, proc_sr, "EDA_tonic"))
        feat.update(compute_stats(phasic, "EDA_phasic"))
        feat.update(compute_shape_features(phasic, "EDA_phasic"))
        feat.update(compute_spectral_features(phasic, proc_sr, "EDA_phasic"))
        feat["EDA_onset"] = float(processed["SCR_Onsets"].sum())
        feat["EDA_peaks"] = float(processed["SCR_Peaks"].sum())
        feat["EDA_recovery"] = float(processed["SCR_Recovery"].sum())

        for src_col, out_name in [
            ("SCR_Height", "EDA_height"),
            ("SCR_Amplitude", "EDA_amplitude"),
            ("SCR_RiseTime", "EDA_risetime"),
            ("SCR_RecoveryTime", "EDA_recoverytime"),
        ]:
            values = processed[src_col].to_numpy()
            values = values[np.isfinite(values) & (values != 0)]
            feat[f"{out_name}_mean"] = float(np.mean(values)) if len(values) > 0 else np.nan
            feat[f"{out_name}_std"] = float(np.std(values)) if len(values) > 0 else np.nan

        feat.update(maybe_flirt_features(cleaned, proc_sr, "EDA", "RawEDA"))
        feat.update(maybe_flirt_features(tonic, proc_sr, "tonic", "Tonic"))
        feat.update(maybe_flirt_features(phasic, proc_sr, "phasic", "Phasic"))
    except Exception:
        pass

    feat.update(maybe_flirt_features(eda_win, eda_sr, "EDA", "EDA"))
    return feat


def extract_acc_features(acc_win, acc_sr):
    feat = {}
    if len(acc_win) == 0:
        return feat

    mag = np.linalg.norm(acc_win, axis=1)
    feat.update(compute_stats(mag, "ACC"))
    feat.update(compute_shape_features(mag, "ACC"))
    feat.update(compute_spectral_features(mag, acc_sr, "ACC"))
    feat["ACC_energy"] = float(np.sum(mag ** 2))
    feat["ACC_diff"] = float(np.mean(np.abs(np.diff(mag)))) if len(mag) > 1 else 0.0
    feat["ACC_sma"] = float(np.mean(np.abs(acc_win[:, 0]) + np.abs(acc_win[:, 1]) + np.abs(acc_win[:, 2])))

    axis_names = ["ACC_x", "ACC_y", "ACC_z"]
    for axis_idx, axis_name in enumerate(axis_names):
        axis_values = acc_win[:, axis_idx]
        feat.update(compute_stats(axis_values, axis_name))
        feat.update(compute_shape_features(axis_values, axis_name))

    if len(acc_win) > 1:
        feat["ACC_xy_corr"] = (
            float(np.corrcoef(acc_win[:, 0], acc_win[:, 1])[0, 1])
            if np.std(acc_win[:, 0]) > 0 and np.std(acc_win[:, 1]) > 0 else np.nan
        )
        feat["ACC_xz_corr"] = (
            float(np.corrcoef(acc_win[:, 0], acc_win[:, 2])[0, 1])
            if np.std(acc_win[:, 0]) > 0 and np.std(acc_win[:, 2]) > 0 else np.nan
        )
        feat["ACC_yz_corr"] = (
            float(np.corrcoef(acc_win[:, 1], acc_win[:, 2])[0, 1])
            if np.std(acc_win[:, 1]) > 0 and np.std(acc_win[:, 2]) > 0 else np.nan
        )

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        feat.update(maybe_flirt_features(acc_win[:, axis_idx], acc_sr, axis_name, f"ACC_{axis_name}"))
    feat.update(maybe_flirt_features(mag, acc_sr, "mag", "ACC_mag"))
    return feat


def extract_temp_features(temp_win, temp_sr):
    feat = compute_stats(temp_win, "TEMP")
    feat.update(compute_shape_features(temp_win, "TEMP"))
    feat.update(compute_spectral_features(temp_win, temp_sr, "TEMP"))
    feat.update(maybe_flirt_features(temp_win, temp_sr, "TEMP", "TEMP"))
    return feat


def extract_hr_features(hr_win, hr_sr):
    feat = compute_stats(hr_win, "HR")
    feat.update(compute_shape_features(hr_win, "HR"))
    feat.update(compute_spectral_features(hr_win, hr_sr, "HR"))
    feat["HR_diff"] = float(np.mean(np.abs(np.diff(hr_win)))) if len(hr_win) > 1 else 0.0
    feat.update(maybe_flirt_features(hr_win, hr_sr, "HR", "HR"))
    return feat


def load_events(h5_file):
    if "events" not in h5_file or "table" not in h5_file["events"]:
        return []

    table = h5_file["events"]["table"][:]
    events = []
    for row in table:
        task = decode_if_bytes(row["task"]) if "task" in table.dtype.names else "unknown"
        stress = int(row["stress"]) if "stress" in table.dtype.names else -1
        craving = int(row["craving"]) if "craving" in table.dtype.names else -1
        events.append(
            {
                "start_ns": int(row["start_time_ns"]),
                "end_ns": int(row["end_time_ns"]),
                "task": str(task),
                "stress": stress,
                "craving": craving,
            }
        )
    return events


def choose_event_label(win_start_ns, win_end_ns, events, mode):
    if not events:
        return "unknown", -1, -1

    if mode == "midpoint":
        mid_ns = (win_start_ns + win_end_ns) // 2
        for event in events:
            if event["start_ns"] <= mid_ns <= event["end_ns"]:
                return event["task"], event["stress"], event["craving"]
        return "unknown", -1, -1

    best_event = None
    best_overlap = -1
    for event in events:
        overlap = max(0, min(win_end_ns, event["end_ns"]) - max(win_start_ns, event["start_ns"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_event = event

    if best_event is None or best_overlap <= 0:
        return "unknown", -1, -1
    return best_event["task"], best_event["stress"], best_event["craving"]
