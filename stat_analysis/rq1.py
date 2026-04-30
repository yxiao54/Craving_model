#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")


ROOT = Path("/home/yi124/process_data/repro_eval_2026_oudlab13")
OUD_PARQUET = ROOT / "data" / "filtered_flat30" / "oud_left_flat30.parquet"
CONTROL_PARQUET = ROOT / "data" / "filtered_flat30" / "control_left_flat30.parquet"
CHECKPOINT_DIR = ROOT / "checkpoints" / "stress_pretrain_left_oud_control"
OUT_DIR = ROOT / "results" / "rq1_rerun_303_strict"
os.makedirs(OUT_DIR, exist_ok=True)


MIN_WINDOWS = 3
MIN_STD = 1e-6
Z_CLIP = 5.0
BASELINE_TASK = "jelly"

RESILIENT_HIGH = {
    "8876",
    "9929",
    "8814",
    "8803",
    "9967",
    "8852",
    "9956",
    "9984",
    "8829",
    "9973",
    "9925",
    "9945v2",
    "9945",
}
RESILIENT_LOW = {
    "9969",
    "8847",
    "9933",
    "9941",
    "9913",
    "9915",
    "9952",
    "8840",
    "9991",
    "8832",
    "9933v2",
    "9948",
    "9915v2",
    "8804",
    "9973v2",
    "8898",
}


def load_303_feature_columns() -> list[str]:
    ckpt_path = next(CHECKPOINT_DIR.glob("*/best_encoder.pt"))
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return list(payload["feature_columns"])


def majority_vote(x: pd.Series) -> float:
    x = x.dropna().astype(int)
    if len(x) == 0:
        return np.nan
    return int(x.sum() >= len(x) / 2)


def global_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std()
        if pd.isna(sd) or sd <= 0:
            df[c] = 0.0
        else:
            df[c] = ((df[c] - mu) / sd).clip(-Z_CLIP, Z_CLIP)
    return df


def gee_fit(df: pd.DataFrame, y: str, x: str):
    cols_needed = ["user", "Group", "n_windows", y, x]
    sub = df[cols_needed].dropna()
    if len(sub) < 10 or sub[y].std() < MIN_STD:
        return None

    if x == "Resilience":
        formula = f"{y} ~ C(Resilience) + C(Group) + n_windows"
    else:
        formula = f"{y} ~ {x} + C(Group) + n_windows"

    try:
        model = smf.gee(
            formula,
            groups="user",
            data=sub,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        res = model.fit(cov_type="robust")

        if x == "Resilience":
            key = "C(Resilience)[T.Low]"
            if key not in res.params.index:
                candidates = [k for k in res.params.index if k.startswith("C(Resilience)")]
                key = candidates[0] if candidates else None
        else:
            key = x
            if key not in res.params.index:
                candidates = [k for k in res.params.index if k.endswith(f"{x}")]
                key = candidates[0] if candidates else None

        if key is None or key not in res.params.index:
            return None

        return {
            "coef": float(res.params[key]),
            "se": float(res.bse[key]),
            "p": float(res.pvalues[key]),
            "ci_low": float(res.conf_int().loc[key][0]),
            "ci_high": float(res.conf_int().loc[key][1]),
            "n_obs": int(len(sub)),
        }
    except Exception:
        return None


def build_input_windows() -> pd.DataFrame:
    oud = pd.read_parquet(OUD_PARQUET)
    control = pd.read_parquet(CONTROL_PARQUET)
    win = pd.concat([oud, control], ignore_index=True)
    win = win[win["side"].astype(str) == "left"].copy()
    win["user"] = win["participant_id"].astype(str)
    if "window_start_sec" not in win.columns:
        raise KeyError("window_start_sec is required for task aggregation.")
    return win


def orient_pc1_positive_for_stress(task: pd.DataFrame, col: str) -> pd.Series:
    out = gee_fit(task, col, "stress")
    if out is not None and out["coef"] < 0:
        return -task[col]
    return task[col]


def main() -> None:
    win = build_input_windows()
    feature_cols = load_303_feature_columns()

    missing = [c for c in feature_cols if c not in win.columns]
    if missing:
        raise KeyError(f"Missing required 303 feature columns: {missing}")

    stress_col = "stress" if "stress" in win.columns else None
    craving_col = "craving" if "craving" in win.columns else None

    meta_cols = {"user", "task", "window_start_sec", "segment_start_utc", "ema_time_utc"}
    if stress_col:
        meta_cols.add(stress_col)
    if craving_col:
        meta_cols.add(craving_col)

    # Strict 303-feature rerun: override old generic feature discovery with final model feature contract.
    win = win.copy()
    win = global_zscore(win, feature_cols)

    agg = {f"{f}_mean": (f, "mean") for f in feature_cols}
    agg["n_windows"] = ("window_start_sec", "count")
    task = win.groupby(["user", "task"], as_index=False).agg(**agg)
    task = task[task["n_windows"] >= MIN_WINDOWS].copy()

    base_cols = [f"{f}_mean" for f in feature_cols]
    base = task[task["task"] == BASELINE_TASK][["user"] + base_cols].copy()
    base = base.rename(columns={c: f"{c}_baseline" for c in base.columns if c != "user"})
    task = task.merge(base, on="user", how="left")
    for f in feature_cols:
        task[f"{f}_delta"] = task[f"{f}_mean"] - task[f"{f}_mean_baseline"]

    oud_users = set(pd.read_parquet(OUD_PARQUET)["participant_id"].astype(str).unique())
    task["Group"] = task["user"].map(lambda u: "OUD" if u in oud_users else "Control")
    task["Resilience"] = task["user"].map(lambda u: "High" if u in RESILIENT_HIGH else ("Low" if u in RESILIENT_LOW else np.nan))

    if stress_col:
        s = win.groupby(["user", "task"])[stress_col].apply(majority_vote).reset_index(name="stress")
        task = task.merge(s, on=["user", "task"], how="left")
    else:
        task["stress"] = np.nan

    if craving_col:
        c = win.groupby(["user", "task"])[craving_col].apply(majority_vote).reset_index(name="craving")
        task = task.merge(c, on=["user", "task"], how="left")
    else:
        task["craving"] = np.nan

    task["Group"] = pd.Categorical(task["Group"], ["Control", "OUD"])
    task["Resilience"] = pd.Categorical(task["Resilience"], ["High", "Low"])
    task.to_csv(OUT_DIR / "task_level_from_raw_303.csv", index=False)

    systems = {
        "Cardiovascular": [f"{c}_delta" for c in feature_cols if c.startswith(("BVP_", "HR_", "HRV_", "PPG_"))],
        "Electrodermal": [f"{c}_delta" for c in feature_cols if c.startswith("EDA_")],
        "Movement": [f"{c}_delta" for c in feature_cols if c.startswith("ACC_")],
        "Thermoregulation": [f"{c}_delta" for c in feature_cols if c.startswith("TEMP_")],
    }

    factors = ["stress", "craving", "Resilience"]
    rows = []
    for x in factors:
        for system, cols in systems.items():
            cols = [c for c in cols if c in task.columns and task[c].std() > MIN_STD]
            if len(cols) < 2:
                continue

            z = (task[cols] - task[cols].mean()) / (task[cols].std() + 1e-8)
            z = z.fillna(0.0)

            pca = PCA(n_components=1, random_state=0)
            task[f"{system}_pc1"] = pca.fit_transform(z.values).ravel()
            task[f"{system}_pc1"] = orient_pc1_positive_for_stress(task, f"{system}_pc1")

            out = gee_fit(task, f"{system}_pc1", x)
            if out:
                rows.append(
                    {
                        "factor": x,
                        "system": system,
                        "explained_var": float(pca.explained_variance_ratio_[0]),
                        "n_features": int(len(cols)),
                        **out,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "rq1_system_pca_all_factors_303.csv", index=False)

    summary = {
        "input_parquets": [str(OUD_PARQUET), str(CONTROL_PARQUET)],
        "baseline_task": BASELINE_TASK,
        "min_windows": MIN_WINDOWS,
        "n_model_features": len(feature_cols),
        "n_total_task_rows": int(len(task)),
        "n_oud_users": int(task.loc[task["Group"] == "OUD", "user"].nunique()),
        "n_control_users": int(task.loc[task["Group"] == "Control", "user"].nunique()),
        "n_resilience_users": int(task["Resilience"].notna().sum()),
        "system_feature_counts": {k: len(v) for k, v in systems.items()},
        "notes": [
            "Strict replica of the old RQ1 pipeline structure with field mapping to current parquets.",
            "The only intentional change is replacing generic feature discovery with the final 303-feature model contract.",
            "Current columns use craving instead of Craving_bin and participant_id instead of user.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
