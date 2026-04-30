#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 1D PLS-based representational alignment against HR-AUC."
    )
    parser.add_argument("--emb-npz", type=str, required=True, help="NPZ file containing embedding matrices.")
    parser.add_argument("--labels-csv", type=str, required=True, help="CSV file containing hr_auc.")
    parser.add_argument("--gb-key", type=str, default="X_gb", help="NPZ key for the autobiographical/good+bad representation.")
    parser.add_argument("--baseline-key", type=str, default="X_bl", help="NPZ key for the reference or baseline representation.")
    parser.add_argument("--label-col", type=str, default="hr_auc", help="Label column in the CSV.")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default="rq2_hr_latent_analysis.csv")
    return parser.parse_args()


def spearman_permutation(x, y, n_perm, seed):
    r_obs, p_obs = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rs = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_rs[i], _ = spearmanr(x, y_perm)
    p_perm = float(np.mean(np.abs(perm_rs) >= np.abs(r_obs)))
    return float(r_obs), float(p_obs), p_perm


def run_pls(X, y, name, n_perm, seed):
    Xs = StandardScaler().fit_transform(X)
    pls = PLSRegression(n_components=1)
    z = pls.fit_transform(Xs, y)[0][:, 0]
    r, p, p_perm = spearman_permutation(z, y, n_perm=n_perm, seed=seed)
    return {
        "representation": name,
        "method": "PLS",
        "spearman_r": r,
        "spearman_p": p,
        "perm_p": p_perm,
        "n_subjects": int(len(y)),
        "latent_dim": 1,
    }


def main():
    args = parse_args()

    emb = np.load(Path(args.emb_npz))
    labels = pd.read_csv(Path(args.labels_csv))

    X_gb = emb[args.gb_key]
    X_bl = emb[args.baseline_key]
    y_hr = labels[args.label_col].to_numpy()

    if X_gb.shape[0] != y_hr.shape[0] or X_bl.shape[0] != y_hr.shape[0]:
        raise ValueError("Embedding rows and label rows do not match.")

    results = [
        run_pls(X_gb, y_hr, "Good+Bad", args.n_perm, args.seed),
        run_pls(X_bl, y_hr, "Baseline", args.n_perm, args.seed),
    ]

    out_df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
