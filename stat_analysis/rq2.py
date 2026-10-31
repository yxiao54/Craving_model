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


def fit_one_component_pls(X_scaled, y):
    """Fit one-component PLS and return the participant-level latent score."""
    pls = PLSRegression(n_components=1)
    return np.asarray(pls.fit_transform(X_scaled, y)[0][:, 0], dtype=float)


def refit_permutation_test(X_scaled, y, observed_r, n_perm, seed):
    """Permute the target and refit PLS for every permutation.

    The embedding matrix is fixed and standardized once. Re-fitting the PLS
    model after each target permutation incorporates the supervised
    projection-learning step into the null distribution.
    """
    rng = np.random.default_rng(seed)
    perm_rs = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        z_perm = fit_one_component_pls(X_scaled, y_perm)
        perm_rs[i] = float(spearmanr(z_perm, y_perm).statistic)
    p_perm = float(
        (1.0 + np.sum(np.abs(perm_rs) >= abs(observed_r))) / (n_perm + 1.0)
    )
    return p_perm


def run_pls(X, y, name, n_perm, seed):
    Xs = StandardScaler().fit_transform(X)
    z = fit_one_component_pls(Xs, y)
    r, p = spearmanr(z, y)
    p_perm = refit_permutation_test(Xs, y, r, n_perm=n_perm, seed=seed)
    return {
        "representation": name,
        "method": "one-component PLS with refit permutation test",
        "spearman_r": float(r),
        "spearman_p": float(p),
        "perm_p": float(p_perm),
        "n_subjects": int(len(y)),
        "latent_dim": 1,
        "n_permutations": int(n_perm),
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
