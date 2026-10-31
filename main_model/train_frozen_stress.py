#!/usr/bin/env python3
"""Train RETRACE with the fixed paper configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from core import ExperimentConfig, Trainer, build_model, build_objective
from core.data import load_base_resources, make_dataset, make_loader, make_weighted_sampler
from core.subject_policy import EXCLUDED_TASKS, filter_oudlab_subjects_for_label, summarize_subject_label_types


def parse_args():
    p = argparse.ArgumentParser(description="Train RETRACE with fixed hyperparameters.")
    p.add_argument("--parquet-path", required=True)
    p.add_argument("--embedding-path", required=True)
    p.add_argument("--splits-path", required=True)
    p.add_argument("--stress-encoder-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-subjects-json", default=None)
    p.add_argument("--test-subjects", nargs="*", default=None)
    p.add_argument("--reference-parquet", default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def safe_auc(y_true, y_prob):
    if len(set(map(int, y_true))) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def load_folds(path):
    payload = json.loads(Path(path).read_text())
    return sorted(payload["folds"].items(), key=lambda item: int(item[0]))


def selected_subjects(args):
    if args.test_subjects:
        return {str(x) for x in args.test_subjects}
    if args.test_subjects_json:
        payload = json.loads(Path(args.test_subjects_json).read_text())
        return {str(x) for x in payload.get("subjects", [])}
    return None


def fixed_config(args, input_dim, feature_columns):
    cfg = ExperimentConfig()
    cfg.data.parquet_path = args.parquet_path
    cfg.data.embedding_path = args.embedding_path
    cfg.data.label_name = "craving"
    cfg.data.normalization = "subject_zscore"
    cfg.data.batch_size = 30

    cfg.model.name = "semantic_dualpath_frozen_stress"
    cfg.model.input_dim = input_dim
    cfg.model.feature_columns = feature_columns
    cfg.model.hidden_dim = 384
    cfg.model.gate_hidden_dim = 64
    cfg.model.num_encoder_layers = 2
    cfg.model.num_classifier_layers = 3
    cfg.model.dropout = 0.193
    cfg.model.use_good_embedding = True
    cfg.model.use_bad_embedding = True
    cfg.model.use_input_feature_gate = True
    cfg.model.use_input_film = False
    cfg.model.use_trainable_phys_branch = True
    cfg.model.use_stress_branch = True
    cfg.model.use_logit_scale = False
    cfg.model.use_affine_calibration = False
    cfg.model.use_semantic_uncertainty = False
    cfg.model.use_stress_aux_head = True
    cfg.model.fusion_mode = "sample_gate"
    cfg.model.freeze_stress_encoder = True

    cfg.objective.name = "erm"
    cfg.objective.label_smoothing = 0.0175
    cfg.objective.sparse_gate_weight = 0.00102
    cfg.objective.stress_aux_weight = 0.000596
    cfg.objective.orthogonality_weight = 0.000541
    cfg.objective.vrex_lambda = 0.0
    cfg.objective.semantic_kl_weight = 0.0

    cfg.train.epochs = 24
    cfg.train.lr = 0.00131
    cfg.train.weight_decay = 1.12e-6
    cfg.train.grad_clip = 0.424
    cfg.train.optimizer_name = "adamw"
    cfg.train.scheduler_name = "none"
    cfg.train.seed = args.seed
    return cfg


def checkpoint_for_subject(root, subject):
    root = Path(root)
    direct = root / str(subject) / "best_encoder.pt"
    if direct.exists():
        return str(direct)
    matches = sorted(root.glob(f"fold_*_{subject}/best_encoder.pt"))
    if not matches:
        raise FileNotFoundError(f"No stress checkpoint found for held-out subject {subject}.")
    return str(matches[0])


def validate_checkpoint(path, feature_columns):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("feature_columns")
    if saved is not None and list(saved) != list(feature_columns):
        raise ValueError("Stress checkpoint feature columns do not match the craving features.")


def restrict_features(feature_columns, reference_parquet):
    if reference_parquet is None:
        return list(feature_columns)
    ref = pd.read_parquet(reference_parquet)
    meta = {"participant_id", "global_subject_id", "cohort", "block_id", "task", "stress", "craving"}
    available = {c for c in ref.columns if c not in meta and pd.api.types.is_numeric_dtype(ref[c])}
    result = [c for c in feature_columns if c in available]
    if not result:
        raise ValueError("No shared feature columns remain after reference-parquet filtering.")
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_cfg = ExperimentConfig().data
    data_cfg.parquet_path = args.parquet_path
    data_cfg.embedding_path = args.embedding_path
    data_cfg.label_name = "craving"
    data_cfg.normalization = "subject_zscore"
    base_df, feature_columns, embeddings = load_base_resources(
        data_cfg, drop_tasks=sorted(EXCLUDED_TASKS)
    )
    if "dataset_name" in base_df:
        base_df = base_df[base_df["dataset_name"].astype(str) == "OUDLab"].copy()
    if "side" in base_df:
        base_df = base_df[base_df["side"].astype(str) == "left"].copy()
    base_df = base_df[base_df["craving"].isin([0, 1])].copy()
    base_df = filter_oudlab_subjects_for_label(base_df, "craving")
    mixed, _, _ = summarize_subject_label_types(base_df, "craving")
    feature_columns = restrict_features(feature_columns, args.reference_parquet)

    folds = [(key, split) for key, split in load_folds(args.splits_path) if str(split["test"][0]) in set(map(str, mixed))]
    keep = selected_subjects(args)
    if keep is not None:
        folds = [(key, split) for key, split in folds if str(split["test"][0]) in keep]
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise RuntimeError("No eligible LOSO folds remain.")

    cfg = fixed_config(args, len(feature_columns), feature_columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions, fold_rows = [], []

    for fold_index, (_, split) in enumerate(folds):
        test_subject = str(split["test"][0])
        train_df = base_df[base_df["participant_id"].astype(str).isin(map(str, split["train"]))].copy()
        val_df = base_df[base_df["participant_id"].astype(str).isin(map(str, split["val"]))].copy()
        test_df = base_df[base_df["participant_id"].astype(str) == test_subject].copy()
        train_ds = make_dataset(train_df, feature_columns, embeddings, cfg.data)
        val_ds = make_dataset(val_df, feature_columns, embeddings, cfg.data)
        test_ds = make_dataset(test_df, feature_columns, embeddings, cfg.data)
        sampler = make_weighted_sampler(train_ds, seed=args.seed + fold_index)
        train_loader = make_loader(train_ds, 30, shuffle=True, num_workers=0, sampler=sampler, seed=args.seed + fold_index)
        val_loader = make_loader(val_ds, 30, shuffle=False, num_workers=0, seed=args.seed + fold_index)
        test_loader = make_loader(test_ds, 30, shuffle=False, num_workers=0, seed=args.seed + fold_index)

        ckpt = checkpoint_for_subject(args.stress_encoder_dir, test_subject)
        validate_checkpoint(ckpt, feature_columns)
        cfg.model.stress_encoder_ckpt = ckpt
        model = build_model(cfg.model).to(device)
        objective = build_objective(cfg.objective)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        trainer = Trainer(model, objective, optimizer, device, grad_clip=cfg.train.grad_clip, scheduler=None)
        fold_dir = out / f"fold_{fold_index:03d}_{test_subject}"
        trainer.fit(train_loader, val_loader, cfg.train.epochs, fold_dir)
        metrics = trainer.evaluate(test_loader)
        pred = trainer.predict(test_loader)
        pred.insert(0, "fold_idx", fold_index)
        pred.insert(1, "test_subject", test_subject)
        all_predictions.append(pred)
        fold_rows.append({"fold_idx": fold_index, "test_subject": test_subject, **metrics, "auc": safe_auc(pred["y_true"], pred["y_prob"])})

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = {
        "n_folds": len(fold_rows),
        "n_predictions": len(predictions),
        "pooled_acc": float(accuracy_score(predictions["y_true"], predictions["y_pred"])),
        "pooled_bacc": float(balanced_accuracy_score(predictions["y_true"], predictions["y_pred"])),
        "pooled_auc": safe_auc(predictions["y_true"], predictions["y_prob"]),
        "pooled_macro_f1": float(f1_score(predictions["y_true"], predictions["y_pred"], average="macro")),
        "config": {
            "epochs": 24, "batch_size": 30, "optimizer": "AdamW", "learning_rate": 0.00131,
            "weight_decay": 1.12e-6, "gradient_clip": 0.424, "dropout": 0.193,
            "normalization": "subject_zscore", "label_smoothing": 0.0175,
            "input_gate_sparsity": 0.00102, "stress_auxiliary_loss": 0.000596,
            "orthogonality_loss": 0.000541, "scheduler": "none", "seed": args.seed,
        },
    }
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    (out / "config.json").write_text(json.dumps(summary["config"], indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
