#!/usr/bin/env python3
"""Train a baseline for OUDLab craving detection.

One invocation trains one baseline over the supplied LOSO folds and writes
fold metrics, pooled predictions, and the resolved configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # Keep non-XGBoost baselines usable without the optional package.
    XGBClassifier = None

from core import ExperimentConfig, Trainer, build_model, build_objective
from core.data import load_base_resources, make_dataset, make_loader, make_weighted_sampler
from core.subject_policy import EXCLUDED_TASKS, filter_oudlab_subjects_for_label, summarize_subject_label_types


BASELINES = ("mlp", "irm", "vrex", "groupdro", "concat", "cross_attention", "film_input", "moe", "rbf_svm", "xgboost", "random_forest")


def parse_args():
    p = argparse.ArgumentParser(description="Train one OUDLab craving baseline.")
    p.add_argument("--baseline", required=True, choices=BASELINES)
    p.add_argument("--parquet-path", required=True)
    p.add_argument("--embedding-path", required=True)
    p.add_argument("--splits-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-subjects-json")
    p.add_argument("--test-subjects", nargs="*")
    p.add_argument("--max-folds", type=int)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sampler", choices=("weighted", "none"), default="weighted")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def safe_auc(y_true, y_prob):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return 0.0
    return float(roc_auc_score(y, p))


def metrics(y_true, y_pred, y_prob):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": safe_auc(y_true, y_prob),
    }


def load_folds(path, max_folds=None):
    payload = json.loads(Path(path).read_text())
    rows = sorted(payload["folds"].items(), key=lambda x: int(x[0]))
    return rows if max_folds is None else rows[:max_folds]


def choose_subjects(args):
    if args.test_subjects:
        return {str(x) for x in args.test_subjects}
    if args.test_subjects_json:
        return {str(x) for x in json.loads(Path(args.test_subjects_json).read_text())["subjects"]}
    return None


def torch_config(args, input_dim, feature_columns):
    cfg = ExperimentConfig()
    cfg.data.label_name = "craving"
    cfg.data.normalization = "subject_zscore"
    cfg.data.batch_size = args.batch_size
    cfg.data.environment_key = "participant_id"
    cfg.model.input_dim = input_dim
    cfg.model.feature_columns = list(feature_columns)
    cfg.model.hidden_dim = 384
    cfg.model.gate_hidden_dim = 64
    cfg.model.num_encoder_layers = 2
    cfg.model.num_classifier_layers = 3
    cfg.model.dropout = 0.193
    cfg.model.use_good_embedding = args.baseline not in {"mlp", "irm", "vrex", "groupdro"}
    cfg.model.use_bad_embedding = cfg.model.use_good_embedding
    cfg.model.use_logit_scale = False
    cfg.model.use_sparse_gate = False
    cfg.model.use_input_feature_gate = False
    cfg.model.use_input_film = False
    cfg.model.use_trainable_phys_branch = True
    cfg.model.use_stress_branch = False
    cfg.model.use_stress_aux_head = False
    cfg.model.freeze_stress_encoder = False

    names = {
        "mlp": "stress_mlp", "irm": "stress_mlp", "vrex": "stress_mlp", "groupdro": "stress_mlp",
        "concat": "semantic_concat", "cross_attention": "semantic_token",
        "film_input": "semantic_input_film", "moe": "semantic_moe",
    }
    cfg.model.name = names[args.baseline]
    if args.baseline == "moe":
        cfg.model.num_experts = 4
        cfg.model.top_k_experts = 0
    cfg.objective.name = {"irm": "irm", "vrex": "vrex", "groupdro": "groupdro"}.get(args.baseline, "erm")
    cfg.objective.irm_lambda = 0.2
    cfg.objective.vrex_lambda = 0.2
    cfg.objective.dro_eta = 0.2
    cfg.objective.label_smoothing = 0.0175
    cfg.objective.moe_load_balance_weight = 1e-2 if args.baseline == "moe" else 0.0
    cfg.train.epochs = args.epochs
    cfg.train.lr = 0.00131
    cfg.train.weight_decay = 1.12e-6
    cfg.train.grad_clip = 0.424
    cfg.train.optimizer_name = "adamw"
    cfg.train.scheduler_name = "none"
    cfg.train.seed = args.seed
    return cfg


def prepare_data(args):
    data_cfg = ExperimentConfig().data
    data_cfg.parquet_path = args.parquet_path
    data_cfg.embedding_path = args.embedding_path
    data_cfg.label_name = "craving"
    data_cfg.normalization = "subject_zscore"
    data_cfg.environment_key = "participant_id"
    df, features, embeddings = load_base_resources(data_cfg, drop_tasks=sorted(EXCLUDED_TASKS))
    if "dataset_name" in df:
        df = df[df["dataset_name"].astype(str) == "OUDLab"].copy()
    if "side" in df:
        df = df[df["side"].astype(str) == "left"].copy()
    df = df[df["craving"].isin([0, 1])].copy()
    df = filter_oudlab_subjects_for_label(df, "craving")
    mixed, _, _ = summarize_subject_label_types(df, "craving")
    return df, features, embeddings, set(map(str, mixed))


def run_torch(args, df, features, embeddings, folds, output):
    cfg = torch_config(args, len(features), features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_pred, fold_rows = [], []
    for fold_idx, (_, split) in enumerate(folds):
        test_subject = str(split["test"][0])
        train = df[df.participant_id.astype(str).isin(map(str, split["train"]))].copy()
        val = df[df.participant_id.astype(str).isin(map(str, split["val"]))].copy()
        test = df[df.participant_id.astype(str) == test_subject].copy()
        train_ds = make_dataset(train, features, embeddings, cfg.data)
        val_ds = make_dataset(val, features, embeddings, cfg.data)
        test_ds = make_dataset(test, features, embeddings, cfg.data)
        sampler = make_weighted_sampler(train_ds, seed=args.seed + fold_idx) if args.sampler == "weighted" else None
        train_loader = make_loader(train_ds, args.batch_size, True, 0, sampler=sampler, seed=args.seed + fold_idx)
        val_loader = make_loader(val_ds, args.batch_size, False, 0, seed=args.seed + fold_idx)
        test_loader = make_loader(test_ds, args.batch_size, False, 0, seed=args.seed + fold_idx)
        model = build_model(cfg.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        trainer = Trainer(model, build_objective(cfg.objective), optimizer, device, grad_clip=cfg.train.grad_clip, scheduler=None)
        fold_dir = output / f"fold_{fold_idx:03d}_{test_subject}"
        trainer.fit(train_loader, val_loader, cfg.train.epochs, fold_dir)
        pred = trainer.predict(test_loader)
        pred.insert(0, "fold_idx", fold_idx)
        pred.insert(1, "test_subject", test_subject)
        all_pred.append(pred)
        m = metrics(pred.y_true, pred.y_pred, pred.y_prob)
        fold_rows.append({"fold_idx": fold_idx, "test_subject": test_subject, "n_predictions": len(pred), **m})
    return pd.concat(all_pred, ignore_index=True), pd.DataFrame(fold_rows), cfg


def run_sklearn(args, df, features, embeddings, folds, output):
    all_pred, fold_rows = [], []
    for fold_idx, (_, split) in enumerate(folds):
        train = df[df.participant_id.astype(str).isin(map(str, split["train"]))].copy()
        test = df[df.participant_id.astype(str) == str(split["test"][0])].copy()
        data_cfg = ExperimentConfig().data
        data_cfg.label_name = "craving"
        data_cfg.normalization = "subject_zscore"
        data_cfg.environment_key = "participant_id"
        train_ds = make_dataset(train, features, embeddings, data_cfg)
        test_ds = make_dataset(test, features, embeddings, data_cfg)
        if args.baseline == "rbf_svm":
            model = Pipeline([("scale", StandardScaler()), ("model", SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, class_weight="balanced", random_state=args.seed))])
        elif args.baseline == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is required for --baseline xgboost.")
            pos = max(int(np.sum(train_ds.labels == 1)), 1)
            neg = max(int(np.sum(train_ds.labels == 0)), 1)
            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=1.0, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", scale_pos_weight=neg / pos, random_state=args.seed, n_jobs=1, tree_method="hist")
        else:
            model = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=args.seed, n_jobs=1))])
        model.fit(train_ds.features, train_ds.labels)
        prob = model.predict_proba(test_ds.features)[:, 1]
        pred = model.predict(test_ds.features)
        out = pd.DataFrame({"fold_idx": fold_idx, "test_subject": str(split["test"][0]), "participant_id": test_ds.participant_ids, "task": test_ds.tasks, "y_true": test_ds.labels, "y_pred": pred, "y_prob": prob})
        all_pred.append(out)
        fold_rows.append({"fold_idx": fold_idx, "test_subject": str(split["test"][0]), "n_predictions": len(out), **metrics(out.y_true, out.y_pred, out.y_prob)})
    return pd.concat(all_pred, ignore_index=True), pd.DataFrame(fold_rows), None


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df, features, embeddings, eligible = prepare_data(args)
    folds = [(k, v) for k, v in load_folds(args.splits_path, args.max_folds) if str(v["test"][0]) in eligible]
    selected = choose_subjects(args)
    if selected is not None:
        folds = [(k, v) for k, v in folds if str(v["test"][0]) in selected]
    if not folds:
        raise RuntimeError("No eligible folds remain.")
    if args.baseline in {"rbf_svm", "random_forest"}:
        pred, fold_metrics, cfg = run_sklearn(args, df, features, embeddings, folds, output)
    else:
        pred, fold_metrics, cfg = run_torch(args, df, features, embeddings, folds, output)
    pooled = metrics(pred.y_true, pred.y_pred, pred.y_prob)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    pred.to_parquet(output / "predictions.parquet", index=False)
    config = {"baseline": args.baseline, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "normalization": "subject_zscore", "feature_count": len(features)}
    if cfg is not None:
        config.update({"model": cfg.model.name, "hidden_dim": cfg.model.hidden_dim, "learning_rate": cfg.train.lr, "weight_decay": cfg.train.weight_decay, "objective": cfg.objective.name})
    (output / "config.json").write_text(json.dumps(config, indent=2))
    summary = {"baseline": args.baseline, "n_folds": len(folds), "n_predictions": len(pred), **{f"pooled_{k}": v for k, v in pooled.items()}, "config": config}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
