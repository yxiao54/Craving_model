#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from core import ExperimentConfig, Trainer, build_model, build_objective
from core.data import load_base_resources, make_dataset, make_loader, make_weighted_sampler
from core.subject_policy import EXCLUDED_TASKS, filter_oudlab_subjects_for_label, summarize_subject_label_types


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna baselines for OUDLab craving detection.")
    parser.add_argument("--baseline", type=str, required=True, choices=[
        "xgboost", "rbf_svm", "random_forest", "mlp", "concat", "cross_attention", "film_input", "moe", "irm", "vrex", "groupdro",
    ])
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--test-subjects-json", type=str, default=None)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sampler", type=str, choices=["weighted", "none"], default="weighted")
    parser.add_argument("--test-subjects", type=str, nargs="*", default=None)
    return parser.parse_args()


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
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    mask = np.isfinite(y_prob_arr)
    if mask.sum() < 2:
        return 0.0
    y_true_arr = y_true_arr[mask]
    y_prob_arr = y_prob_arr[mask]
    if len(set(map(int, y_true_arr.tolist()))) < 2:
        return 0.0
    return float(roc_auc_score(y_true_arr, y_prob_arr))


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }


def load_fold_splits(path, max_folds=None):
    with Path(path).open() as handle:
        payload = json.load(handle)
    folds = payload["folds"]
    ordered = sorted(folds.items(), key=lambda kv: int(kv[0]))
    if max_folds is not None:
        ordered = ordered[:max_folds]
    return ordered


def load_selected_subjects(test_subjects, test_subjects_json):
    if test_subjects:
        return {str(s) for s in test_subjects}
    if test_subjects_json:
        payload = json.loads(Path(test_subjects_json).read_text())
        subjects = payload.get("subjects", [])
        if not subjects:
            raise ValueError(f"No subjects found in test-subjects JSON: {test_subjects_json}")
        return {str(s) for s in subjects}
    return None


def prepare_base_dataframe(parquet_path, embedding_path):
    cfg = ExperimentConfig()
    cfg.data.parquet_path = parquet_path
    cfg.data.embedding_path = embedding_path
    cfg.data.label_name = "craving"
    base_df, feature_cols, embeddings = load_base_resources(cfg.data, drop_tasks=sorted(EXCLUDED_TASKS))
    if "dataset_name" in base_df.columns:
        base_df = base_df[base_df["dataset_name"].astype(str) == "OUDLab"].copy()
    if "side" in base_df.columns:
        base_df = base_df[base_df["side"].astype(str) == "left"].copy()
    base_df = base_df[base_df["craving"].isin([0, 1])].copy()
    base_df = filter_oudlab_subjects_for_label(base_df, "craving")
    mixed_subjects, _, _ = summarize_subject_label_types(base_df, "craving")
    return base_df, feature_cols, embeddings, set(mixed_subjects)


def build_numpy_split(df, feature_cols, embeddings, normalization):
    cfg = ExperimentConfig().data
    cfg.label_name = "craving"
    cfg.environment_key = "participant_id"
    cfg.normalization = normalization
    dataset = make_dataset(df, feature_cols, embeddings, cfg)
    return dataset.features, dataset.labels, dataset.participant_ids, dataset.tasks, dataset.environments


def suggest_torch_config(trial, args, input_dim):
    cfg = ExperimentConfig()
    cfg.data.parquet_path = args.parquet_path
    cfg.data.embedding_path = args.embedding_path
    cfg.data.label_name = "craving"
    cfg.data.normalization = "subject_zscore"
    cfg.data.batch_size = 30
    cfg.data.environment_key = "participant_id"

    if args.baseline in {"mlp", "irm", "vrex", "groupdro"}:
        cfg.model.name = "stress_mlp"
        cfg.model.use_good_embedding = False
        cfg.model.use_bad_embedding = False
    elif args.baseline == "concat":
        cfg.model.name = "semantic_concat"
        cfg.model.use_good_embedding = True
        cfg.model.use_bad_embedding = True
    elif args.baseline == "cross_attention":
        cfg.model.name = "semantic_token"
        cfg.model.use_good_embedding = True
        cfg.model.use_bad_embedding = True
    elif args.baseline == "film_input":
        cfg.model.name = "semantic_input_film"
        cfg.model.use_good_embedding = True
        cfg.model.use_bad_embedding = True
    elif args.baseline == "moe":
        cfg.model.name = "semantic_moe"
        cfg.model.use_good_embedding = True
        cfg.model.use_bad_embedding = True
    else:
        raise ValueError(f"Unsupported torch baseline: {args.baseline}")

    cfg.model.input_dim = input_dim
    cfg.model.hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256, 320, 384, 512])
    cfg.model.gate_hidden_dim = 64
    cfg.model.num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 5)
    cfg.model.num_classifier_layers = trial.suggest_int("num_classifier_layers", 1, 4)
    cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.4)
    cfg.model.use_logit_scale = False
    cfg.model.use_film = False
    cfg.model.use_sparse_gate = False
    cfg.model.use_input_feature_gate = False
    cfg.model.use_input_film = False
    cfg.model.use_trainable_phys_branch = True
    cfg.model.use_stress_branch = False
    cfg.model.use_affine_calibration = False
    cfg.model.use_semantic_uncertainty = False
    cfg.model.use_stress_aux_head = False
    if args.baseline == "moe":
        cfg.model.num_experts = trial.suggest_categorical("num_experts", [2, 4, 6])
        cfg.model.top_k_experts = trial.suggest_categorical("top_k_experts", [0, 1, 2])
        if cfg.model.top_k_experts > 0 and cfg.model.top_k_experts > cfg.model.num_experts:
            cfg.model.top_k_experts = cfg.model.num_experts

    if args.baseline == "irm":
        cfg.objective.name = "irm"
        cfg.objective.irm_lambda = trial.suggest_float("irm_lambda", 0.1, 0.4)
    elif args.baseline == "vrex":
        cfg.objective.name = "vrex"
        cfg.objective.vrex_lambda = trial.suggest_float("vrex_lambda", 0.1, 0.4)
    elif args.baseline == "groupdro":
        cfg.objective.name = "groupdro"
        cfg.objective.dro_eta = trial.suggest_float("dro_eta", 0.1, 0.4)
    else:
        cfg.objective.name = "erm"

    cfg.objective.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.12)
    cfg.objective.sparse_gate_weight = 0.0
    cfg.objective.stress_aux_weight = 0.0
    cfg.objective.orthogonality_weight = 0.0
    cfg.objective.semantic_kl_weight = 0.0
    cfg.objective.moe_load_balance_weight = (
        trial.suggest_float("moe_load_balance_weight", 1e-4, 5e-2, log=True)
        if args.baseline == "moe" else 0.0
    )

    cfg.train.lr = trial.suggest_float("lr", 1e-4, 8e-3, log=True)
    cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 2e-4, log=True)
    cfg.train.epochs = args.epochs
    cfg.train.grad_clip = trial.suggest_float("grad_clip", 0.1, 2.0, log=True)
    cfg.train.optimizer_name = "adamw"
    cfg.train.scheduler_name = "none"
    return cfg


def build_optimizer_and_scheduler(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return optimizer, None


def sklearn_model_from_trial(trial, baseline, train_y):
    if baseline == "rbf_svm":
        c_val = trial.suggest_float("C", 1e-2, 50.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 2.0, log=True)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=c_val, gamma=gamma, kernel="rbf", probability=True, class_weight="balanced", random_state=2026)),
        ])
        return model

    if baseline == "xgboost":
        pos = int(np.sum(train_y == 1))
        neg = int(np.sum(train_y == 0))
        scale_pos_weight = float(neg / max(pos, 1))
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=2026,
            n_jobs=1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )

    if baseline == "random_forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 16),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                class_weight="balanced",
                random_state=2026,
                n_jobs=1,
            )),
        ])

    raise ValueError(f"Unsupported sklearn baseline: {baseline}")


def trial_config_payload(args, trial, extra=None):
    embedding_name = Path(args.embedding_path).name
    if "good_bad_gap10" in embedding_name:
        embedding_variant = "good_bad_gap10"
    elif "user_embeddings" in embedding_name:
        embedding_variant = "semantic"
    else:
        embedding_variant = embedding_name
    payload = {
        "baseline": args.baseline,
        "embedding_path": args.embedding_path,
        "embedding_variant": embedding_variant,
    }
    payload.update(trial.params)
    if extra:
        payload.update(extra)
    return payload


def evaluate_sklearn_trial(trial, args, base_df, feature_cols, embeddings, folds):
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    fold_rows = []
    fold_scores = []

    for fold_index, (_, split) in enumerate(folds):
        train_subjects = split["train"]
        test_subject = split["test"][0]

        train_df = base_df[base_df["participant_id"].astype(str).isin(train_subjects)].copy()
        test_df = base_df[base_df["participant_id"].astype(str) == test_subject].copy()

        x_train, y_train, _, _, _ = build_numpy_split(train_df, feature_cols, embeddings, "subject_zscore")
        x_test, y_test, pids, tasks, envs = build_numpy_split(test_df, feature_cols, embeddings, "subject_zscore")

        model = sklearn_model_from_trial(trial, args.baseline, y_train)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)

        if args.baseline == "rbf_svm":
            decision = model.named_steps["svc"].decision_function(model.named_steps["scaler"].transform(x_test))
        elif args.baseline == "random_forest":
            decision = model.named_steps["rf"].predict_proba(model.named_steps["scaler"].transform(x_test))[:, 1]
        else:
            decision = model.predict(x_test, output_margin=True)

        row_metrics = compute_metrics(y_test.tolist(), pred.tolist(), prob.tolist())
        fold_scores.append(row_metrics["bacc"])
        fold_rows.append({
            "fold_idx": fold_index,
            "test_subject": str(test_subject),
            "n_predictions": int(len(y_test)),
            "acc": row_metrics["acc"],
            "bacc": row_metrics["bacc"],
            "f1": row_metrics["macro_f1"],
            "auc": row_metrics["auc"],
            "loss": float("nan"),
        })

        pred_df = pd.DataFrame({
            "fold_idx": fold_index,
            "test_subject": str(test_subject),
            "participant_id": pids,
            "task": tasks,
            "environment": envs,
            "y_true": y_test.astype(int),
            "y_pred": pred.astype(int),
            "y_prob": prob.astype(float),
            "logit_0": (-np.asarray(decision)).astype(float),
            "logit_1": np.asarray(decision).astype(float),
        })
        all_preds.append(pred_df)
        trial.report(float(np.mean(fold_scores)), step=fold_index)

    pred_all = pd.concat(all_preds, ignore_index=True)
    pooled = compute_metrics(pred_all["y_true"].tolist(), pred_all["y_pred"].tolist(), pred_all["y_prob"].tolist())
    mean_fold_bacc = float(np.mean(fold_scores))
    config_payload = trial_config_payload(args, trial)
    summary_payload = {
        "trial": int(trial.number),
        "n_folds": int(len(fold_rows)),
        "n_predictions": int(len(pred_all)),
        "pooled_acc": pooled["acc"],
        "pooled_bacc": pooled["bacc"],
        "pooled_auc": pooled["auc"],
        "pooled_micro_f1": pooled["micro_f1"],
        "pooled_macro_f1": pooled["macro_f1"],
        "mean_fold_bacc": mean_fold_bacc,
        "feature_columns": list(feature_cols),
        "config": config_payload,
    }

    pd.DataFrame(fold_rows).to_csv(trial_dir / "fold_metrics.csv", index=False)
    pred_all.to_parquet(trial_dir / "predictions.parquet", index=False)
    (trial_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (trial_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    trial.set_user_attr("pooled_acc", pooled["acc"])
    trial.set_user_attr("pooled_macro_f1", pooled["macro_f1"])
    trial.set_user_attr("pooled_micro_f1", pooled["micro_f1"])
    trial.set_user_attr("pooled_auc", pooled["auc"])
    trial.set_user_attr("mean_fold_bacc", mean_fold_bacc)
    trial.set_user_attr("config", json.dumps(config_payload))
    return pooled["bacc"]


def evaluate_torch_trial(trial, args, base_df, feature_cols, embeddings, folds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = suggest_torch_config(trial, args, input_dim=len(feature_cols))
    cfg.model.feature_columns = feature_cols
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    fold_scores = []
    fold_rows = []
    for fold_index, (_, split) in enumerate(folds):
        train_subjects = split["train"]
        val_subjects = split["val"]
        test_subject = split["test"][0]

        train_df = base_df[base_df["participant_id"].astype(str).isin(train_subjects)].copy()
        val_df = base_df[base_df["participant_id"].astype(str).isin(val_subjects)].copy()
        test_df = base_df[base_df["participant_id"].astype(str) == test_subject].copy()

        train_dataset = make_dataset(train_df, feature_cols, embeddings, cfg.data)
        val_dataset = make_dataset(val_df, feature_cols, embeddings, cfg.data)
        test_dataset = make_dataset(test_df, feature_cols, embeddings, cfg.data)

        sampler = make_weighted_sampler(train_dataset, seed=args.seed + fold_index) if args.sampler == "weighted" else None
        train_loader = make_loader(train_dataset, cfg.data.batch_size, shuffle=True, num_workers=0, sampler=sampler, seed=args.seed + fold_index)
        val_loader = make_loader(val_dataset, cfg.data.batch_size, shuffle=False, num_workers=0, seed=args.seed + fold_index)
        test_loader = make_loader(test_dataset, cfg.data.batch_size, shuffle=False, num_workers=0, seed=args.seed + fold_index)

        model = build_model(cfg.model).to(device)
        objective = build_objective(cfg.objective)
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
        trainer = Trainer(model, objective, optimizer, device, grad_clip=cfg.train.grad_clip, scheduler=scheduler)

        trial_fold_dir = trial_dir / f"fold_{fold_index:03d}_{test_subject}"
        trainer.fit(train_loader, val_loader, cfg.train.epochs, trial_fold_dir)
        metrics = trainer.evaluate(test_loader)
        fold_scores.append(metrics["bacc"])
        pred_df = trainer.predict(test_loader)
        pred_df.insert(0, "fold_idx", fold_index)
        pred_df.insert(1, "test_subject", str(test_subject))
        all_preds.append(pred_df)
        fold_rows.append({
            "fold_idx": fold_index,
            "test_subject": str(test_subject),
            "n_predictions": int(len(pred_df)),
            "acc": float(metrics["acc"]),
            "bacc": float(metrics["bacc"]),
            "f1": float(metrics["f1"]),
            "auc": safe_auc(pred_df["y_true"].tolist(), pred_df["y_prob"].tolist()),
            "loss": float(metrics["loss"]),
        })
        trial.report(float(np.mean(fold_scores)), step=fold_index)

    pred_all = pd.concat(all_preds, ignore_index=True)
    pooled = compute_metrics(pred_all["y_true"].tolist(), pred_all["y_pred"].tolist(), pred_all["y_prob"].tolist())
    mean_fold_bacc = float(np.mean(fold_scores))
    config_payload = trial_config_payload(args, trial, extra={
        "model": cfg.model.name,
        "hidden_dim": cfg.model.hidden_dim,
        "num_encoder_layers": cfg.model.num_encoder_layers,
        "num_classifier_layers": cfg.model.num_classifier_layers,
        "dropout": cfg.model.dropout,
        "lr": cfg.train.lr,
        "optimizer_name": cfg.train.optimizer_name,
        "objective": cfg.objective.name,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.data.batch_size,
        "num_experts": getattr(cfg.model, "num_experts", None),
        "top_k_experts": getattr(cfg.model, "top_k_experts", None),
        "moe_load_balance_weight": cfg.objective.moe_load_balance_weight,
    })
    summary_payload = {
        "trial": int(trial.number),
        "n_folds": int(len(fold_rows)),
        "n_predictions": int(len(pred_all)),
        "pooled_acc": pooled["acc"],
        "pooled_bacc": pooled["bacc"],
        "pooled_auc": pooled["auc"],
        "pooled_micro_f1": pooled["micro_f1"],
        "pooled_macro_f1": pooled["macro_f1"],
        "mean_fold_bacc": mean_fold_bacc,
        "feature_columns": list(feature_cols),
        "config": config_payload,
    }

    pd.DataFrame(fold_rows).to_csv(trial_dir / "fold_metrics.csv", index=False)
    pred_all.to_parquet(trial_dir / "predictions.parquet", index=False)
    (trial_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (trial_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    trial.set_user_attr("pooled_acc", pooled["acc"])
    trial.set_user_attr("pooled_macro_f1", pooled["macro_f1"])
    trial.set_user_attr("pooled_micro_f1", pooled["micro_f1"])
    trial.set_user_attr("pooled_auc", pooled["auc"])
    trial.set_user_attr("mean_fold_bacc", mean_fold_bacc)
    trial.set_user_attr("config", json.dumps(config_payload))
    return pooled["bacc"]


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base_df, feature_cols, embeddings, mixed_subjects = prepare_base_dataframe(args.parquet_path, args.embedding_path)
    folds = load_fold_splits(args.splits_path, None)
    folds = [(k, v) for k, v in folds if v["test"][0] in mixed_subjects]
    if not folds:
        raise RuntimeError("No eligible folds remain after filtering to mixed-label test subjects.")
    keep_subjects = load_selected_subjects(args.test_subjects, args.test_subjects_json)
    if keep_subjects is not None:
        folds = [(k, v) for k, v in folds if str(v["test"][0]) in keep_subjects]
        if not folds:
            raise RuntimeError("No folds remain after applying fixed test-subject selection.")
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if args.baseline in {"xgboost", "rbf_svm", "random_forest"}:
        objective_fn = lambda trial: evaluate_sklearn_trial(trial, args, base_df, feature_cols, embeddings, folds)
    else:
        objective_fn = lambda trial: evaluate_torch_trial(trial, args, base_df, feature_cols, embeddings, folds)

    study.optimize(objective_fn, n_trials=args.n_trials, timeout=args.timeout)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
    }
    best_path = Path(args.output_dir) / "optuna_best.json"
    best_path.write_text(json.dumps(best, indent=2))
    print(f"saved best summary to {best_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
