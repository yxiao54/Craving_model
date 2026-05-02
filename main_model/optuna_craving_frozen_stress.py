#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from core import ExperimentConfig, Trainer, build_model, build_objective
from core.data import load_base_resources, make_dataset, make_loader, make_weighted_sampler
from core.subject_policy import (
    summarize_subject_label_types,
    filter_oudlab_subjects_for_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna search for craving with a frozen stress encoder.")
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--test-subjects-json", type=str, default=None)
    parser.add_argument("--stress-encoder-dir", type=str, required=True)
    parser.add_argument("--reference-parquet", type=str, default=None, help="Optional parquet used to restrict OUDLab features to a shared set, e.g. OUDWildNew flat-mean parquet.")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--objective-choices", type=str, nargs="*", default=["erm", "vrex"])
    parser.add_argument("--sampler", type=str, choices=["weighted", "none"], default="weighted")
    parser.add_argument("--fixed-fusion-mode", type=str, default=None)
    parser.add_argument("--test-subjects", type=str, nargs="*", default=None)
    parser.add_argument("--search-space-mode", type=str, choices=["constrained", "relaxed"], default="constrained")
    parser.add_argument("--fixed-architecture-erm", action="store_true")
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "no_guidance_zero",
            "subject_id_random",
            "no_guidance_global",
            "good_only",
            "bad_only",
            "no_input_gate",
            "no_sample_gate",
            "stress_only",
            "trainable_only",
            "no_gate_sparsity",
            "no_orthogonality",
        ],
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _safe_auc(y_true, y_prob):
    if len(set(map(int, y_true))) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def _compute_head_metrics(y_true, y_pred, y_prob):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_prob),
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
    base_df, feature_cols, embeddings = load_base_resources(cfg.data)
    if "dataset_name" in base_df.columns:
        base_df = base_df[base_df["dataset_name"].astype(str) == "OUDLab"].copy()
    if "side" in base_df.columns:
        base_df = base_df[base_df["side"].astype(str) == "left"].copy()
    base_df = base_df[base_df["craving"].isin([0, 1])].copy()
    base_df = filter_oudlab_subjects_for_label(base_df, "craving")
    mixed_subjects, _, _ = summarize_subject_label_types(base_df, "craving")
    return base_df, feature_cols, embeddings, set(mixed_subjects)


def transform_embeddings(embeddings, mode, seed):
    keys = sorted(embeddings.keys())
    if mode == "none":
        return embeddings

    def subject_random_vec(subject_id, like_array, tag):
        digest = hashlib.sha256(f"{seed}:{subject_id}:{tag}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(local_seed)
        vec = rng.standard_normal(int(np.prod(like_array.shape)), dtype=np.float32).reshape(like_array.shape)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.astype(np.float32)

    out = {}
    if mode == "no_guidance_zero":
        for key in keys:
            out[key] = {
                "good_embedding": np.zeros_like(embeddings[key]["good_embedding"], dtype=np.float32),
                "bad_embedding": np.zeros_like(embeddings[key]["bad_embedding"], dtype=np.float32),
            }
        return out

    if mode == "subject_id_random":
        for key in keys:
            out[key] = {
                "good_embedding": subject_random_vec(key, embeddings[key]["good_embedding"], "good"),
                "bad_embedding": subject_random_vec(key, embeddings[key]["bad_embedding"], "bad"),
            }
        return out

    raise ValueError(f"Unsupported embedding transform mode: {mode}")


def maybe_restrict_feature_columns(feature_cols, reference_parquet):
    if reference_parquet is None:
        return feature_cols
    ref_df = pd.read_parquet(reference_parquet)
    meta_cols = {
        "participant_id",
        "global_subject_id",
        "cohort",
        "block_id",
        "window_start_ns",
        "window_end_ns",
        "window_start_sec",
        "window_end_sec",
        "task",
        "stress",
        "craving",
        "ema_name",
        "ema_index",
        "event_type",
        "ema_start_ns",
        "ema_end_ns",
        "ema_duration_sec",
        "stress_score",
        "craving_score",
        "window_sec",
        "step_sec",
        "n_windows",
        "n_blocks_used",
    }
    ref_features = {
        c for c in ref_df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(ref_df[c])
    }
    out = [c for c in feature_cols if c in ref_features]
    if not out:
        raise RuntimeError("No shared feature columns remain after applying --reference-parquet.")
    return out


def align_features_to_stress_checkpoints(base_df, feature_cols, stress_encoder_dir):
    matches = sorted(Path(stress_encoder_dir).glob("*/best_encoder.pt"))
    if not matches:
        raise FileNotFoundError(f"No best_encoder.pt files found under {stress_encoder_dir}")
    payload = torch.load(matches[0], map_location="cpu")
    ckpt_feature_cols = list(payload.get("feature_columns", []))
    if not ckpt_feature_cols:
        raise ValueError("Stress encoder checkpoints do not contain feature_columns metadata.")

    if "window_duration_sec" in ckpt_feature_cols and "window_duration_sec" not in base_df.columns:
        if {"window_start_ns", "window_end_ns"}.issubset(base_df.columns):
            base_df["window_duration_sec"] = (
                (base_df["window_end_ns"] - base_df["window_start_ns"]).astype(np.float64) / 1e9
            )
        else:
            base_df["window_duration_sec"] = 30.0
    if "EDA_native_sr" in ckpt_feature_cols and "EDA_native_sr" not in base_df.columns:
        base_df["EDA_native_sr"] = 4.0

    missing = [c for c in ckpt_feature_cols if c not in base_df.columns]
    if missing:
        raise ValueError(
            "Craving parquet is missing stress-checkpoint feature columns even after backfill: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return base_df, list(ckpt_feature_cols)


def resolve_stress_encoder_ckpt(stress_encoder_dir, test_subject):
    root = Path(stress_encoder_dir)
    direct = root / str(test_subject) / "best_encoder.pt"
    if direct.exists():
        return str(direct)
    matches = sorted(root.glob(f"fold_*_{test_subject}/best_encoder.pt"))
    if not matches:
        raise FileNotFoundError(
            f"Missing stress encoder checkpoint for subject {test_subject} under {stress_encoder_dir}"
        )
    return str(matches[0])


def validate_stress_encoder_ckpt(path, feature_cols):
    payload = torch.load(path, map_location="cpu")
    ckpt_feature_cols = payload.get("feature_columns")
    ckpt_input_dim = payload.get("input_dim")
    if ckpt_feature_cols is not None:
        if list(ckpt_feature_cols) != list(feature_cols):
            raise ValueError(
                "Stress encoder checkpoint feature columns do not match Optuna feature columns. "
                f"checkpoint_dim={len(ckpt_feature_cols)} optuna_dim={len(feature_cols)}"
            )
    elif ckpt_input_dim is not None:
        if int(ckpt_input_dim) != int(len(feature_cols)):
            raise ValueError(
                "Stress encoder checkpoint input_dim does not match Optuna feature columns. "
                f"checkpoint_dim={ckpt_input_dim} optuna_dim={len(feature_cols)}"
            )
    else:
        raise ValueError(
            "Stress encoder checkpoint has no feature_columns/input_dim metadata. "
            "Re-run pretrain_stress_encoder.py with the updated code."
        )


def suggest_config(trial, args, input_dim):
    cfg = ExperimentConfig()
    cfg.data.parquet_path = args.parquet_path
    cfg.data.embedding_path = args.embedding_path
    cfg.data.label_name = "craving"
    cfg.data.normalization = trial.suggest_categorical("normalization", ["subject_zscore"])
    cfg.data.batch_size = 30

    cfg.model.name = "semantic_dualpath_frozen_stress"
    cfg.model.input_dim = input_dim
    cfg.model.hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256, 320, 384, 512])
    if args.fixed_architecture_erm:
        cfg.model.gate_hidden_dim = 64
    elif args.search_space_mode == "relaxed":
        cfg.model.gate_hidden_dim = trial.suggest_categorical("gate_hidden_dim", [64, 128])
    else:
        cfg.model.gate_hidden_dim = 64
    cfg.model.num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 5)
    cfg.model.num_classifier_layers = trial.suggest_int("num_classifier_layers", 1, 4)
    if args.fixed_architecture_erm:
        cfg.model.dropout = trial.suggest_float("dropout", 0.15, 0.50)
    elif args.search_space_mode == "relaxed":
        cfg.model.dropout = trial.suggest_float("dropout", 0.15, 0.50)
    else:
        cfg.model.dropout = trial.suggest_float("dropout", 0.25, 0.42)
    cfg.model.use_good_embedding = True
    cfg.model.use_bad_embedding = True
    if args.fixed_architecture_erm:
        input_conditioning = "gate"
    elif args.search_space_mode == "relaxed":
        input_conditioning = trial.suggest_categorical("input_conditioning", ["gate", "none", "film"])
    else:
        input_conditioning = "gate"
    cfg.model.use_input_feature_gate = input_conditioning == "gate"
    cfg.model.use_input_film = input_conditioning == "film"
    cfg.model.use_trainable_phys_branch = True
    cfg.model.use_stress_branch = True
    cfg.model.use_logit_scale = False if args.fixed_architecture_erm else True
    if args.fixed_architecture_erm:
        cfg.model.use_affine_calibration = False
        cfg.model.use_semantic_uncertainty = False
        cfg.model.use_stress_aux_head = False
    elif args.search_space_mode == "relaxed":
        cfg.model.use_affine_calibration = trial.suggest_categorical("use_affine_calibration", [False, True])
        cfg.model.use_semantic_uncertainty = trial.suggest_categorical("use_semantic_uncertainty", [False, True])
        cfg.model.use_stress_aux_head = False
    else:
        cfg.model.use_affine_calibration = False
        cfg.model.use_semantic_uncertainty = False
        cfg.model.use_stress_aux_head = False
    if args.fixed_architecture_erm:
        cfg.model.fusion_mode = "sample_gate"
    elif args.fixed_fusion_mode is not None:
        cfg.model.fusion_mode = args.fixed_fusion_mode
    else:
        cfg.model.fusion_mode = trial.suggest_categorical(
            "fusion_mode",
            ["sample_gate", "scalar", "residual", "mixture", "concat_mlp", "attention", "cross_gate", "cross_film"],
        )
    cfg.model.freeze_stress_encoder = True

    if args.fixed_architecture_erm:
        cfg.objective.name = "erm"
        cfg.objective.sparse_gate_weight = trial.suggest_float("sparse_gate_weight", 1e-4, 1e-2, log=True)
        cfg.objective.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        cfg.objective.vrex_lambda = 0.0
        cfg.objective.stress_aux_weight = 0.0
        cfg.objective.orthogonality_weight = trial.suggest_float("orthogonality_weight", 1e-4, 1e-2, log=True)
        cfg.objective.semantic_kl_weight = 0.0
    elif args.search_space_mode == "relaxed":
        cfg.objective.name = trial.suggest_categorical("objective", args.objective_choices)
        cfg.objective.sparse_gate_weight = trial.suggest_float("sparse_gate_weight", 1e-4, 1e-2, log=True)
        cfg.objective.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        cfg.objective.vrex_lambda = trial.suggest_float("vrex_lambda", 1e-2, 2.0, log=True)
        cfg.objective.stress_aux_weight = 0.0
        cfg.objective.orthogonality_weight = trial.suggest_float("orthogonality_weight", 1e-4, 1e-2, log=True)
        cfg.objective.semantic_kl_weight = (
            trial.suggest_float("semantic_kl_weight", 1e-5, 1e-2, log=True) if cfg.model.use_semantic_uncertainty else 0.0
        )
    else:
        cfg.objective.name = "vrex"
        cfg.objective.sparse_gate_weight = trial.suggest_float("sparse_gate_weight", 5e-4, 2e-3, log=True)
        cfg.objective.label_smoothing = trial.suggest_float("label_smoothing", 0.10, 0.13)
        cfg.objective.vrex_lambda = trial.suggest_float("vrex_lambda", 0.1, 0.9, log=True)
        cfg.objective.stress_aux_weight = 0.0
        cfg.objective.orthogonality_weight = trial.suggest_float("orthogonality_weight", 5e-4, 5e-3, log=True)
        cfg.objective.semantic_kl_weight = 0.0

    cfg.train.lr = trial.suggest_float("lr", 1e-4, 8e-3, log=True)
    if args.fixed_architecture_erm:
        cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    elif args.search_space_mode == "relaxed":
        cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    else:
        cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-5, 2e-4, log=True)
    cfg.train.epochs = args.epochs
    cfg.train.grad_clip = trial.suggest_float("grad_clip", 0.1, 2.0, log=True)
    if args.fixed_architecture_erm:
        cfg.train.optimizer_name = "adamw"
    elif args.search_space_mode == "relaxed":
        cfg.train.optimizer_name = trial.suggest_categorical("optimizer_name", ["adamw", "adam"])
    else:
        cfg.train.optimizer_name = "adamw"
    cfg.train.scheduler_name = trial.suggest_categorical("scheduler_name", ["none"])
    cfg.train.scheduler_step_size = 10
    cfg.train.scheduler_gamma = 0.5
    apply_ablation_to_cfg(cfg, args.ablation)
    return cfg


def apply_ablation_to_cfg(cfg, ablation):
    if ablation == "none":
        return
    if ablation == "good_only":
        cfg.model.use_good_embedding = True
        cfg.model.use_bad_embedding = False
    elif ablation == "bad_only":
        cfg.model.use_good_embedding = False
        cfg.model.use_bad_embedding = True
    elif ablation == "no_input_gate":
        cfg.model.use_input_feature_gate = False
        cfg.model.use_input_film = False
    elif ablation == "no_sample_gate":
        cfg.model.fusion_mode = "average"
    elif ablation == "stress_only":
        cfg.model.use_stress_branch = True
        cfg.model.use_trainable_phys_branch = False
        cfg.model.use_input_feature_gate = False
        cfg.model.use_input_film = False
        cfg.model.use_stress_aux_head = False
        cfg.objective.stress_aux_weight = 0.0
    elif ablation == "trainable_only":
        cfg.model.use_stress_branch = False
        cfg.model.use_trainable_phys_branch = True
    elif ablation == "no_gate_sparsity":
        cfg.objective.sparse_gate_weight = 0.0
    elif ablation == "no_orthogonality":
        cfg.objective.orthogonality_weight = 0.0
    elif ablation == "no_guidance_global":
        cfg.model.use_global_guidance = True
    elif ablation in {"no_guidance_zero", "subject_id_random"}:
        pass
    else:
        raise ValueError(f"Unsupported ablation: {ablation}")


def build_optimizer_and_scheduler(model, cfg):
    if cfg.train.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    scheduler = None
    if cfg.train.scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(cfg.train.scheduler_step_size, 1),
            gamma=cfg.train.scheduler_gamma,
        )
    return optimizer, scheduler


def describe_input_conditioning(cfg):
    if getattr(cfg.model, "use_input_feature_gate", False):
        return "gate"
    if getattr(cfg.model, "use_input_film", False):
        return "film"
    return "none"


def evaluate_trial(trial, args, base_df, feature_cols, embeddings, folds):
    if not folds:
        raise RuntimeError("No folds available for evaluation after subject filtering.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = suggest_config(trial, args, input_dim=len(feature_cols))
    input_conditioning = describe_input_conditioning(cfg)
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
        train_loader = make_loader(
            train_dataset,
            cfg.data.batch_size,
            shuffle=True,
            num_workers=0,
            sampler=sampler,
            seed=args.seed + fold_index,
        )
        val_loader = make_loader(val_dataset, cfg.data.batch_size, shuffle=False, num_workers=0, seed=args.seed + fold_index)
        test_loader = make_loader(test_dataset, cfg.data.batch_size, shuffle=False, num_workers=0, seed=args.seed + fold_index)

        cfg.model.stress_encoder_ckpt = resolve_stress_encoder_ckpt(args.stress_encoder_dir, test_subject)
        validate_stress_encoder_ckpt(cfg.model.stress_encoder_ckpt, feature_cols)
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
        fold_rows.append(
            {
                "fold_idx": fold_index,
                "test_subject": str(test_subject),
                "n_predictions": int(len(pred_df)),
                "acc": float(metrics["acc"]),
                "bacc": float(metrics["bacc"]),
                "f1": float(metrics["f1"]),
                "auc": _safe_auc(pred_df["y_true"].tolist(), pred_df["y_prob"].tolist()),
                "loss": float(metrics["loss"]),
            }
        )
        if "stress_y_pred" in pred_df.columns and pred_df["stress_y_pred"].notna().any():
            stress_df = pred_df.dropna(subset=["stress_y_pred", "stress_y_prob"]).copy()
            if not stress_df.empty:
                stress_metrics = _compute_head_metrics(
                    stress_df["stress_y_true"].astype(int).tolist(),
                    stress_df["stress_y_pred"].astype(int).tolist(),
                    stress_df["stress_y_prob"].astype(float).tolist(),
                )
                fold_rows[-1].update(
                    {
                        "stress_acc": stress_metrics["acc"],
                        "stress_bacc": stress_metrics["bacc"],
                        "stress_auc": stress_metrics["auc"],
                        "stress_macro_f1": stress_metrics["macro_f1"],
                        "stress_micro_f1": stress_metrics["micro_f1"],
                    }
                )

        running_score = float(np.mean(fold_scores))
        trial.report(running_score, step=fold_index)

    pred_all = pd.concat(all_preds, ignore_index=True)
    pooled_bacc = float(balanced_accuracy_score(pred_all["y_true"], pred_all["y_pred"]))
    pooled_acc = float(accuracy_score(pred_all["y_true"], pred_all["y_pred"]))
    pooled_auc = _safe_auc(pred_all["y_true"].tolist(), pred_all["y_prob"].tolist())
    pooled_macro_f1 = float(f1_score(pred_all["y_true"], pred_all["y_pred"], average="macro"))
    pooled_micro_f1 = float(f1_score(pred_all["y_true"], pred_all["y_pred"], average="micro"))
    mean_fold_bacc = float(np.mean(fold_scores))
    fold_metrics_df = pd.DataFrame(fold_rows)
    pooled_stress_metrics = None
    if "stress_y_pred" in pred_all.columns and pred_all["stress_y_pred"].notna().any():
        stress_all = pred_all.dropna(subset=["stress_y_pred", "stress_y_prob"]).copy()
        if not stress_all.empty:
            pooled_stress_metrics = _compute_head_metrics(
                stress_all["stress_y_true"].astype(int).tolist(),
                stress_all["stress_y_pred"].astype(int).tolist(),
                stress_all["stress_y_prob"].astype(float).tolist(),
            )

    config_payload = {
        "data": cfg.data.normalization,
        "model": cfg.model.name,
        "hidden_dim": cfg.model.hidden_dim,
        "gate_hidden_dim": cfg.model.gate_hidden_dim,
        "num_encoder_layers": cfg.model.num_encoder_layers,
        "num_classifier_layers": cfg.model.num_classifier_layers,
        "objective": cfg.objective.name,
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "batch_size": cfg.data.batch_size,
        "optimizer_name": cfg.train.optimizer_name,
        "scheduler_name": cfg.train.scheduler_name,
        "grad_clip": cfg.train.grad_clip,
        "label_smoothing": cfg.objective.label_smoothing,
        "sparse_gate_weight": cfg.objective.sparse_gate_weight,
        "orthogonality_weight": cfg.objective.orthogonality_weight,
        "semantic_kl_weight": cfg.objective.semantic_kl_weight,
        "input_conditioning": input_conditioning,
        "use_good_embedding": cfg.model.use_good_embedding,
        "use_bad_embedding": cfg.model.use_bad_embedding,
        "use_global_guidance": cfg.model.use_global_guidance,
        "use_input_feature_gate": cfg.model.use_input_feature_gate,
        "use_input_film": cfg.model.use_input_film,
        "use_trainable_phys_branch": cfg.model.use_trainable_phys_branch,
        "use_stress_branch": cfg.model.use_stress_branch,
        "use_affine_calibration": cfg.model.use_affine_calibration,
        "use_semantic_uncertainty": cfg.model.use_semantic_uncertainty,
        "use_logit_scale": cfg.model.use_logit_scale,
        "fusion_mode": cfg.model.fusion_mode,
        "ablation": args.ablation,
    }
    summary_payload = {
        "trial": int(trial.number),
        "n_folds": int(len(fold_rows)),
        "n_predictions": int(len(pred_all)),
        "pooled_acc": pooled_acc,
        "pooled_bacc": pooled_bacc,
        "pooled_auc": pooled_auc,
        "pooled_micro_f1": pooled_micro_f1,
        "pooled_macro_f1": pooled_macro_f1,
        "mean_fold_bacc": mean_fold_bacc,
        "feature_columns": list(feature_cols),
        "config": config_payload,
    }
    if pooled_stress_metrics is not None:
        summary_payload.update(
            {
                "pooled_stress_acc": pooled_stress_metrics["acc"],
                "pooled_stress_bacc": pooled_stress_metrics["bacc"],
                "pooled_stress_auc": pooled_stress_metrics["auc"],
                "pooled_stress_macro_f1": pooled_stress_metrics["macro_f1"],
                "pooled_stress_micro_f1": pooled_stress_metrics["micro_f1"],
            }
        )

    fold_metrics_df.to_csv(trial_dir / "fold_metrics.csv", index=False)
    pred_all.to_parquet(trial_dir / "predictions.parquet", index=False)
    (trial_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (trial_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    trial.set_user_attr("pooled_acc", pooled_acc)
    trial.set_user_attr("pooled_macro_f1", pooled_macro_f1)
    trial.set_user_attr("pooled_micro_f1", pooled_micro_f1)
    trial.set_user_attr("pooled_auc", pooled_auc)
    trial.set_user_attr("mean_fold_bacc", mean_fold_bacc)
    if pooled_stress_metrics is not None:
        trial.set_user_attr("pooled_stress_acc", pooled_stress_metrics["acc"])
        trial.set_user_attr("pooled_stress_bacc", pooled_stress_metrics["bacc"])
        trial.set_user_attr("pooled_stress_auc", pooled_stress_metrics["auc"])
        trial.set_user_attr("pooled_stress_macro_f1", pooled_stress_metrics["macro_f1"])
        trial.set_user_attr("pooled_stress_micro_f1", pooled_stress_metrics["micro_f1"])
    trial.set_user_attr("config", json.dumps(config_payload))
    return pooled_bacc


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base_df, feature_cols, embeddings, mixed_subjects = prepare_base_dataframe(
        args.parquet_path,
        args.embedding_path,
    )
    if args.ablation in {"no_guidance_zero", "subject_id_random"}:
        embeddings = transform_embeddings(embeddings, args.ablation, args.seed)
    base_df, feature_cols = align_features_to_stress_checkpoints(base_df, feature_cols, args.stress_encoder_dir)
    feature_cols = maybe_restrict_feature_columns(feature_cols, args.reference_parquet)
    folds = load_fold_splits(args.splits_path, None)
    folds = [(k, v) for k, v in folds if v["test"][0] in mixed_subjects]
    keep_subjects = load_selected_subjects(args.test_subjects, args.test_subjects_json)
    if keep_subjects is not None:
        folds = [(k, v) for k, v in folds if str(v["test"][0]) in keep_subjects]
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise RuntimeError("No folds remain after applying the current fold filters.")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    study.optimize(
        lambda trial: evaluate_trial(trial, args, base_df, feature_cols, embeddings, folds),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

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
