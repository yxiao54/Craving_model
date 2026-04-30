#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from core import ExperimentConfig, Trainer, build_model, build_objective
from core.data import load_base_resources, make_dataset, make_loader, make_weighted_sampler


DEFAULT_DROP_TASKS = ["good", "bad", "stress"]
DEFAULT_EXTRA_EVAL_SUBJECTS = ["8876", "8898", "9933", "9941", "9973"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a stress encoder on each craving fold and save encoder checkpoints."
    )
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--control-parquet-path", type=str, default=None)
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--craving-splits-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--normalization", type=str, choices=["none", "subject_zscore", "subject_robust"], default="subject_zscore")
    parser.add_argument("--environment-key", type=str, default="participant_id")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-classifier-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--objective", type=str, choices=["erm", "irm", "vrex", "groupdro"], default="erm")
    parser.add_argument("--train-sampler", type=str, choices=["none", "weighted"], default="weighted")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--drop-tasks", type=str, nargs="*", default=DEFAULT_DROP_TASKS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--extra-eval-subjects",
        type=str,
        nargs="*",
        default=DEFAULT_EXTRA_EVAL_SUBJECTS,
        help="Extra subject IDs for which to create held-out stress checkpoints in addition to craving split folds.",
    )
    parser.add_argument("--reference-parquet", type=str, default=None, help="Optional parquet whose numeric feature columns will be intersected with the stress training feature set.")
    return parser.parse_args()


def load_craving_splits(path, max_folds=None):
    with Path(path).open() as handle:
        payload = json.load(handle)
    folds = payload["folds"]
    ordered = sorted(folds.items(), key=lambda kv: int(kv[0]))
    if max_folds is not None:
        ordered = ordered[:max_folds]
    return ordered


def save_encoder_checkpoint(model, path, feature_columns):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("Model does not expose an encoder attribute.")
    payload = {
        "encoder_state_dict": encoder.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "feature_columns": list(feature_columns),
        "input_dim": int(len(feature_columns)),
    }
    torch.save(payload, path)


def build_fold_specs(craving_splits, extra_eval_subjects, available_subjects):
    fold_specs = []
    seen_test_subjects = set()

    for fold_idx, (_, split) in enumerate(craving_splits):
        test_subject = str(split["test"][0])
        val_subjects = [str(s) for s in split["val"]]
        fold_specs.append(
            {
                "fold_idx": fold_idx,
                "test_subject": test_subject,
                "val_subjects": val_subjects,
                "source": "craving_split",
            }
        )
        seen_test_subjects.add(test_subject)

    next_fold_idx = len(fold_specs)
    for subject in [str(s) for s in extra_eval_subjects]:
        if subject in seen_test_subjects or subject not in available_subjects:
            continue
        fold_specs.append(
            {
                "fold_idx": next_fold_idx,
                "test_subject": subject,
                "val_subjects": [subject],
                "source": "extra_eval_subject",
            }
        )
        seen_test_subjects.add(subject)
        next_fold_idx += 1

    return fold_specs


def make_dummy_embeddings(subject_ids, dim):
    zero = np.zeros(int(dim), dtype=np.float32)
    return {
        str(subject_id): {
            "good_embedding": zero.copy(),
            "bad_embedding": zero.copy(),
        }
        for subject_id in sorted(set(map(str, subject_ids)))
    }


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.data.parquet_path = args.parquet_path
    cfg.data.embedding_path = args.embedding_path
    cfg.data.label_name = "stress"
    cfg.data.normalization = args.normalization
    cfg.data.batch_size = args.batch_size
    cfg.data.environment_key = args.environment_key

    cfg.model.name = "stress_mlp"
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_encoder_layers = args.num_encoder_layers
    cfg.model.num_classifier_layers = args.num_classifier_layers
    cfg.model.dropout = args.dropout

    cfg.objective.name = args.objective
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.output_dir = args.output_dir

    oud_df = pd.read_parquet(args.parquet_path)
    if args.control_parquet_path:
        control_df = pd.read_parquet(args.control_parquet_path)
        base_df = pd.concat([oud_df, control_df], ignore_index=True)
    else:
        base_df = oud_df
    base_df["participant_id"] = base_df["participant_id"].astype(str)
    if args.drop_tasks:
        base_df = base_df[~base_df["task"].astype(str).isin(set(map(str, args.drop_tasks)))].copy()
    base_df = base_df[base_df["stress"].isin([0, 1])].copy()

    from core.data import _read_embeddings, _select_feature_columns  # local import to avoid changing shared data module API

    feature_cols = _select_feature_columns(base_df, cfg.data)
    embeddings = {}
    emb_dim = 1
    if args.embedding_path and Path(args.embedding_path).exists():
        embeddings = _read_embeddings(args.embedding_path)
        if embeddings:
            first = next(iter(embeddings.values()))
            emb_dim = int(np.asarray(first["good_embedding"]).shape[0])
    missing_subjects = sorted(set(base_df["participant_id"].astype(str)) - set(embeddings))
    if missing_subjects:
        embeddings.update(make_dummy_embeddings(missing_subjects, emb_dim))

    if args.reference_parquet:
        ref_df = pd.read_parquet(args.reference_parquet)
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
        feature_cols = [c for c in feature_cols if c in ref_features]
        if not feature_cols:
            raise RuntimeError("No shared feature columns remain after intersecting with reference parquet.")

    cfg.model.input_dim = len(feature_cols)
    cfg.model.feature_columns = feature_cols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    fold_metrics = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available_subjects = set(base_df["participant_id"].astype(str).unique().tolist())
    craving_splits = load_craving_splits(args.craving_splits_path, args.max_folds)
    fold_specs = build_fold_specs(craving_splits, args.extra_eval_subjects, available_subjects)

    for spec in fold_specs:
        fold_idx = spec["fold_idx"]
        val_subjects = spec["val_subjects"]
        test_subject = spec["test_subject"]
        eval_subjects = set(val_subjects + [test_subject])

        train_df = base_df[~base_df["participant_id"].astype(str).isin(eval_subjects)].copy()
        val_df = base_df[base_df["participant_id"].astype(str).isin(val_subjects)].copy()
        test_df = base_df[base_df["participant_id"].astype(str) == test_subject].copy()

        train_dataset = make_dataset(train_df, feature_cols, embeddings, cfg.data)
        val_dataset = make_dataset(val_df, feature_cols, embeddings, cfg.data)
        test_dataset = make_dataset(test_df, feature_cols, embeddings, cfg.data)

        train_sampler = make_weighted_sampler(train_dataset, seed=args.seed + fold_idx) if args.train_sampler == "weighted" else None
        train_loader = make_loader(
            train_dataset,
            cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            sampler=train_sampler,
            seed=args.seed + fold_idx,
        )
        val_loader = make_loader(val_dataset, cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, seed=args.seed + fold_idx)
        test_loader = make_loader(test_dataset, cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, seed=args.seed + fold_idx)

        model = build_model(cfg.model).to(device)
        objective = build_objective(cfg.objective)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        trainer = Trainer(model, objective, optimizer, device, grad_clip=cfg.train.grad_clip)

        subject_dir = output_dir / str(test_subject)
        trainer.fit(train_loader, val_loader, cfg.train.epochs, subject_dir)

        best_model_path = subject_dir / "best.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        save_encoder_checkpoint(model, subject_dir / "best_encoder.pt", feature_cols)

        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)
        pred_df = trainer.predict(test_loader)
        pred_df.insert(0, "fold_idx", fold_idx)
        pred_df.insert(1, "test_subject", test_subject)
        all_predictions.append(pred_df)

        fold_metrics.append(
            {
                "fold_idx": fold_idx,
                "test_subject": test_subject,
                "val_subjects": "|".join(val_subjects),
                "n_train_subjects": int(train_df["participant_id"].astype(str).nunique()),
                "n_val_subjects": int(val_df["participant_id"].astype(str).nunique()),
                "n_test_subjects": int(test_df["participant_id"].astype(str).nunique()),
                "n_train_rows": int(len(train_df)),
                "n_val_rows": int(len(val_df)),
                "n_test_rows": int(len(test_df)),
                "train_sampler": args.train_sampler,
                "fold_source": spec["source"],
                "val_bacc": float(val_metrics["bacc"]),
                "val_f1": float(val_metrics["f1"]),
                "test_loss": float(test_metrics["loss"]),
                "test_acc": float(test_metrics["acc"]),
                "test_bacc": float(test_metrics["bacc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_auc": float(test_metrics["auc"]),
            }
        )
        print(
            f"fold={fold_idx} test_subject={test_subject} "
            f"val_subjects={val_subjects} test_metrics={test_metrics}"
        )

    if not all_predictions:
        raise RuntimeError("No folds executed.")

    pred_all = pd.concat(all_predictions, ignore_index=True)
    summary = {
        "n_folds": int(len(fold_metrics)),
        "n_predictions": int(len(pred_all)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(pred_all["y_true"], pred_all["y_pred"])),
        "pooled_micro_f1": float(f1_score(pred_all["y_true"], pred_all["y_pred"], average="micro")),
        "pooled_macro_f1": float(f1_score(pred_all["y_true"], pred_all["y_pred"], average="macro")),
        "drop_tasks": list(args.drop_tasks),
        "craving_splits_path": args.craving_splits_path,
        "parquet_path": args.parquet_path,
        "control_parquet_path": args.control_parquet_path,
        "extra_eval_subjects": [str(s) for s in args.extra_eval_subjects],
    }

    pd.DataFrame(fold_metrics).to_csv(output_dir / "stress_fold_metrics.csv", index=False)
    pred_all.to_parquet(output_dir / "stress_predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
