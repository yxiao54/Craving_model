import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import DataConfig


META_COLS = {
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
}


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_embeddings(path):
    with Path(path).open("rb") as handle:
        emb = pickle.load(handle)
    out = {}
    for key, value in emb.items():
        out[str(key)] = {
            "good_embedding": np.asarray(value["good_embedding"], dtype=np.float32),
            "bad_embedding": np.asarray(value["bad_embedding"], dtype=np.float32),
        }
    return out


def _load_split_json(path, fold_key):
    with Path(path).open() as handle:
        payload = json.load(handle)
    if fold_key is None:
        if isinstance(payload, dict) and len(payload) == 1:
            payload = next(iter(payload.values()))
        else:
            raise ValueError("fold_key is required when split_json contains multiple folds.")
    else:
        payload = payload[str(fold_key)]
    return payload["train"], payload.get("val", []), payload.get("test", [])


def _random_subject_split(subjects, val_ratio, test_ratio, seed):
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_subjects = subjects[:n_test]
    val_subjects = subjects[n_test : n_test + n_val]
    train_subjects = subjects[n_test + n_val :]
    if not train_subjects:
        raise ValueError("No training subjects left after split. Reduce val_ratio/test_ratio.")
    return train_subjects, val_subjects, test_subjects


def _build_subject_splits(df, cfg: DataConfig):
    subjects = df["participant_id"].astype(str).unique().tolist()
    if cfg.split_json:
        return _load_split_json(cfg.split_json, cfg.fold_key)
    return _random_subject_split(subjects, cfg.val_ratio, cfg.test_ratio, cfg.split_seed)


def _select_feature_columns(df, cfg: DataConfig):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if cfg.use_clean_numeric_only:
        feature_cols = numeric
    else:
        feature_cols = [c for c in feature_cols if c in numeric]
    if cfg.feature_drop_nan_threshold < 1.0:
        missing_rate = df[feature_cols].isna().mean()
        feature_cols = [c for c in feature_cols if missing_rate[c] <= cfg.feature_drop_nan_threshold]
    return feature_cols


def _normalize_by_subject(features_df, subject_ids, mode, eps=1e-6):
    if mode == "none":
        return features_df

    out = features_df.copy()
    subject_ids = pd.Series(subject_ids).astype(str).reset_index(drop=True)

    for subject_id in subject_ids.unique():
        mask = subject_ids == subject_id
        block = out.loc[mask].copy()

        if mode == "subject_zscore":
            center = block.mean(axis=0, skipna=True)
            scale = block.std(axis=0, skipna=True).replace(0.0, np.nan)
        elif mode == "subject_robust":
            center = block.median(axis=0, skipna=True)
            q75 = block.quantile(0.75)
            q25 = block.quantile(0.25)
            scale = (q75 - q25).replace(0.0, np.nan)
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

        scale = scale.fillna(1.0).astype(np.float32)
        center = center.fillna(0.0).astype(np.float32)
        out.loc[mask] = ((block - center) / (scale + eps)).astype(np.float32)

    return out


class OUDFeatureDataset(Dataset):
    def __init__(self, df, feature_cols, embeddings, label_name, environment_key, normalization="none"):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.embeddings = embeddings
        self.label_name = label_name
        self.environment_key = environment_key
        self.normalization = normalization

        self.df["participant_id"] = self.df["participant_id"].astype(str)
        self.df = self.df[self.df["participant_id"].isin(self.embeddings)].reset_index(drop=True)
        self.df = self.df[self.df[self.label_name].isin([0, 1])].reset_index(drop=True)

        self.features = (
            self.df[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
        self.fill_values = self.features.median(axis=0, skipna=True).fillna(0.0)
        self.features = self.features.fillna(self.fill_values)
        self.features = _normalize_by_subject(
            self.features,
            self.df["participant_id"].astype(str),
            mode=getattr(self, "normalization", "none"),
        )
        self.features = self.features.fillna(0.0).to_numpy(dtype=np.float32)

        self.labels = self.df[self.label_name].to_numpy(dtype=np.int64)
        self.stress_labels = self.df["stress"].fillna(0).to_numpy(dtype=np.int64)
        self.participant_ids = self.df["participant_id"].astype(str).to_numpy()
        self.tasks = self.df["task"].astype(str).to_numpy()
        self.environments = self.df[self.environment_key].astype(str).to_numpy()

        unique_subjects = sorted(self.df["participant_id"].astype(str).unique().tolist())
        unique_envs = sorted(self.df[self.environment_key].astype(str).unique().tolist())
        self.subject_to_index = {sid: i for i, sid in enumerate(unique_subjects)}
        self.env_to_index = {env: i for i, env in enumerate(unique_envs)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        participant_id = self.participant_ids[idx]
        env_name = self.environments[idx]
        embed = self.embeddings[participant_id]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "stress_label": torch.tensor(self.stress_labels[idx], dtype=torch.long),
            "good_embedding": torch.tensor(embed["good_embedding"], dtype=torch.float32),
            "bad_embedding": torch.tensor(embed["bad_embedding"], dtype=torch.float32),
            "participant_id": participant_id,
            "subject_index": torch.tensor(self.subject_to_index[participant_id], dtype=torch.long),
            "task": self.tasks[idx],
            "environment": env_name,
            "environment_index": torch.tensor(self.env_to_index[env_name], dtype=torch.long),
        }


def _collate(batch):
    out = {}
    tensor_keys = {"features", "label", "stress_label", "good_embedding", "bad_embedding", "subject_index", "environment_index"}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in tensor_keys:
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def load_base_resources(cfg: DataConfig, drop_tasks=None):
    df = pd.read_parquet(cfg.parquet_path)
    df["participant_id"] = df["participant_id"].astype(str)
    if drop_tasks:
        drop_set = set(map(str, drop_tasks))
        df = df[~df["task"].astype(str).isin(drop_set)].copy()
    feature_cols = _select_feature_columns(df, cfg)
    embeddings = _read_embeddings(cfg.embedding_path)
    return df, feature_cols, embeddings


def make_dataset(df, feature_cols, embeddings, cfg: DataConfig):
    return OUDFeatureDataset(
        df=df,
        feature_cols=feature_cols,
        embeddings=embeddings,
        label_name=cfg.label_name,
        environment_key=cfg.environment_key,
        normalization=cfg.normalization,
    )


def make_loader(dataset, batch_size, shuffle, num_workers, sampler=None, seed=2026):
    if len(dataset) == 0:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_weighted_sampler(dataset, seed=2026):
    if len(dataset) == 0:
        return None
    labels = dataset.labels
    unique, counts = np.unique(labels, return_counts=True)
    label_to_weight = {int(label): float(len(labels)) / float(count) for label, count in zip(unique, counts)}
    weights = [label_to_weight[int(label)] for label in labels]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(cfg: DataConfig):
    df, feature_cols, embeddings = load_base_resources(cfg)

    train_subjects, val_subjects, test_subjects = _build_subject_splits(df, cfg)
    split_map = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    loaders = {}
    datasets = {}
    for split_name, subject_ids in split_map.items():
        split_df = df[df["participant_id"].astype(str).isin(set(map(str, subject_ids)))].copy()
        ds = OUDFeatureDataset(
            df=split_df,
            feature_cols=feature_cols,
            embeddings=embeddings,
            label_name=cfg.label_name,
            environment_key=cfg.environment_key,
            normalization=cfg.normalization,
        )
        datasets[split_name] = ds
        loaders[split_name] = make_loader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split_name == "train"),
            num_workers=cfg.num_workers,
            sampler=None,
        )

    metadata = {
        "feature_columns": feature_cols,
        "input_dim": len(feature_cols),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "config": asdict(cfg),
    }
    return loaders, metadata, datasets
