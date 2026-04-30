import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import pandas as pd


def _safe_auc(y_true, y_prob):
    if len(set(y_true)) < 2:
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
    return roc_auc_score(y_true_arr, y_prob_arr)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Trainer:
    def __init__(self, model, objective, optimizer, device, grad_clip=1.0, scheduler=None):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.scheduler = scheduler

    def _move_batch(self, batch):
        return {
            "features": batch["features"].to(self.device),
            "labels": batch["label"].to(self.device),
            "stress_labels": batch["stress_label"].to(self.device),
            "good_embedding": batch["good_embedding"].to(self.device),
            "bad_embedding": batch["bad_embedding"].to(self.device),
            "environment_index": batch["environment_index"].to(self.device),
            "participant_id": batch["participant_id"],
            "task": batch["task"],
            "environment": batch["environment"],
        }

    def _run_loader(self, loader, train):
        if loader is None:
            return {"loss": 0.0, "acc": 0.0, "bacc": 0.0, "f1": 0.0, "auc": 0.0}

        self.model.train(mode=train)
        total_loss = 0.0
        total_steps = 0
        all_y, all_hat, all_prob = [], [], []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                batch = self._move_batch(batch)
                if train:
                    self.optimizer.zero_grad()

                output = self.model(
                    batch["features"],
                    batch["good_embedding"],
                    batch["bad_embedding"],
                )
                loss, logs = self.objective(
                    output,
                    batch["labels"],
                    stress_labels=batch["stress_labels"],
                    environment_index=batch["environment_index"],
                )

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                logits = output["logits"]
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)

                total_loss += float(loss.detach())
                total_steps += 1
                all_y.extend(batch["labels"].detach().cpu().tolist())
                all_hat.extend(preds.detach().cpu().tolist())
                all_prob.extend(probs.detach().cpu().tolist())

        return {
            "loss": total_loss / max(total_steps, 1),
            "acc": accuracy_score(all_y, all_hat) if all_y else 0.0,
            "bacc": balanced_accuracy_score(all_y, all_hat) if all_y else 0.0,
            "f1": f1_score(all_y, all_hat, average="macro") if all_y else 0.0,
            "auc": _safe_auc(all_y, all_prob) if all_y else 0.0,
        }

    def fit(self, train_loader, val_loader, epochs, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True)
            val_metrics = self._run_loader(val_loader, train=False)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

        torch.save(self.model.state_dict(), output_dir / "best.pt")
        torch.save(self.model.state_dict(), output_dir / "last.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        return history

    def evaluate(self, loader):
        return self._run_loader(loader, train=False)

    def predict(self, loader):
        if loader is None:
            return pd.DataFrame(
                columns=[
                    "participant_id",
                    "task",
                    "environment",
                    "y_true",
                    "y_pred",
                    "y_prob",
                    "logit_0",
                    "logit_1",
                    "stress_y_true",
                    "stress_y_pred",
                    "stress_y_prob",
                    "stress_logit_0",
                    "stress_logit_1",
                ]
            )

        self.model.eval()
        rows = []
        with torch.no_grad():
            for batch in loader:
                moved = self._move_batch(batch)
                output = self.model(
                    moved["features"],
                    moved["good_embedding"],
                    moved["bad_embedding"],
                )
                logits = output["logits"]
                logits_cpu = logits.detach().cpu().tolist()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                labels = moved["labels"].detach().cpu().tolist()
                stress_labels = moved["stress_labels"].detach().cpu().tolist()

                if "stress_logits" in output:
                    stress_logits = output["stress_logits"]
                    stress_logits_cpu = stress_logits.detach().cpu().tolist()
                    stress_probs = torch.softmax(stress_logits, dim=1)[:, 1].detach().cpu().tolist()
                    stress_preds = stress_logits.argmax(dim=1).detach().cpu().tolist()
                else:
                    stress_logits_cpu = [[float("nan"), float("nan")] for _ in labels]
                    stress_probs = [float("nan") for _ in labels]
                    stress_preds = [None for _ in labels]

                for (
                    participant_id,
                    task,
                    environment,
                    y_true,
                    y_pred,
                    y_prob,
                    raw_logits,
                    stress_y_true,
                    stress_y_pred,
                    stress_y_prob,
                    stress_raw_logits,
                ) in zip(
                    moved["participant_id"],
                    moved["task"],
                    moved["environment"],
                    labels,
                    preds,
                    probs,
                    logits_cpu,
                    stress_labels,
                    stress_preds,
                    stress_probs,
                    stress_logits_cpu,
                ):
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "task": task,
                            "environment": environment,
                            "y_true": int(y_true),
                            "y_pred": int(y_pred),
                            "y_prob": float(y_prob),
                            "logit_0": float(raw_logits[0]),
                            "logit_1": float(raw_logits[1]),
                            "stress_y_true": int(stress_y_true),
                            "stress_y_pred": None if stress_y_pred is None else int(stress_y_pred),
                            "stress_y_prob": float(stress_y_prob) if stress_y_prob == stress_y_prob else float("nan"),
                            "stress_logit_0": float(stress_raw_logits[0]) if stress_raw_logits[0] == stress_raw_logits[0] else float("nan"),
                            "stress_logit_1": float(stress_raw_logits[1]) if stress_raw_logits[1] == stress_raw_logits[1] else float("nan"),
                        }
                    )
        return pd.DataFrame(rows)
