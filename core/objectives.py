from collections import defaultdict

import torch
import torch.nn.functional as F


def _cross_entropy_per_sample(logits, labels):
    return F.cross_entropy(logits, labels, reduction="none")


def _group_means(losses, group_indices):
    grouped = defaultdict(list)
    for idx, env in enumerate(group_indices.tolist()):
        grouped[int(env)].append(losses[idx])
    return [(env_id, torch.stack(vals).mean()) for env_id, vals in grouped.items() if vals]


class FlexibleObjective:
    def __init__(
        self,
        name="erm",
        irm_lambda=1.0,
        vrex_lambda=1.0,
        dro_eta=0.1,
        aux_ce_weight=0.0,
        alignment_weight=0.1,
        sparse_gate_weight=1e-3,
        modality_gate_sparsity_weight=1e-3,
        feature_gate_sparsity_weight=1e-3,
        label_smoothing=0.0,
        moe_load_balance_weight=1e-2,
        stress_aux_weight=0.0,
        orthogonality_weight=0.0,
        semantic_kl_weight=0.0,
    ):
        self.name = name.lower()
        self.irm_lambda = irm_lambda
        self.vrex_lambda = vrex_lambda
        self.dro_eta = dro_eta
        self.aux_ce_weight = aux_ce_weight
        self.alignment_weight = alignment_weight
        self.sparse_gate_weight = sparse_gate_weight
        self.modality_gate_sparsity_weight = modality_gate_sparsity_weight
        self.feature_gate_sparsity_weight = feature_gate_sparsity_weight
        self.label_smoothing = label_smoothing
        self.moe_load_balance_weight = moe_load_balance_weight
        self.stress_aux_weight = stress_aux_weight
        self.orthogonality_weight = orthogonality_weight
        self.semantic_kl_weight = semantic_kl_weight
        self.group_weights = {}

    def __call__(self, output, labels, stress_labels=None, environment_index=None):
        logits = output["logits"]
        base_losses = F.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        base_loss = base_losses.mean()
        total_loss = base_loss
        logs = {"erm": float(base_loss.detach())}

        if "similarity_logits" in output:
            sim_loss = F.cross_entropy(
                output["similarity_logits"],
                labels,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + self.alignment_weight * sim_loss
            logs["alignment"] = float(sim_loss.detach())

        if self.sparse_gate_weight > 0.0 and "gate_values" in output:
            gate_penalty = output["gate_values"].mean()
            total_loss = total_loss + self.sparse_gate_weight * gate_penalty
            logs["sparse_gate"] = float(gate_penalty.detach())

        if self.modality_gate_sparsity_weight > 0.0 and "modality_gate_values" in output:
            modality_gate_penalty = output["modality_gate_values"].mean()
            total_loss = total_loss + self.modality_gate_sparsity_weight * modality_gate_penalty
            logs["modality_gate_sparse"] = float(modality_gate_penalty.detach())

        if self.feature_gate_sparsity_weight > 0.0 and "feature_gate_values" in output:
            feature_gate_penalty = output["feature_gate_values"].mean()
            total_loss = total_loss + self.feature_gate_sparsity_weight * feature_gate_penalty
            logs["feature_gate_sparse"] = float(feature_gate_penalty.detach())

        if self.moe_load_balance_weight > 0.0 and "router_probs" in output:
            router_probs = output["router_probs"]
            expert_usage = router_probs.mean(dim=0)
            uniform = torch.full_like(expert_usage, 1.0 / expert_usage.numel())
            load_penalty = torch.sum((expert_usage - uniform) ** 2)
            total_loss = total_loss + self.moe_load_balance_weight * load_penalty
            logs["moe_load_balance"] = float(load_penalty.detach())

        if self.aux_ce_weight > 0.0 and "logits_phys" in output:
            aux_loss = F.cross_entropy(
                output["logits_phys"],
                labels,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + self.aux_ce_weight * aux_loss
            logs["aux_ce"] = float(aux_loss.detach())

        if self.stress_aux_weight > 0.0 and stress_labels is not None and "stress_logits" in output:
            stress_aux_loss = F.cross_entropy(
                output["stress_logits"],
                stress_labels,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + self.stress_aux_weight * stress_aux_loss
            logs["stress_aux"] = float(stress_aux_loss.detach())

        if self.orthogonality_weight > 0.0 and "stress_hidden_projected" in output and "craving_hidden" in output:
            a = F.normalize(output["stress_hidden_projected"], dim=-1)
            b = F.normalize(output["craving_hidden"], dim=-1)
            ortho_loss = torch.mean(torch.sum(a * b, dim=-1).pow(2))
            total_loss = total_loss + self.orthogonality_weight * ortho_loss
            logs["orthogonality"] = float(ortho_loss.detach())

        if self.semantic_kl_weight > 0.0 and "semantic_kl" in output:
            semantic_kl = output["semantic_kl"]
            total_loss = total_loss + self.semantic_kl_weight * semantic_kl
            logs["semantic_kl"] = float(semantic_kl.detach())

        if environment_index is None or self.name == "erm":
            logs["loss"] = float(total_loss.detach())
            return total_loss, logs

        env_losses = _group_means(base_losses, environment_index)
        if not env_losses:
            logs["loss"] = float(total_loss.detach())
            return total_loss, logs

        if self.name == "irm":
            if not torch.is_grad_enabled() or not base_loss.requires_grad:
                penalty = torch.tensor(0.0, device=logits.device)
            else:
                scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
                penalty = torch.tensor(0.0, device=logits.device)
                for _, env_loss in env_losses:
                    grad = torch.autograd.grad(env_loss * scale, [scale], create_graph=True)[0]
                    penalty = penalty + grad.pow(2)
            total_loss = total_loss + self.irm_lambda * penalty
            logs["irm_penalty"] = float(penalty.detach())
        elif self.name == "vrex":
            env_stack = torch.stack([env_loss for _, env_loss in env_losses])
            penalty = env_stack.var(unbiased=False)
            total_loss = total_loss + self.vrex_lambda * penalty
            logs["vrex_penalty"] = float(penalty.detach())
        elif self.name == "groupdro":
            env_stack = []
            env_ids = []
            for env_id, env_loss in env_losses:
                prev = self.group_weights.get(env_id, 1.0)
                updated = prev * torch.exp(self.dro_eta * env_loss.detach()).item()
                self.group_weights[env_id] = updated
                env_ids.append(env_id)
                env_stack.append(env_loss)
            weight_tensor = torch.tensor(
                [self.group_weights[env_id] for env_id in env_ids],
                device=logits.device,
                dtype=torch.float32,
            )
            weight_tensor = weight_tensor / weight_tensor.sum()
            penalty = torch.sum(weight_tensor * torch.stack(env_stack))
            total_loss = penalty + (total_loss - base_loss)
            logs["groupdro_weighted"] = float(penalty.detach())
        else:
            raise ValueError(f"Unsupported objective: {self.name}")

        logs["loss"] = float(total_loss.detach())
        return total_loss, logs


def build_objective(cfg):
    return FlexibleObjective(
        name=cfg.name,
        irm_lambda=cfg.irm_lambda,
        vrex_lambda=cfg.vrex_lambda,
        dro_eta=cfg.dro_eta,
        aux_ce_weight=cfg.aux_ce_weight,
        alignment_weight=cfg.alignment_weight,
        sparse_gate_weight=cfg.sparse_gate_weight,
        modality_gate_sparsity_weight=cfg.modality_gate_sparsity_weight,
        feature_gate_sparsity_weight=cfg.feature_gate_sparsity_weight,
        label_smoothing=cfg.label_smoothing,
        moe_load_balance_weight=cfg.moe_load_balance_weight,
        stress_aux_weight=cfg.stress_aux_weight,
        orthogonality_weight=cfg.orthogonality_weight,
        semantic_kl_weight=cfg.semantic_kl_weight,
    )
