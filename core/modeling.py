import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUD_FINAL_DIR = ROOT / "OUD_final"
if str(OUD_FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(OUD_FINAL_DIR))

from models.factory import get_model as get_legacy_model  # noqa: E402


def _load_stress_encoder_from_checkpoint(path):
    payload = torch.load(path, map_location="cpu")
    encoder_state = payload["encoder_state_dict"]
    linear_keys = sorted(
        [k for k in encoder_state.keys() if k.endswith(".weight") and encoder_state[k].ndim == 2],
        key=lambda x: int(x.split(".")[0]),
    )
    if not linear_keys:
        raise ValueError(f"No linear layers found in stress encoder checkpoint: {path}")

    layers = []
    for key in linear_keys:
        weight = encoder_state[key]
        out_dim, in_dim = weight.shape
        layers.extend(
            [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(0.0),
            ]
        )
    encoder = nn.Sequential(*layers)
    encoder.load_state_dict(encoder_state, strict=True)
    hidden_dim = encoder_state[linear_keys[-1]].shape[0]
    return encoder, hidden_dim


class SemanticGuidedClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_similarity_head=True,
        use_logit_scale=True,
        use_film=True,
        use_sparse_gate=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_similarity_head = use_similarity_head
        self.use_logit_scale = use_logit_scale
        self.use_film = use_film
        self.use_sparse_gate = use_sparse_gate
        self.hidden_dim = hidden_dim

        encoder_layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.phys_encoder = nn.Sequential(*encoder_layers)

        self.embedding_dim = None
        self.embedding_proj = None
        self.film_proj = None
        self.gate_proj = None
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
        classifier_layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)])
        self.classifier = nn.Sequential(*classifier_layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _build_embedding_layers(self, good_embedding, bad_embedding, hidden_dim):
        emb_dim = 0
        if self.use_good_embedding:
            emb_dim += good_embedding.shape[-1]
        if self.use_bad_embedding:
            emb_dim += bad_embedding.shape[-1]
        self.embedding_dim = emb_dim
        self.embedding_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if self.use_film:
            self.film_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        if self.use_sparse_gate:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one of use_good_embedding/use_bad_embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        hidden_dim = self.hidden_dim
        if self.embedding_proj is None:
            self._build_embedding_layers(good_embedding, bad_embedding, hidden_dim)
            self.embedding_proj = self.embedding_proj.to(features.device)
            if self.film_proj is not None:
                self.film_proj = self.film_proj.to(features.device)
            if self.gate_proj is not None:
                self.gate_proj = self.gate_proj.to(features.device)

        semantics = self._combine_semantics(good_embedding, bad_embedding)
        semantic_hidden = self.embedding_proj(semantics)

        phys_hidden = self.phys_encoder(features)
        if self.use_film:
            gamma, beta = self.film_proj(semantic_hidden).chunk(2, dim=-1)
            gamma = 1.0 + 0.2 * torch.tanh(gamma)
            beta = 0.2 * torch.tanh(beta)
            phys_hidden = gamma * phys_hidden + beta

        gate_values = None
        if self.use_sparse_gate:
            gate_values = torch.sigmoid(self.gate_proj(semantic_hidden))
            phys_hidden = phys_hidden * gate_values

        logits = self.classifier(phys_hidden)
        output = {
            "logits": logits,
            "phys_hidden": phys_hidden,
            "semantic_hidden": semantic_hidden,
        }
        if gate_values is not None:
            output["gate_values"] = gate_values

        if self.use_similarity_head:
            good_ref = F.normalize(good_embedding, dim=-1)
            bad_ref = F.normalize(bad_embedding, dim=-1)
            good_proj = F.normalize(F.linear(good_ref, torch.eye(good_ref.shape[-1], device=good_ref.device)), dim=-1)
            bad_proj = F.normalize(F.linear(bad_ref, torch.eye(bad_ref.shape[-1], device=bad_ref.device)), dim=-1)
            aligned = F.normalize(semantic_hidden, dim=-1)
            if aligned.shape[-1] != good_proj.shape[-1]:
                projector = getattr(self, "_sim_projector", None)
                if projector is None:
                    self._sim_projector = nn.Linear(aligned.shape[-1], good_proj.shape[-1]).to(aligned.device)
                    projector = self._sim_projector
                aligned = F.normalize(projector(aligned), dim=-1)
            sim_good = (aligned * good_proj).sum(dim=-1, keepdim=True)
            sim_bad = (aligned * bad_proj).sum(dim=-1, keepdim=True)
            similarity_logits = torch.cat([sim_bad, sim_good], dim=-1)
            if self.use_logit_scale:
                similarity_logits = similarity_logits * self.logit_scale.exp().clamp(max=100.0)
            output["similarity_logits"] = similarity_logits

        return output


class SemanticConcatClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.hidden_dim = hidden_dim

        phys_layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            phys_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.phys_encoder = nn.Sequential(*phys_layers)

        self.semantic_proj = None
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        phys_hidden = self.phys_encoder(features)
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantics.shape[-1], self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ).to(features.device)
        semantic_hidden = self.semantic_proj(semantics)
        fused = self.fusion(torch.cat([phys_hidden, semantic_hidden], dim=-1))
        return {
            "logits": self.classifier(fused),
            "phys_hidden": phys_hidden,
            "semantic_hidden": semantic_hidden,
        }


class SemanticResidualClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_logit_scale=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_logit_scale = use_logit_scale
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.phys_encoder = nn.Sequential(*layers)
        self.semantic_proj = None
        self.alpha_proj = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        phys_hidden = self.phys_encoder(features)
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantics.shape[-1], self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ).to(features.device)
        semantic_hidden = self.semantic_proj(semantics)
        alpha = self.alpha_proj(semantic_hidden)
        fused = phys_hidden + alpha * semantic_hidden
        logits = self.classifier(fused)
        if self.use_logit_scale:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)
        return {
            "logits": logits,
            "phys_hidden": phys_hidden,
            "semantic_hidden": semantic_hidden,
            "alpha": alpha,
        }


class SemanticTokenAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.phys_encoder = nn.Sequential(*layers)
        self.semantic_proj = None
        self.token_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        phys_hidden = self.phys_encoder(features)
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantics.shape[-1], self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ).to(features.device)
        semantic_hidden = self.semantic_proj(semantics)
        tokens = torch.stack([phys_hidden, semantic_hidden], dim=1)
        attn = torch.softmax(self.token_scorer(tokens), dim=1)
        fused = self.fusion_proj((attn * tokens).sum(dim=1))
        return {
            "logits": self.classifier(fused),
            "phys_hidden": phys_hidden,
            "semantic_hidden": semantic_hidden,
            "attn": attn,
        }


class SemanticInputFiLMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_logit_scale=False,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_logit_scale = use_logit_scale
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.phys_encoder = nn.Sequential(*layers)
        self.semantic_proj = None
        self.input_film_proj = None

        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None or self.input_film_proj is None:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantics.shape[-1], self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ).to(features.device)
            self.input_film_proj = nn.Linear(self.hidden_dim, self.input_dim * 2).to(features.device)

        semantic_hidden = self.semantic_proj(semantics)
        gamma, beta = self.input_film_proj(semantic_hidden).chunk(2, dim=-1)
        gamma = 1.0 + 0.2 * torch.tanh(gamma)
        beta = 0.2 * torch.tanh(beta)
        modulated_features = gamma * features + beta
        phys_hidden = self.phys_encoder(modulated_features)
        logits = self.classifier(phys_hidden)
        if self.use_logit_scale:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)
        return {
            "logits": logits,
            "phys_hidden": phys_hidden,
            "semantic_hidden": semantic_hidden,
            "input_film_values": modulated_features,
        }


class SemanticMoEClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        num_experts=4,
        top_k_experts=0,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_logit_scale=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_logit_scale = use_logit_scale
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts

        self.semantic_proj = None
        self.router = None
        self.experts = nn.ModuleList(
            [
                self._make_expert(input_dim, hidden_dim, num_encoder_layers, dropout)
                for _ in range(num_experts)
            ]
        )
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _make_expert(input_dim, hidden_dim, num_layers, dropout):
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def _init_semantic_modules(self, semantic_dim, device):
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        ).to(device)
        self.router = nn.Linear(self.hidden_dim, self.num_experts).to(device)

    def forward(self, features, good_embedding, bad_embedding):
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None or self.router is None:
            self._init_semantic_modules(semantics.shape[-1], features.device)

        semantic_hidden = self.semantic_proj(semantics)
        router_logits = self.router(semantic_hidden)
        router_probs = torch.softmax(router_logits, dim=-1)

        if self.top_k_experts and self.top_k_experts < self.num_experts:
            top_vals, top_idx = torch.topk(router_probs, k=self.top_k_experts, dim=-1)
            masked = torch.zeros_like(router_probs)
            masked.scatter_(1, top_idx, top_vals)
            router_probs = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = [expert(features) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)
        fused = (router_probs.unsqueeze(-1) * expert_stack).sum(dim=1)
        logits = self.classifier(fused)
        if self.use_logit_scale:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)

        return {
            "logits": logits,
            "phys_hidden": fused,
            "semantic_hidden": semantic_hidden,
            "router_probs": router_probs,
            "router_logits": router_logits,
        }


class SemanticHierarchicalModalityClassifier(nn.Module):
    GROUP_PREFIXES = {
        "hr": ("HR_", "HRV_"),
        "bvp": ("BVP_", "PPG_"),
        "eda": ("EDA_", "RawEDA_", "Tonic_", "Phasic_"),
        "acc": ("ACC_",),
        "temp": ("TEMP_",),
    }

    def __init__(
        self,
        input_dim,
        feature_columns=None,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_logit_scale=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_logit_scale = use_logit_scale
        self.hidden_dim = hidden_dim
        self.feature_columns = feature_columns or [f"feat_{i}" for i in range(input_dim)]

        self.group_indices = self._build_group_indices(self.feature_columns)
        self.group_names = list(self.group_indices.keys())
        self.group_encoders = nn.ModuleDict(
            {
                name: self._make_encoder(len(indices), hidden_dim, num_encoder_layers, dropout)
                for name, indices in self.group_indices.items()
            }
        )
        self.semantic_proj = None
        self.modality_router = None
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @classmethod
    def _build_group_indices(cls, feature_columns):
        groups = {name: [] for name in cls.GROUP_PREFIXES}
        other = []
        for idx, col in enumerate(feature_columns):
            matched = False
            for group_name, prefixes in cls.GROUP_PREFIXES.items():
                if any(str(col).startswith(prefix) for prefix in prefixes):
                    groups[group_name].append(idx)
                    matched = True
                    break
            if not matched:
                other.append(idx)
        groups = {k: v for k, v in groups.items() if v}
        if other:
            groups["other"] = other
        if not groups:
            groups["all"] = list(range(len(feature_columns)))
        return groups

    @staticmethod
    def _make_encoder(input_dim, hidden_dim, num_layers, dropout):
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)])
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def forward(self, features, good_embedding, bad_embedding):
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None or self.modality_router is None:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantics.shape[-1], self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ).to(features.device)
            self.modality_router = nn.Linear(self.hidden_dim, len(self.group_names)).to(features.device)

        semantic_hidden = self.semantic_proj(semantics)
        modality_tokens = []
        for group_name in self.group_names:
            indices = self.group_indices[group_name]
            idx = torch.as_tensor(indices, device=features.device, dtype=torch.long)
            group_feat = torch.index_select(features, dim=1, index=idx)
            modality_tokens.append(self.group_encoders[group_name](group_feat))
        modality_stack = torch.stack(modality_tokens, dim=1)
        modality_weights = torch.softmax(self.modality_router(semantic_hidden), dim=-1)
        fused = (modality_weights.unsqueeze(-1) * modality_stack).sum(dim=1)
        fused = self.fusion_proj(fused)
        logits = self.classifier(fused)
        if self.use_logit_scale:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)
        return {
            "logits": logits,
            "phys_hidden": fused,
            "semantic_hidden": semantic_hidden,
            "modality_weights": modality_weights,
            "group_names": self.group_names,
        }


class SemanticHierarchicalFeatureGateClassifier(nn.Module):
    GROUP_PREFIXES = SemanticHierarchicalModalityClassifier.GROUP_PREFIXES

    def __init__(
        self,
        input_dim,
        feature_columns=None,
        hidden_dim=128,
        gate_hidden_dim=64,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_logit_scale=True,
    ):
        super().__init__()
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_logit_scale = use_logit_scale
        self.hidden_dim = hidden_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.feature_columns = feature_columns or [f"feat_{i}" for i in range(input_dim)]

        self.group_indices = self._build_group_indices(self.feature_columns)
        self.group_names = list(self.group_indices.keys())
        self.group_encoders = nn.ModuleDict(
            {
                name: self._make_encoder(len(indices), hidden_dim, num_encoder_layers, dropout)
                for name, indices in self.group_indices.items()
            }
        )
        self.semantic_proj = None
        self.modality_gate_proj = None
        self.feature_gate_projs = None
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @classmethod
    def _build_group_indices(cls, feature_columns):
        return SemanticHierarchicalModalityClassifier._build_group_indices(feature_columns)

    @staticmethod
    def _make_encoder(input_dim, hidden_dim, num_layers, dropout):
        return SemanticHierarchicalModalityClassifier._make_encoder(input_dim, hidden_dim, num_layers, dropout)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def _init_semantic_modules(self, semantic_dim, device):
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.gate_hidden_dim),
        ).to(device)
        self.modality_gate_proj = nn.Linear(self.gate_hidden_dim, len(self.group_names)).to(device)
        self.feature_gate_projs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.gate_hidden_dim, len(indices)),
                )
                for name, indices in self.group_indices.items()
            }
        ).to(device)

    def forward(self, features, good_embedding, bad_embedding):
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None or self.modality_gate_proj is None or self.feature_gate_projs is None:
            self._init_semantic_modules(semantics.shape[-1], features.device)

        semantic_hidden = self.semantic_proj(semantics)
        modality_gate_values = torch.softmax(self.modality_gate_proj(semantic_hidden), dim=-1)

        modality_tokens = []
        feature_gate_values = []
        for group_idx, group_name in enumerate(self.group_names):
            indices = self.group_indices[group_name]
            idx = torch.as_tensor(indices, device=features.device, dtype=torch.long)
            group_feat = torch.index_select(features, dim=1, index=idx)
            local_gate = torch.sigmoid(self.feature_gate_projs[group_name](semantic_hidden))
            gated_group_feat = group_feat * local_gate
            encoded = self.group_encoders[group_name](gated_group_feat)
            encoded = encoded * modality_gate_values[:, group_idx].unsqueeze(-1)
            modality_tokens.append(encoded)
            feature_gate_values.append(local_gate)

        fused = torch.stack(modality_tokens, dim=1).sum(dim=1)
        fused = self.fusion_proj(fused)
        logits = self.classifier(fused)
        if self.use_logit_scale:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)

        return {
            "logits": logits,
            "phys_hidden": fused,
            "semantic_hidden": semantic_hidden,
            "modality_gate_values": modality_gate_values,
            "feature_gate_values": torch.cat(feature_gate_values, dim=-1),
            "group_names": self.group_names,
        }


class LegacyModelAdapter(nn.Module):
    def __init__(self, model_name, input_dim, num_classes, baseline_ckpt=None):
        super().__init__()
        self.model = get_legacy_model(
            model_name=model_name,
            input_dim=input_dim,
            num_class=num_classes,
            baseline_ckpt=baseline_ckpt,
        )

    def forward(self, features, good_embedding, bad_embedding):
        output = self.model(features, good_embedding, bad_embedding)
        if isinstance(output, dict):
            return output
        return {"logits": output}


class StressEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_encoder_layers=2,
        num_classifier_layers=1,
        num_classes=2,
        dropout=0.2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_encoder_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def encode(self, features):
        return self.encoder(features)

    def forward(self, features, good_embedding=None, bad_embedding=None):
        hidden = self.encode(features)
        logits = self.classifier(hidden)
        return {
            "logits": logits,
            "phys_hidden": hidden,
        }


class SemanticGuidedFrozenStressClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        stress_encoder_ckpt,
        hidden_dim=128,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_film=True,
        use_sparse_gate=True,
        freeze_stress_encoder=True,
    ):
        super().__init__()
        if not stress_encoder_ckpt:
            raise ValueError("stress_encoder_ckpt is required for SemanticGuidedFrozenStressClassifier.")
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_film = use_film
        self.use_sparse_gate = use_sparse_gate
        self.freeze_stress_encoder = freeze_stress_encoder

        self.stress_encoder, stress_hidden_dim = _load_stress_encoder_from_checkpoint(stress_encoder_ckpt)
        self.stress_hidden_dim = stress_hidden_dim
        if self.freeze_stress_encoder:
            for param in self.stress_encoder.parameters():
                param.requires_grad = False

        self.semantic_proj = None
        self.film_proj = None
        self.gate_proj = None

        fusion_in_dim = stress_hidden_dim
        fusion_hidden = hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout),
        )
        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(fusion_hidden, fusion_hidden), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(fusion_hidden, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_good_embedding:
            parts.append(good_embedding)
        if self.use_bad_embedding:
            parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def _init_semantic_layers(self, semantic_dim, device):
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, self.stress_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.stress_hidden_dim),
        ).to(device)
        if self.use_film:
            self.film_proj = nn.Linear(self.stress_hidden_dim, self.stress_hidden_dim * 2).to(device)
        if self.use_sparse_gate:
            self.gate_proj = nn.Linear(self.stress_hidden_dim, self.stress_hidden_dim).to(device)

    def forward(self, features, good_embedding, bad_embedding):
        if self.freeze_stress_encoder:
            with torch.no_grad():
                stress_hidden = self.stress_encoder(features)
        else:
            stress_hidden = self.stress_encoder(features)

        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None:
            self._init_semantic_layers(semantics.shape[-1], features.device)
        semantic_hidden = self.semantic_proj(semantics)

        fused = stress_hidden
        if self.use_film:
            gamma, beta = self.film_proj(semantic_hidden).chunk(2, dim=-1)
            gamma = 1.0 + 0.2 * torch.tanh(gamma)
            beta = 0.2 * torch.tanh(beta)
            fused = gamma * fused + beta

        gate_values = None
        if self.use_sparse_gate:
            gate_values = torch.sigmoid(self.gate_proj(semantic_hidden))
            fused = fused * gate_values

        fused = self.fusion_proj(fused)
        logits = self.classifier(fused)
        output = {
            "logits": logits,
            "phys_hidden": fused,
            "stress_hidden": stress_hidden,
            "semantic_hidden": semantic_hidden,
        }
        if gate_values is not None:
            output["gate_values"] = gate_values
        return output


class SemanticDualPathFrozenStressClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        stress_encoder_ckpt,
        hidden_dim=128,
        gate_hidden_dim=64,
        num_encoder_layers=2,
        num_classifier_layers=2,
        num_classes=2,
        dropout=0.2,
        use_good_embedding=True,
        use_bad_embedding=True,
        use_global_guidance=False,
        use_input_feature_gate=True,
        use_input_film=False,
        use_trainable_phys_branch=True,
        use_stress_branch=True,
        fusion_mode="sample_gate",
        use_logit_scale=True,
        use_affine_calibration=False,
        use_semantic_uncertainty=False,
        use_stress_aux_head=False,
        freeze_stress_encoder=True,
    ):
        super().__init__()
        if not stress_encoder_ckpt:
            raise ValueError("stress_encoder_ckpt is required for SemanticDualPathFrozenStressClassifier.")
        if not use_trainable_phys_branch and not use_stress_branch:
            raise ValueError("At least one of use_trainable_phys_branch/use_stress_branch must be enabled.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.use_good_embedding = use_good_embedding
        self.use_bad_embedding = use_bad_embedding
        self.use_global_guidance = use_global_guidance
        self.use_input_feature_gate = use_input_feature_gate
        self.use_input_film = use_input_film
        self.use_trainable_phys_branch = use_trainable_phys_branch
        self.use_stress_branch = use_stress_branch
        self.fusion_mode = fusion_mode
        self.use_logit_scale = use_logit_scale
        self.use_affine_calibration = use_affine_calibration
        self.use_semantic_uncertainty = use_semantic_uncertainty
        self.use_stress_aux_head = use_stress_aux_head
        self.freeze_stress_encoder = freeze_stress_encoder

        self.stress_encoder, stress_hidden_dim = _load_stress_encoder_from_checkpoint(stress_encoder_ckpt)
        self.stress_hidden_dim = stress_hidden_dim
        if self.freeze_stress_encoder:
            for param in self.stress_encoder.parameters():
                param.requires_grad = False

        self.semantic_proj = None
        self.semantic_mu_proj = None
        self.semantic_logvar_proj = None
        self.semantic_gate_proj = None
        self.input_gate_proj = None
        self.input_film_gamma = None
        self.input_film_beta = None
        self.semantic_fusion_proj = None
        self.decision_scale_proj = None
        self.decision_bias_proj = None
        self.fusion_mlp = None
        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        self.latent_gate_proj = None
        self.latent_film_proj = None
        self.global_good_embedding = None
        self.global_bad_embedding = None

        if self.use_trainable_phys_branch:
            encoder_layers = []
            in_dim = input_dim
            for _ in range(num_encoder_layers):
                encoder_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            self.phys_encoder = nn.Sequential(*encoder_layers)
        else:
            self.phys_encoder = None

        self.stress_proj = None
        if self.use_stress_branch:
            self.stress_proj = nn.Sequential(
                nn.Linear(stress_hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

        if self.use_stress_branch and self.use_trainable_phys_branch:
            if fusion_mode == "sample_gate":
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 3, gate_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(gate_hidden_dim),
                    nn.Linear(gate_hidden_dim, 1),
                )
                self.fusion_weight_logits = None
            elif fusion_mode == "scalar":
                self.fusion_weight_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
                self.fusion_gate = None
            elif fusion_mode == "average":
                self.fusion_gate = None
                self.fusion_weight_logits = None
            elif fusion_mode == "residual":
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 3, gate_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(gate_hidden_dim),
                    nn.Linear(gate_hidden_dim, hidden_dim),
                )
                self.fusion_weight_logits = None
            elif fusion_mode == "mixture":
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 3, gate_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(gate_hidden_dim),
                    nn.Linear(gate_hidden_dim, 3),
                )
                self.fusion_weight_logits = None
            elif fusion_mode == "concat_mlp":
                self.fusion_gate = None
                self.fusion_weight_logits = None
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(hidden_dim * 6, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
            elif fusion_mode == "attention":
                self.fusion_gate = None
                self.fusion_weight_logits = None
                self.query_proj = nn.Linear(hidden_dim, hidden_dim)
                self.key_proj = nn.Linear(hidden_dim, hidden_dim)
                self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            elif fusion_mode == "cross_gate":
                self.fusion_gate = None
                self.fusion_weight_logits = None
                self.latent_gate_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                )
            elif fusion_mode == "cross_film":
                self.fusion_gate = None
                self.fusion_weight_logits = None
                self.latent_film_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim * 4),
                )
            else:
                raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        else:
            self.fusion_gate = None
            self.fusion_weight_logits = None

        classifier_layers = []
        for _ in range(max(num_classifier_layers - 1, 0)):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.stress_aux_head = None
        if self.use_stress_aux_head and self.use_trainable_phys_branch:
            self.stress_aux_head = nn.Linear(hidden_dim, num_classes)

    def _combine_semantics(self, good_embedding, bad_embedding):
        parts = []
        if self.use_global_guidance:
            batch_size = good_embedding.shape[0]
            if self.use_good_embedding:
                if self.global_good_embedding is None:
                    self.global_good_embedding = nn.Parameter(
                        torch.zeros(good_embedding.shape[-1], device=good_embedding.device, dtype=good_embedding.dtype)
                    )
                parts.append(self.global_good_embedding.unsqueeze(0).expand(batch_size, -1))
            if self.use_bad_embedding:
                if self.global_bad_embedding is None:
                    self.global_bad_embedding = nn.Parameter(
                        torch.zeros(bad_embedding.shape[-1], device=bad_embedding.device, dtype=bad_embedding.dtype)
                    )
                parts.append(self.global_bad_embedding.unsqueeze(0).expand(batch_size, -1))
        else:
            if self.use_good_embedding:
                parts.append(good_embedding)
            if self.use_bad_embedding:
                parts.append(bad_embedding)
        if not parts:
            raise ValueError("At least one semantic embedding must be enabled.")
        return torch.cat(parts, dim=-1)

    def _init_semantic_layers(self, semantic_dim, device):
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.gate_hidden_dim),
        ).to(device)
        if self.use_semantic_uncertainty:
            self.semantic_mu_proj = nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim).to(device)
            self.semantic_logvar_proj = nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim).to(device)
        self.semantic_fusion_proj = nn.Sequential(
            nn.Linear(self.gate_hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        ).to(device)
        self.semantic_gate_proj = nn.Sequential(
            nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.gate_hidden_dim),
        ).to(device)
        if self.use_logit_scale:
            self.decision_scale_proj = nn.Linear(self.gate_hidden_dim, 1).to(device)
            if self.use_affine_calibration:
                self.decision_bias_proj = nn.Linear(self.gate_hidden_dim, 1).to(device)
        if self.use_input_feature_gate:
            self.input_gate_proj = nn.Linear(self.gate_hidden_dim, self.input_dim).to(device)
        if self.use_input_film:
            self.input_film_gamma = nn.Linear(self.gate_hidden_dim, self.input_dim).to(device)
            self.input_film_beta = nn.Linear(self.gate_hidden_dim, self.input_dim).to(device)

    def forward(self, features, good_embedding, bad_embedding):
        semantics = self._combine_semantics(good_embedding, bad_embedding)
        if self.semantic_proj is None:
            self._init_semantic_layers(semantics.shape[-1], features.device)
        semantic_base = self.semantic_proj(semantics)
        semantic_kl = None
        if self.use_semantic_uncertainty:
            semantic_mu = self.semantic_mu_proj(semantic_base)
            semantic_logvar = self.semantic_logvar_proj(semantic_base).clamp(min=-8.0, max=8.0)
            semantic_std = torch.exp(0.5 * semantic_logvar)
            z_semantic = semantic_mu + semantic_std * torch.randn_like(semantic_std)
            semantic_kl = 0.5 * torch.mean(
                torch.sum(
                    torch.exp(semantic_logvar) + semantic_mu.pow(2) - 1.0 - semantic_logvar,
                    dim=-1,
                )
            )
        else:
            semantic_mu = None
            semantic_logvar = None
            z_semantic = semantic_base
        z_semantic_gate = self.semantic_gate_proj(z_semantic)
        z_semantic_fusion = self.semantic_fusion_proj(z_semantic)

        input_gate_values = None
        gated_features = features
        if self.use_input_feature_gate:
            input_gate_values = torch.sigmoid(self.input_gate_proj(z_semantic_gate))
            gated_features = features * input_gate_values
        elif self.use_input_film:
            gamma = 1.0 + torch.tanh(self.input_film_gamma(z_semantic_gate))
            beta = self.input_film_beta(z_semantic_gate)
            gated_features = gamma * features + beta

        z_modulated = None
        if self.use_trainable_phys_branch:
            z_modulated = self.phys_encoder(gated_features)

        z_stress = None
        z_stress_proj = None
        if self.use_stress_branch:
            if self.freeze_stress_encoder:
                with torch.no_grad():
                    z_stress = self.stress_encoder(features)
            else:
                z_stress = self.stress_encoder(features)
            z_stress_proj = self.stress_proj(z_stress)

        if self.use_trainable_phys_branch and self.use_stress_branch:
            if self.fusion_mode == "sample_gate":
                fusion_alpha = torch.sigmoid(
                    self.fusion_gate(torch.cat([z_stress_proj, z_modulated, z_semantic_fusion], dim=-1))
                )
                z_fused = fusion_alpha * z_stress_proj + (1.0 - fusion_alpha) * z_modulated
            elif self.fusion_mode == "scalar":
                weights = torch.softmax(self.fusion_weight_logits, dim=0)
                fusion_alpha = torch.full(
                    (features.shape[0], 1),
                    float(weights[0].detach()),
                    device=features.device,
                    dtype=features.dtype,
                )
                z_fused = fusion_alpha * z_stress_proj + (1.0 - fusion_alpha) * z_modulated
            elif self.fusion_mode == "average":
                fusion_alpha = torch.full(
                    (features.shape[0], 1),
                    0.5,
                    device=features.device,
                    dtype=features.dtype,
                )
                z_fused = 0.5 * (z_stress_proj + z_modulated)
            elif self.fusion_mode == "residual":
                fusion_alpha = torch.sigmoid(
                    self.fusion_gate(torch.cat([z_stress_proj, z_modulated, z_semantic_fusion], dim=-1))
                )
                z_fused = z_stress_proj + fusion_alpha * z_modulated
            elif self.fusion_mode == "mixture":
                fusion_weights = torch.softmax(
                    self.fusion_gate(torch.cat([z_stress_proj, z_modulated, z_semantic_fusion], dim=-1)),
                    dim=-1,
                )
                delta = z_modulated - z_stress_proj
                z_fused = (
                    fusion_weights[:, 0:1] * z_stress_proj
                    + fusion_weights[:, 1:2] * z_modulated
                    + fusion_weights[:, 2:3] * delta
                )
                fusion_alpha = fusion_weights
            elif self.fusion_mode == "concat_mlp":
                delta = z_modulated - z_stress_proj
                z_fused = self.fusion_mlp(
                    torch.cat(
                        [
                            z_stress_proj,
                            z_modulated,
                            z_semantic_fusion,
                            delta,
                            z_stress_proj * z_semantic_fusion,
                            z_modulated * z_semantic_fusion,
                        ],
                        dim=-1,
                    )
                )
                fusion_alpha = None
            elif self.fusion_mode == "attention":
                tokens = torch.stack([z_stress_proj, z_modulated], dim=1)
                query = self.query_proj(z_semantic_fusion).unsqueeze(1)
                keys = self.key_proj(tokens)
                values = self.value_proj(tokens)
                attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(float(self.hidden_dim))
                attn_weights = torch.softmax(attn_scores, dim=-1)
                z_fused = torch.matmul(attn_weights, values).squeeze(1)
                fusion_alpha = attn_weights.squeeze(1)
            elif self.fusion_mode == "cross_gate":
                gates = torch.sigmoid(self.latent_gate_proj(z_semantic_fusion))
                stress_gate, mod_gate = gates.chunk(2, dim=-1)
                z_stress_tilde = z_stress_proj * stress_gate
                z_mod_tilde = z_modulated * mod_gate
                z_fused = 0.5 * (z_stress_tilde + z_mod_tilde)
                fusion_alpha = torch.cat([stress_gate.mean(dim=-1, keepdim=True), mod_gate.mean(dim=-1, keepdim=True)], dim=-1)
            elif self.fusion_mode == "cross_film":
                film_params = self.latent_film_proj(z_semantic_fusion)
                s_gamma, s_beta, m_gamma, m_beta = film_params.chunk(4, dim=-1)
                s_gamma = 1.0 + 0.2 * torch.tanh(s_gamma)
                s_beta = 0.2 * torch.tanh(s_beta)
                m_gamma = 1.0 + 0.2 * torch.tanh(m_gamma)
                m_beta = 0.2 * torch.tanh(m_beta)
                z_stress_tilde = s_gamma * z_stress_proj + s_beta
                z_mod_tilde = m_gamma * z_modulated + m_beta
                z_fused = 0.5 * (z_stress_tilde + z_mod_tilde)
                fusion_alpha = None
            else:
                raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        elif self.use_stress_branch:
            fusion_alpha = None
            z_fused = z_stress_proj
        else:
            fusion_alpha = None
            z_fused = z_modulated

        logits = self.classifier(z_fused)
        decision_alpha = None
        decision_beta = None
        if self.use_logit_scale:
            decision_alpha = F.softplus(self.decision_scale_proj(z_semantic)) + 1e-4
            logits = logits * decision_alpha
            if self.use_affine_calibration:
                decision_beta = self.decision_bias_proj(z_semantic)
                logits = logits + decision_beta
        output = {
            "logits": logits,
            "phys_hidden": z_fused,
            "craving_hidden": z_modulated if z_modulated is not None else z_fused,
            "semantic_hidden": z_semantic,
            "semantic_gate_hidden": z_semantic_gate,
            "semantic_fusion_hidden": z_semantic_fusion,
        }
        if z_stress is not None:
            output["stress_hidden"] = z_stress
            output["stress_hidden_projected"] = z_stress_proj
        if z_modulated is not None:
            output["modulated_hidden"] = z_modulated
        if input_gate_values is not None:
            output["gate_values"] = input_gate_values
            output["input_gate_values"] = input_gate_values
        if self.use_input_film:
            output["input_film_values"] = gated_features
        if fusion_alpha is not None:
            output["fusion_alpha"] = fusion_alpha
        if decision_alpha is not None:
            output["decision_alpha"] = decision_alpha
        if decision_beta is not None:
            output["decision_beta"] = decision_beta
        if self.use_stress_aux_head and z_modulated is not None:
            output["stress_logits"] = self.stress_aux_head(z_modulated)
        if semantic_kl is not None:
            output["semantic_kl"] = semantic_kl
            output["semantic_mu"] = semantic_mu
            output["semantic_logvar"] = semantic_logvar
        return output


def build_model(cfg):
    if cfg.name.lower() == "semantic_guided":
        return SemanticGuidedClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_similarity_head=cfg.use_similarity_head,
            use_logit_scale=cfg.use_logit_scale,
            use_film=cfg.use_film,
            use_sparse_gate=cfg.use_sparse_gate,
        )
    if cfg.name.lower() == "semantic_concat":
        return SemanticConcatClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
        )
    if cfg.name.lower() == "semantic_residual":
        return SemanticResidualClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_logit_scale=cfg.use_logit_scale,
        )
    if cfg.name.lower() == "semantic_token":
        return SemanticTokenAttentionClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
        )
    if cfg.name.lower() == "semantic_input_film":
        return SemanticInputFiLMClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_logit_scale=cfg.use_logit_scale,
        )
    if cfg.name.lower() == "semantic_moe":
        return SemanticMoEClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            num_experts=cfg.num_experts,
            top_k_experts=cfg.top_k_experts,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_logit_scale=cfg.use_logit_scale,
        )
    if cfg.name.lower() == "semantic_hierarchical":
        return SemanticHierarchicalModalityClassifier(
            input_dim=cfg.input_dim,
            feature_columns=cfg.feature_columns,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_logit_scale=cfg.use_logit_scale,
        )
    if cfg.name.lower() == "semantic_hierarchical_feature_gate":
        return SemanticHierarchicalFeatureGateClassifier(
            input_dim=cfg.input_dim,
            feature_columns=cfg.feature_columns,
            hidden_dim=cfg.hidden_dim,
            gate_hidden_dim=cfg.gate_hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_logit_scale=cfg.use_logit_scale,
        )
    if cfg.name.lower() == "stress_mlp":
        return StressEncoderClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )
    if cfg.name.lower() == "semantic_guided_frozen_stress":
        return SemanticGuidedFrozenStressClassifier(
            input_dim=cfg.input_dim,
            stress_encoder_ckpt=cfg.stress_encoder_ckpt,
            hidden_dim=cfg.hidden_dim,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_film=cfg.use_film,
            use_sparse_gate=cfg.use_sparse_gate,
            freeze_stress_encoder=cfg.freeze_stress_encoder,
        )
    if cfg.name.lower() == "semantic_dualpath_frozen_stress":
        return SemanticDualPathFrozenStressClassifier(
            input_dim=cfg.input_dim,
            stress_encoder_ckpt=cfg.stress_encoder_ckpt,
            hidden_dim=cfg.hidden_dim,
            gate_hidden_dim=cfg.gate_hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_classifier_layers=cfg.num_classifier_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            use_good_embedding=cfg.use_good_embedding,
            use_bad_embedding=cfg.use_bad_embedding,
            use_global_guidance=cfg.use_global_guidance,
            use_input_feature_gate=cfg.use_input_feature_gate,
            use_input_film=cfg.use_input_film,
            use_trainable_phys_branch=cfg.use_trainable_phys_branch,
            use_stress_branch=cfg.use_stress_branch,
            fusion_mode=cfg.fusion_mode,
            use_logit_scale=cfg.use_logit_scale,
            use_affine_calibration=cfg.use_affine_calibration,
            use_semantic_uncertainty=cfg.use_semantic_uncertainty,
            use_stress_aux_head=cfg.use_stress_aux_head,
            freeze_stress_encoder=cfg.freeze_stress_encoder,
        )
    return LegacyModelAdapter(
        model_name=cfg.name,
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        baseline_ckpt=cfg.legacy_baseline_ckpt,
    )
