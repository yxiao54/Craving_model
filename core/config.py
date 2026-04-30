from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    parquet_path: str = "./oudlab_windows_clean.parquet"
    embedding_path: str = "./user_embeddings_small.pickle"
    label_name: str = "craving"
    batch_size: int = 64
    num_workers: int = 0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    split_json: Optional[str] = None
    fold_key: Optional[str] = None
    environment_key: str = "cohort"
    feature_drop_nan_threshold: float = 1.0
    use_clean_numeric_only: bool = True
    normalization: str = "subject_zscore"


@dataclass
class ModelConfig:
    name: str = "semantic_guided"
    input_dim: Optional[int] = None
    feature_columns: Optional[list[str]] = None
    hidden_dim: int = 128
    gate_hidden_dim: int = 64
    num_encoder_layers: int = 2
    num_classifier_layers: int = 2
    num_experts: int = 4
    top_k_experts: int = 0
    dropout: float = 0.2
    num_classes: int = 2
    use_good_embedding: bool = True
    use_bad_embedding: bool = True
    use_global_guidance: bool = False
    use_similarity_head: bool = True
    use_logit_scale: bool = True
    use_film: bool = True
    use_sparse_gate: bool = True
    use_input_feature_gate: bool = True
    use_input_film: bool = False
    use_trainable_phys_branch: bool = True
    use_stress_branch: bool = True
    fusion_mode: str = "sample_gate"
    use_affine_calibration: bool = False
    use_semantic_uncertainty: bool = False
    use_stress_aux_head: bool = False
    stress_encoder_ckpt: Optional[str] = None
    freeze_stress_encoder: bool = True
    legacy_baseline_ckpt: Optional[str] = None


@dataclass
class ObjectiveConfig:
    name: str = "erm"
    irm_lambda: float = 1.0
    vrex_lambda: float = 1.0
    dro_eta: float = 0.1
    aux_ce_weight: float = 0.0
    alignment_weight: float = 0.1
    sparse_gate_weight: float = 1e-3
    modality_gate_sparsity_weight: float = 1e-3
    feature_gate_sparsity_weight: float = 1e-3
    label_smoothing: float = 0.0
    moe_load_balance_weight: float = 1e-2
    stress_aux_weight: float = 0.0
    orthogonality_weight: float = 0.0
    semantic_kl_weight: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    optimizer_name: str = "adam"
    scheduler_name: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    seed: int = 2026
    device: str = "auto"
    output_dir: str = "./runs/oud_next"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
