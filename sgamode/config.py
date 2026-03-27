from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SGAMODEConfig:
    pop_size: int = 100
    max_generations: int = 100

    p_best_rate: float = 0.10
    jade_c: float = 0.10
    mu_f_init: float = 0.50
    mu_cr_init: float = 0.90
    sigma_f: float = 0.10
    sigma_cr: float = 0.10

    feedback_interval: int = 10
    feedback_eta: float = 0.10

    gamma_struct: float = 1.0
    gamma_sem: float = 1.0

    pool_rate: float = 0.5
    gnn_layers: int = 2
    gnn_heads: int = 2
    gnn_dropout: float = 0.2
    aemgnn_hidden_dim: int = 128
    embedding_dim: int = 128
    aemgnn_init_epochs: int = 80
    aemgnn_feedback_epochs: int = 12
    aemgnn_lr: float = 1e-3
    aemgnn_weight_decay: float = 1e-4
    aemgnn_neg_ratio: float = 1.0
    aemgnn_neg_loss_weight: float = 1.0
    aemgnn_max_pos_edges: int = 250000
    aemgnn_max_neg_edges: int = 250000
    gnn_grad_clip: float = 5.0
    gnn_device: str = "auto"
    gnn_use_amp: bool = True
    gnn_train_verbose: bool = False

    knn_candidates: Sequence[int] = field(default_factory=lambda: (3, 5, 7))
    inner_cv_folds: int = 3

    corr_threshold: float = 0.5
    corr_block_size: int = 256

    decode_threshold: float = 0.75
    decode_with_sigmoid: bool = True
    decode_sigmoid_scale: float = 10.0
    selection_strategy: str = "knee"
    sparsity_error_tolerance: float = 0.02

    max_semantic_features: int = 120
    sparsity_mutation_prob: float = 0.20
    random_state: int | None = 2024
    verbose: bool = False
