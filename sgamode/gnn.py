from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn

from .config import SGAMODEConfig
from .graph import GraphSpec, minmax_norm, normalized_relations, pagerank_centrality


@dataclass
class _EdgeTensor:
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor


@dataclass
class GuidanceState:
    embeddings: np.ndarray
    centrality: np.ndarray
    pool_indices: np.ndarray
    pool_importance: np.ndarray
    relation_attention: np.ndarray
    relation_node_masks: list[np.ndarray]


class _AEMGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_rel: int, n_heads: int, dropout: float):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.rel_proj = nn.ModuleList(
            [nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_heads)]) for _ in range(n_rel)]
        )
        self.node_query = nn.Parameter(torch.empty(n_rel, n_heads, out_dim))
        self.rel_vector = nn.Parameter(torch.empty(n_rel, out_dim))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for rel_heads in self.rel_proj:
            for linear in rel_heads:
                nn.init.xavier_uniform_(linear.weight)
        nn.init.xavier_uniform_(self.node_query)
        nn.init.xavier_uniform_(self.rel_vector)

    def forward(
        self,
        h: torch.Tensor,
        relation_adj: list[torch.Tensor],
        relation_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_outputs: list[torch.Tensor] = []
        for ridx, adj in enumerate(relation_adj):
            head_outputs: list[torch.Tensor] = []
            for hidx in range(self.n_heads):
                hp = self.rel_proj[ridx][hidx](h)
                gate = torch.sigmoid((hp * self.node_query[ridx, hidx]).sum(dim=1, keepdim=True))
                msg = torch.sparse.mm(adj, hp * gate)
                head_outputs.append(msg)
            relation_outputs.append(torch.stack(head_outputs, dim=0).mean(dim=0))

        rel_stack = torch.stack(relation_outputs, dim=0)  # [R, N, D]
        rel_logits = (rel_stack * self.rel_vector.unsqueeze(1)).sum(dim=2).transpose(0, 1)  # [N, R]
        rel_logits = rel_logits + relation_bias.unsqueeze(0)
        rel_attn_node = torch.softmax(rel_logits, dim=1)
        h_next = (rel_attn_node.transpose(0, 1).unsqueeze(2) * rel_stack).sum(dim=0)
        h_next = self.dropout(self.act(h_next))
        rel_attn_global = rel_attn_node.mean(dim=0)
        return h_next, rel_attn_global


class _AEMGNNEncoder(nn.Module):
    def __init__(self, cfg: SGAMODEConfig, in_dim: int, n_rel: int):
        super().__init__()
        hidden_dim = max(8, int(cfg.aemgnn_hidden_dim))
        embedding_dim = max(8, int(cfg.embedding_dim))
        dims = [in_dim]
        if cfg.gnn_layers <= 1:
            dims.append(embedding_dim)
        else:
            dims.extend([hidden_dim] * (cfg.gnn_layers - 1))
            dims.append(embedding_dim)

        self.layers = nn.ModuleList(
            [
                _AEMGNNLayer(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    n_rel=n_rel,
                    n_heads=max(1, int(cfg.gnn_heads)),
                    dropout=float(cfg.gnn_dropout),
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        relation_adj: list[torch.Tensor],
        relation_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        rel_attn = torch.full((len(relation_adj),), 1.0 / max(1, len(relation_adj)), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            h, rel_attn = layer(h, relation_adj, relation_bias)
        return h, rel_attn


class AEMGNN:
    """Trainable AEMGNN with bidirectional MODE feedback."""

    def __init__(self, config: SGAMODEConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.device = self._resolve_device(config.gnn_device)

        self.relation_bias: np.ndarray | None = None
        self.model: _AEMGNNEncoder | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler: torch.cuda.amp.GradScaler | None = None
        self.runtime_signature: tuple[int, int, tuple[tuple[int, int, int], ...]] | None = None
        self.node_features_t: torch.Tensor | None = None
        self.rel_norm_mats: list[sparse.csr_matrix] = []
        self.relation_adj_t: list[torch.Tensor] = []
        self.agg_target_edges_t: _EdgeTensor | None = None
        self.relation_node_masks: list[np.ndarray] = []
        self._oom_fallback_used = False
        self._set_seed(config.random_state)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self, spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _graph_signature(self, X: np.ndarray, graph: GraphSpec) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
        rel_sig = tuple((rel.shape[0], rel.shape[1], int(rel.nnz)) for rel in graph.relations)
        return int(X.shape[0]), int(X.shape[1]), rel_sig

    def _csr_to_sparse_tensor(self, mat: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
        coo = mat.tocoo()
        if coo.nnz == 0:
            idx = torch.zeros((2, 0), dtype=torch.long, device=device)
            vals = torch.zeros((0,), dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(idx, vals, size=coo.shape, device=device).coalesce()
        idx_np = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
        idx = torch.from_numpy(idx_np).to(device)
        vals = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(device)
        return torch.sparse_coo_tensor(idx, vals, size=coo.shape, device=device).coalesce()

    def _csr_to_edge_tensor(self, mat: sparse.csr_matrix, device: torch.device) -> _EdgeTensor:
        coo = mat.tocoo()
        src = torch.from_numpy(coo.row.astype(np.int64, copy=False)).to(device)
        dst = torch.from_numpy(coo.col.astype(np.int64, copy=False)).to(device)
        w = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(device)
        return _EdgeTensor(src=src, dst=dst, weight=w)

    def _build_runtime(self, X: np.ndarray, graph: GraphSpec, force_rebuild_model: bool = False) -> None:
        signature = self._graph_signature(X, graph)
        if self.runtime_signature == signature and not force_rebuild_model:
            return

        node_features = X.T.astype(np.float32, copy=True)
        node_features -= node_features.mean(axis=1, keepdims=True)
        node_std = node_features.std(axis=1, ddof=1, keepdims=True)
        node_std[node_std == 0] = 1.0
        node_features /= node_std
        self.node_features_t = torch.from_numpy(node_features).to(self.device)

        self.rel_norm_mats = normalized_relations(graph)
        self.relation_adj_t = [self._csr_to_sparse_tensor(rel, device=self.device) for rel in self.rel_norm_mats]

        agg = sparse.csr_matrix(self.rel_norm_mats[0].shape, dtype=np.float64)
        for rel in self.rel_norm_mats:
            agg = agg + rel
        agg = agg / max(1, len(self.rel_norm_mats))
        self.agg_target_edges_t = self._csr_to_edge_tensor(agg, device=self.device)

        self.relation_node_masks = []
        for rel in graph.relations:
            deg = np.asarray(rel.sum(axis=1)).ravel() + np.asarray(rel.sum(axis=0)).ravel()
            self.relation_node_masks.append(deg > 0)

        n_rel = len(self.rel_norm_mats)
        if self.relation_bias is None or len(self.relation_bias) != n_rel:
            self.relation_bias = np.zeros(n_rel, dtype=np.float64)
            force_rebuild_model = True

        if self.model is None or force_rebuild_model:
            in_dim = int(self.node_features_t.shape[1])
            self.model = _AEMGNNEncoder(self.config, in_dim=in_dim, n_rel=n_rel).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.config.aemgnn_lr),
                weight_decay=float(self.config.aemgnn_weight_decay),
            )
            # Sparse-dense kernels in this module run in float32 for stability/compatibility.
            amp_enabled = False
            self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=amp_enabled)

        self.runtime_signature = signature

    def _switch_to_cpu(self) -> None:
        if self.device.type == "cpu":
            return
        self.device = torch.device("cpu")
        if self.node_features_t is not None:
            self.node_features_t = self.node_features_t.to(self.device)
        self.relation_adj_t = [adj.to(self.device) for adj in self.relation_adj_t]
        if self.agg_target_edges_t is not None:
            self.agg_target_edges_t = _EdgeTensor(
                src=self.agg_target_edges_t.src.to(self.device),
                dst=self.agg_target_edges_t.dst.to(self.device),
                weight=self.agg_target_edges_t.weight.to(self.device),
            )
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.config.aemgnn_lr),
                weight_decay=float(self.config.aemgnn_weight_decay),
            )
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=False)

    def _sample_edges(self, edge: _EdgeTensor, max_edges: int) -> _EdgeTensor:
        if edge.src.numel() <= max_edges:
            return edge
        idx = torch.randperm(edge.src.numel(), device=edge.src.device)[:max_edges]
        return _EdgeTensor(src=edge.src[idx], dst=edge.dst[idx], weight=edge.weight[idx])

    def _reconstruction_loss(self, emb: torch.Tensor) -> torch.Tensor:
        if self.agg_target_edges_t is None:
            return emb.square().mean() * 0.0

        pos = self._sample_edges(self.agg_target_edges_t, int(self.config.aemgnn_max_pos_edges))
        if pos.src.numel() == 0:
            return emb.square().mean() * 0.0

        pos_pred = torch.tanh((emb[pos.src] * emb[pos.dst]).sum(dim=1))
        pos_loss = torch.mean((pos_pred - pos.weight) ** 2)

        neg_count = min(
            int(max(1, pos.src.numel()) * max(0.0, float(self.config.aemgnn_neg_ratio))),
            int(self.config.aemgnn_max_neg_edges),
        )
        if neg_count <= 0:
            return pos_loss

        n_nodes = emb.shape[0]
        neg_src = torch.randint(0, n_nodes, (neg_count,), device=emb.device)
        neg_dst = torch.randint(0, n_nodes, (neg_count,), device=emb.device)
        keep = neg_src != neg_dst
        if torch.any(keep):
            neg_src = neg_src[keep]
            neg_dst = neg_dst[keep]
            neg_pred = torch.tanh((emb[neg_src] * emb[neg_dst]).sum(dim=1))
            neg_loss = torch.mean(neg_pred.square())
        else:
            neg_loss = pos_loss * 0.0
        return pos_loss + float(self.config.aemgnn_neg_loss_weight) * neg_loss

    def _train_epochs(self, epochs: int) -> None:
        if epochs <= 0:
            return
        if self.model is None or self.optimizer is None or self.node_features_t is None:
            return

        amp_enabled = False
        try:
            self.model.train()
            for epoch in range(1, epochs + 1):
                self.optimizer.zero_grad(set_to_none=True)
                relation_bias_t = torch.from_numpy(self.relation_bias.astype(np.float32, copy=False)).to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    emb, _ = self.model(self.node_features_t, self.relation_adj_t, relation_bias_t)
                    loss = self._reconstruction_loss(emb)

                if amp_enabled and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.config.gnn_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.gnn_grad_clip))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.gnn_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.gnn_grad_clip))
                    self.optimizer.step()

                if self.config.gnn_train_verbose and (epoch == 1 or epoch == epochs or epoch % 10 == 0):
                    print(f"[AEMGNN] epoch={epoch:03d}/{epochs}, recon_loss={float(loss.detach().cpu().item()):.6f}")
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if oom and self.device.type == "cuda" and not self._oom_fallback_used:
                self._oom_fallback_used = True
                torch.cuda.empty_cache()
                self._switch_to_cpu()
                self._train_epochs(epochs)
                return
            raise

    def _inference_state(self) -> GuidanceState:
        if self.model is None or self.node_features_t is None:
            raise RuntimeError("AEMGNN runtime not initialized.")
        self.model.eval()
        with torch.no_grad():
            relation_bias_t = torch.from_numpy(self.relation_bias.astype(np.float32, copy=False)).to(self.device)
            emb_t, rel_attn_t = self.model(self.node_features_t, self.relation_adj_t, relation_bias_t)

        embeddings = emb_t.detach().cpu().numpy().astype(np.float64, copy=False)
        rel_attn = rel_attn_t.detach().cpu().numpy().astype(np.float64, copy=False)
        rel_attn = rel_attn / (np.sum(rel_attn) + 1e-12)

        agg_adj = sparse.csr_matrix(self.rel_norm_mats[0].shape, dtype=np.float64)
        for w, rel in zip(rel_attn, self.rel_norm_mats):
            agg_adj = agg_adj + float(w) * rel

        centrality = pagerank_centrality(agg_adj)
        spagp_repr = agg_adj @ embeddings
        spagp_score = np.tanh(np.linalg.norm(spagp_repr, axis=1))
        importance = 0.7 * minmax_norm(spagp_score) + 0.3 * minmax_norm(centrality)

        n_nodes = importance.shape[0]
        k_pool = max(1, int(np.ceil(self.config.pool_rate * n_nodes)))
        pool_indices = np.argsort(-importance)[:k_pool]
        pool_mask = np.zeros(n_nodes, dtype=bool)
        pool_mask[pool_indices] = True
        pool_importance = importance.copy()
        pool_importance[~pool_mask] *= 0.1

        return GuidanceState(
            embeddings=embeddings,
            centrality=centrality,
            pool_indices=pool_indices,
            pool_importance=pool_importance,
            relation_attention=rel_attn,
            relation_node_masks=self.relation_node_masks,
        )

    def initialize(self, X: np.ndarray, graph: GraphSpec) -> GuidanceState:
        self._build_runtime(X, graph, force_rebuild_model=True)
        self._train_epochs(int(self.config.aemgnn_init_epochs))
        return self._inference_state()

    def forward(self, X: np.ndarray, graph: GraphSpec) -> GuidanceState:
        self._build_runtime(X, graph, force_rebuild_model=False)
        return self._inference_state()

    def refine(self, X: np.ndarray, graph: GraphSpec) -> GuidanceState:
        self._build_runtime(X, graph, force_rebuild_model=False)
        self._train_epochs(int(self.config.aemgnn_feedback_epochs))
        return self._inference_state()

    def feedback_update(self, feature_freq: np.ndarray, relation_node_masks: list[np.ndarray]) -> None:
        if self.relation_bias is None:
            return
        global_avg = float(np.mean(feature_freq))
        for ridx, mask in enumerate(relation_node_masks):
            if mask.sum() == 0:
                continue
            rel_avg = float(np.mean(feature_freq[mask]))
            self.relation_bias[ridx] += self.config.feedback_eta * (rel_avg - global_avg)

