from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class GraphSpec:
    relations: list[sparse.csr_matrix]
    relation_names: list[str]
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def minmax_norm(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    return (values - vmin) / (vmax - vmin + eps)


def _row_normalize_with_self_loops(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    n = adj.shape[0]
    mat = adj.tocsr().astype(np.float64)
    mat = mat + sparse.eye(n, dtype=np.float64, format="csr")
    rowsum = np.asarray(mat.sum(axis=1)).ravel()
    rowsum[rowsum == 0.0] = 1.0
    inv = sparse.diags(1.0 / rowsum)
    return inv @ mat


def _safe_int(x: str) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def load_edge_list_graph(
    edge_file: str | Path,
    feature_names: list[str],
) -> sparse.csr_matrix:
    path = Path(edge_file)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Edge list requires at least two columns: {path}")

    cols = [str(c).lower() for c in df.columns]
    if {"source", "target"}.issubset(cols):
        src_col = df.columns[cols.index("source")]
        dst_col = df.columns[cols.index("target")]
    else:
        src_col, dst_col = df.columns[0], df.columns[1]

    if "weight" in cols:
        w_col = df.columns[cols.index("weight")]
        weights = pd.to_numeric(df[w_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        weights = np.ones(len(df), dtype=float)

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    n = len(feature_names)
    rows: list[int] = []
    cols_out: list[int] = []
    vals: list[float] = []

    src_values = df[src_col].astype(str).to_numpy()
    dst_values = df[dst_col].astype(str).to_numpy()

    for s_raw, t_raw, w in zip(src_values, dst_values, weights):
        s_idx = name_to_idx.get(s_raw)
        t_idx = name_to_idx.get(t_raw)
        if s_idx is None:
            s_try = _safe_int(s_raw)
            if s_try is not None and 0 <= s_try < n:
                s_idx = s_try
        if t_idx is None:
            t_try = _safe_int(t_raw)
            if t_try is not None and 0 <= t_try < n:
                t_idx = t_try
        if s_idx is None or t_idx is None or s_idx == t_idx:
            continue
        ww = float(abs(w))
        rows.extend([s_idx, t_idx])
        cols_out.extend([t_idx, s_idx])
        vals.extend([ww, ww])

    if not rows:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    mat = sparse.coo_matrix((vals, (rows, cols_out)), shape=(n, n), dtype=np.float64)
    mat.sum_duplicates()
    return mat.tocsr()


def build_correlation_graph(
    X: np.ndarray,
    threshold: float = 0.5,
    block_size: int = 256,
) -> sparse.csr_matrix:
    n_samples, n_features = X.shape
    if n_features <= 1:
        return sparse.csr_matrix((n_features, n_features), dtype=np.float64)

    Xc = X.astype(np.float64, copy=True)
    Xc -= Xc.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    Z = (Xc / std).T.astype(np.float32, copy=False)  # [n_features, n_samples]
    denom = float(max(1, n_samples - 1))

    rows: list[int] = []
    cols_out: list[int] = []
    vals: list[float] = []

    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)
        block = Z[start:end]  # [b, n_samples]
        corr = (block @ Z.T) / denom  # [b, n_features]
        abs_corr = np.abs(corr)
        for local_i in range(end - start):
            i = start + local_i
            indices = np.where(abs_corr[local_i] >= threshold)[0]
            for j in indices:
                if j <= i:
                    continue
                v = float(abs_corr[local_i, j])
                rows.extend([i, j])
                cols_out.extend([j, i])
                vals.extend([v, v])

    if not rows:
        return sparse.csr_matrix((n_features, n_features), dtype=np.float64)
    mat = sparse.coo_matrix((vals, (rows, cols_out)), shape=(n_features, n_features), dtype=np.float64)
    mat.sum_duplicates()
    return mat.tocsr()


def pagerank_centrality(
    adj: sparse.csr_matrix,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    P = _row_normalize_with_self_loops(adj)
    c = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        nxt = teleport + damping * (P.T @ c)
        if np.linalg.norm(nxt - c, ord=1) < tol:
            c = nxt
            break
        c = nxt
    return c


def build_graph_spec(
    X: np.ndarray,
    feature_names: list[str],
    external_graph_files: Iterable[str | Path] | None = None,
    corr_threshold: float = 0.5,
    corr_block_size: int = 256,
) -> GraphSpec:
    rels: list[sparse.csr_matrix] = []
    names: list[str] = []

    if external_graph_files:
        for path in external_graph_files:
            rel = load_edge_list_graph(path, feature_names=feature_names)
            rels.append(rel)
            names.append(Path(path).stem)
    else:
        rels.append(build_correlation_graph(X, threshold=corr_threshold, block_size=corr_block_size))
        names.append("corr_threshold")

    rels = [r.tocsr() for r in rels]
    return GraphSpec(relations=rels, relation_names=names, feature_names=feature_names)


def normalized_relations(spec: GraphSpec) -> list[sparse.csr_matrix]:
    return [_row_normalize_with_self_loops(rel) for rel in spec.relations]
