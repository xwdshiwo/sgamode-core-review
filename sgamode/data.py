from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    label_name: str
    source_file: Path
    notes: list[str]


def _is_id_like_column(values: np.ndarray) -> bool:
    if values.ndim != 1:
        return False
    if len(values) < 10:
        return False
    if not np.issubdtype(values.dtype, np.number):
        return False
    finite = values[np.isfinite(values)]
    if len(finite) != len(values):
        return False
    rounded = np.round(values)
    if not np.all(np.abs(values - rounded) < 1e-8):
        return False
    uniq = np.unique(values)
    if len(uniq) < len(values) * 0.98:
        return False
    diffs = np.diff(values)
    monotonic_inc = np.all(diffs > 0)
    monotonic_dec = np.all(diffs < 0)
    return monotonic_inc or monotonic_dec


def load_csv_dataset(
    file_path: str | Path,
    dataset_name: str | None = None,
    auto_drop_id_column: bool = False,
) -> DatasetBundle:
    path = Path(file_path)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least one feature and one label column: {path}")

    notes: list[str] = []
    feature_df = df.iloc[:, :-1].copy()
    label_series = df.iloc[:, -1].copy()

    if auto_drop_id_column and feature_df.shape[1] > 1:
        first_col = pd.to_numeric(feature_df.iloc[:, 0], errors="coerce").to_numpy()
        if np.isfinite(first_col).all() and _is_id_like_column(first_col):
            notes.append("Dropped first feature column by ID-like heuristic.")
            feature_df = feature_df.iloc[:, 1:]

    X = feature_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        notes.append("Filled NaN feature values with column means.")

    y_raw = label_series.astype(str).to_numpy()
    y = LabelEncoder().fit_transform(y_raw)

    name = dataset_name or path.stem
    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        feature_names=[str(c) for c in feature_df.columns],
        label_name=str(df.columns[-1]),
        source_file=path,
        notes=notes,
    )


def find_dataset_file(dataset_name: str, roots: Iterable[str | Path]) -> Path | None:
    candidates: list[Path] = []
    aliases = {dataset_name, dataset_name.replace(".", "-"), dataset_name.replace("-", ".")}
    if dataset_name == "molec-biol-prom":
        aliases.add("molec-biol-promoter")
    if dataset_name == "molec-biol-promoter":
        aliases.add("molec-biol-prom")

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for alias in aliases:
            for match in root_path.rglob(f"{alias}.csv"):
                candidates.append(match)
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]
