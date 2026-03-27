from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from sgamode import SGAMODE, SGAMODEConfig
from sgamode.data import load_csv_dataset


def main() -> None:
    root = Path(__file__).resolve().parent
    data_file = root / "example_data" / "example_dataset.csv"
    graph_file = root / "example_data" / "example_graph_edges.csv"

    bundle = load_csv_dataset(data_file, dataset_name="example_small_biomarker_like")
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.25,
        random_state=2024,
        stratify=bundle.y,
    )

    cfg = SGAMODEConfig(
        pop_size=24,
        max_generations=8,
        feedback_interval=4,
        feedback_eta=0.10,
        gamma_struct=1.0,
        gamma_sem=1.0,
        gnn_layers=2,
        gnn_heads=2,
        aemgnn_hidden_dim=32,
        embedding_dim=32,
        aemgnn_init_epochs=10,
        aemgnn_feedback_epochs=3,
        gnn_device="cpu",
        verbose=False,
        gnn_train_verbose=False,
        random_state=2024,
    )

    model = SGAMODE(config=cfg)
    model.fit(
        X_train,
        y_train,
        feature_names=bundle.feature_names,
        external_graph_files=[str(graph_file)],
    )
    metrics = model.evaluate_test_split(X_train, y_train, X_test, y_test)
    selected_idx = model.get_selected_feature_indices().tolist()
    selected_features = [bundle.feature_names[i] for i in selected_idx]

    summary = {
        "dataset": bundle.name,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_total": int(X_train.shape[1]),
        "selected_feature_indices": selected_idx,
        "selected_feature_names": selected_features,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
