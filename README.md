# SGAMODE Core Implementation

This repository contains the core implementation of SGAMODE, a synergistic graph-aware multi-objective differential evolution framework for high-dimensional feature selection.

The current package is intended for method inspection and peer-review assessment of the core algorithmic components. It includes:

- `sgamode/config.py`: hyperparameter configuration
- `sgamode/data.py`: lightweight CSV dataset loading
- `sgamode/graph.py`: feature-graph construction and loading
- `sgamode/gnn.py`: AEMGNN guidance module
- `sgamode/moea.py`: IFAMODE search engine and SGAMODE wrapper
- `example_data/`: a small demonstration dataset and graph file
- `run_demo.py`: a minimal runnable example

The example data are only for demonstrating the input format and the end-to-end execution flow. They are not part of the formal benchmark data used in the manuscript experiments.

## Quick start

1. Install dependencies from `requirements.txt`.
2. Run:

```bash
python run_demo.py
```

The script loads the small example dataset, fits SGAMODE with a lightweight demo configuration, and reports selected features together with simple held-out accuracy and weighted F1-score.

## Scope

This repository provides the core method implementation. The complete reproducibility package for the full experimental workflow is maintained separately.
