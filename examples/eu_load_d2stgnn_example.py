#!/usr/bin/env python3
"""Run D2STGNN on the EU load dataset (no hyperparameter tuning).

Dataset:
    ENTSO-E European zonal electricity load (hourly, 2023-2024).  49 nodes
    (bidding zones), 1 feature ("load", MW) which is also the target.
    Edges are undirected, unweighted cross-zone interconnections.  Loader
    auto-downloads both files from Google Drive on first use.

Hyperparameters (no grid search):
    Sticking with D2STGNN's defaults — for a 49-node graph with a single
    feature, the default num_hidden=32 is sufficient.  D2STGNN's default
    learning rate (2e-3) is higher than the other models and works well
    with the MultiStep LR scheduler that's enabled by default.  Only the
    training-budget knobs (max_epochs, batch_size, early_stop) are
    touched: max_epochs is bumped to 50 since we run a single trial
    instead of a grid.

D2STGNN consumes the interconnection adjacency via doubletransition.

Usage:
    python examples/eu_load_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel

WORKSPACE = "./benchmark_workspace"
DATASET = "eu-load"

print(f"[example] D2STGNN on {DATASET} — workspace={WORKSPACE}")

config = D2STGNNConfig(
    lr=2e-3,
    num_hidden=32,
    dropout=0.1,
    batch_size=32,
    max_epochs=50,
    early_stop=10,
)

runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(D2STGNNModel(), config=config)
print(result.summary())
