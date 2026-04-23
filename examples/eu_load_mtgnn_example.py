#!/usr/bin/env python3
"""Run MTGNN on the EU load dataset (no hyperparameter tuning).

Dataset:
    ENTSO-E European zonal electricity load (hourly, 2023-2024).  49 nodes
    (bidding zones), 1 feature ("load", MW) which is also the target.
    Edges are undirected, unweighted cross-zone interconnections.  Loader
    auto-downloads both files from Google Drive on first use.

Hyperparameters (no grid search):
    Sticking with MTGNN's defaults — for a 49-node graph with a single
    feature, the default conv_channels=32 is sufficient and 1e-3 lr /
    0.3 dropout converge cleanly.  Only the training-budget knobs
    (max_epochs, batch_size, early_stop) are touched: max_epochs is
    bumped to 50 since we run a single trial instead of a grid.

Note:
    MTGNN learns its own graph and ignores the supplied adjacency.

Usage:
    python examples/eu_load_mtgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGNNConfig, MTGNNModel

WORKSPACE = "./benchmark_workspace"
DATASET = "eu-load"

print(f"[example] MTGNN on {DATASET} — workspace={WORKSPACE}")

config = MTGNNConfig(
    lr=1e-3,
    conv_channels=32,
    dropout=0.3,
    batch_size=64,
    max_epochs=50,
    early_stop=10,
)

runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(MTGNNModel(), config=config)
print(result.summary())
