#!/usr/bin/env python3
"""Run Graph WaveNet on the EU load dataset (no hyperparameter tuning).

Dataset:
    ENTSO-E European zonal electricity load (hourly, 2023-2024).  49 nodes
    (bidding zones), 1 feature ("load", MW) which is also the target.
    Edges are undirected, unweighted cross-zone interconnections (cost 1
    if connected, 0 otherwise).  Loader auto-downloads both files from
    Google Drive on first use.

Hyperparameters (no grid search):
    Sticking with GWN's defaults — for a 49-node graph with a single
    feature, the default capacity (nhid=32) is plenty and the default
    1e-3 lr / 0.3 dropout converge cleanly.  Only the training-budget
    knobs (max_epochs, batch_size, early_stop) are touched: max_epochs
    is bumped to 50 since we run a single trial instead of a grid.

GWN consumes the interconnection adjacency.

Usage:
    python examples/eu_load_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel

WORKSPACE = "./benchmark_workspace"
DATASET = "eu-load"

print(f"[example] GWN on {DATASET} — workspace={WORKSPACE}")

config = GWNConfig(
    lr=1e-3,
    nhid=32,
    dropout=0.3,
    batch_size=64,
    max_epochs=50,
    early_stop=10,
)

runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(GWNModel(), config=config)
print(result.summary())
