#!/usr/bin/env python3
"""Run D2STGNN on the Beijing Air dataset.

Usage:
    python examples/beijing_air_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["beijing-air"],
)

# Override only the config values you want to tune for your run.
config = D2STGNNConfig(
    max_epochs=20,
    batch_size=32,
    lr=0.002,
    early_stop=10,
)

result = runner.run(D2STGNNModel(), config=config)
print(result.summary())
