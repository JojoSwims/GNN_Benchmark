#!/usr/bin/env python3
"""Run Graph WaveNet on the Beijing Air dataset.

Usage:
    python examples/beijing_air_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["beijing-air"],
)

# Override only the config values you want to tune for your run.
config = GWNConfig(
    max_epochs=20,
    batch_size=32,
    lr=0.001,
    early_stop=10,
)

result = runner.run(GWNModel(), config=config)
print(result.summary())
