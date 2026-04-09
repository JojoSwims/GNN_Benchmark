#!/usr/bin/env python3
"""Run STAEFormer on the Beijing Air dataset.

Usage:
    python examples/beijing_air_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["beijing-air"],
)

# Override only the config values you want to tune for your run.
config = STAEFormerConfig(
    max_epochs=20,
    batch_size=16,
    lr=0.001,
    early_stop=10,
)

result = runner.run(STAEFormerModel(), config=config)
print(result.summary())
