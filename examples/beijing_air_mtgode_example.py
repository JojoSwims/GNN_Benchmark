#!/usr/bin/env python3
"""Run MTGODE on the Beijing Air dataset with a default config.

A single ``fit()`` + ``predict()`` over the standard benchmark pipeline.
For a tuning workflow, see e.g. ``examples/beijing_air_mtgnn_example.py``.

Usage:
    python examples/beijing_air_mtgode_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGODEConfig, MTGODEModel

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

config = MTGODEConfig(
    max_epochs=20,
    batch_size=32,
    early_stop=5,
)

runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(MTGODEModel(), config=config)
print(result.summary())
