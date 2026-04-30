#!/usr/bin/env python3
"""MLP baseline single-run benchmark (no hyperparameter tuning).

This baseline counterpart to the GNN scripts in ``examples_new/`` keeps the
same dataset-specific training schedule from ``DATASET_SCHEDULE`` but runs only
one final evaluation with a hand-picked strong MLP configuration.
"""

import sys
from pathlib import Path

# Reuse the new-examples schedule helper without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples_new"))

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MLPMultivariateConfig, MLPMultivariateModel

from _shared import WORKSPACE, apply_schedule

DATASET = "divvy-bikeshare-static"
# Chosen as a robust single-run setting from prior MLP tuning trends.
base_config = apply_schedule(
    MLPMultivariateConfig(
        lr=2e-3,
        hidden_size=1024,
        num_layers=3,
        dropout=0.2,
        weight_decay=5e-5,
        seed=0,
    ),
    DATASET,
)

print(f"[example] MLPMultivariate on {DATASET} — single run (no tuning)")
runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(MLPMultivariateModel(), config=base_config)
print(result.summary())
