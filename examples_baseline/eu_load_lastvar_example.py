#!/usr/bin/env python3
"""LastVar baseline single-run benchmark (no hyperparameter tuning).

This baseline counterpart to the GNN scripts in ``examples_new/`` keeps the
same dataset-specific training schedule from ``DATASET_SCHEDULE`` and runs a
single deterministic LastVar/LastValue evaluation.
"""

import sys
from pathlib import Path

# Reuse the new-examples schedule helper without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples_new"))

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import LastValueModel

from _shared import WORKSPACE

DATASET = "eu-load"

print(f"[example] LastVar (LastValue) on {DATASET} — single run (no tuning)")
runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(LastValueModel(), config=None)
print(result.summary())
