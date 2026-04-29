#!/usr/bin/env python3
"""MLP on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

Baseline counterpart to the GNN scripts in ``examples_new/`` — runs the
identical protocol (18 random trials, seed 0, dataset training schedule
from ``DATASET_SCHEDULE``) on ``MLPMultivariateModel`` so the resulting
test-set numbers are directly comparable.  See
``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.
"""

import sys
from pathlib import Path

# Reuse the new-examples protocol helper without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples_new"))

from gnn_benchmark.models import MLPMultivariateConfig, MLPMultivariateModel
from gnn_benchmark.tuning import mlp_multivariate_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(MLPMultivariateConfig(), DATASET)

run_example(
    model_factory=lambda: MLPMultivariateModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=mlp_multivariate_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
    ),
)
