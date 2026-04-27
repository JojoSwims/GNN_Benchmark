#!/usr/bin/env python3
"""D2STGNN on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``num_hidden``
is capped at 32 — D2STGNN holds an N×N dynamic-graph scores tensor per
layer (N=859 makes the upper end memory-tight).
"""

from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, d2stgnn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(D2STGNNConfig(), DATASET)

run_example(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=d2stgnn_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        num_hidden=Categorical([16, 32]),
    ),
)
