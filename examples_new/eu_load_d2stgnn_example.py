#!/usr/bin/env python3
"""D2STGNN on EU electricity load — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.
"""

from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import d2stgnn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "eu-load"
base_config = apply_schedule(D2STGNNConfig(), DATASET)

run_example(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=d2stgnn_search_space(lr_low=LR_LOW, lr_high=LR_HIGH),
)
