#!/usr/bin/env python3
"""MTGNN on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``conv_channels``
is capped at 32 for the 3212-node graph.
"""

from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, mtgnn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(MTGNNConfig(), DATASET)

run_example(
    model_factory=lambda: MTGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=mtgnn_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        conv_channels=Categorical([16, 32]),
    ),
)
