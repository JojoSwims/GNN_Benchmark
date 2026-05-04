#!/usr/bin/env python3
"""MTGODE on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``conv_channels``
is capped at 32 for the 859-node graph.
"""

from gnn_benchmark.models import MTGODEConfig, MTGODEModel
from gnn_benchmark.tuning import Categorical, mtgode_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce"
base_config = apply_schedule(MTGODEConfig(), DATASET)

run_example(
    model_factory=lambda: MTGODEModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=mtgode_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        conv_channels=Categorical([16, 32]),
    ),
)
