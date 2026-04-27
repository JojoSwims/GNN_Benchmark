#!/usr/bin/env python3
"""MTGODE on NOAA buoys — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.
"""

from gnn_benchmark.models import MTGODEConfig, MTGODEModel
from gnn_benchmark.tuning import mtgode_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "noaa-buoy"
base_config = apply_schedule(MTGODEConfig(), DATASET)

run_example(
    model_factory=lambda: MTGODEModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=mtgode_search_space(lr_low=LR_LOW, lr_high=LR_HIGH),
)
