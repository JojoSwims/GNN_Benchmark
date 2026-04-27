#!/usr/bin/env python3
"""GTS on NOAA buoys — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.
"""

from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import gts_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "noaa-buoy"
base_config = apply_schedule(GTSConfig(), DATASET)

run_example(
    model_factory=lambda: GTSModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=gts_search_space(lr_low=LR_LOW, lr_high=LR_HIGH),
)
