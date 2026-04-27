#!/usr/bin/env python3
"""GWN on NOAA buoys — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol (random search, 12 trials,
log-uniform LR, dataset-pinned training schedule).  Capacity axis: ``nhid``,
GWN's residual / dilation channel width.  Reg axis: ``dropout``.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "noaa-buoy"
base_config = apply_schedule(GWNConfig(), DATASET)

run_example(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":      LogUniform(LR_LOW, LR_HIGH),
        "dropout": Categorical([0.0, 0.1, 0.3, 0.5]),
        "nhid":    Categorical([16, 32, 64]),
    },
)
