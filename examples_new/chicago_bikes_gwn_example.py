#!/usr/bin/env python3
"""GWN on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  Uses the
``divvy-bikeshare-static`` registry entry (N ≈ 515 stations); ``nhid``
is capped at 32 because GWN's skip/end channels scale as ``nhid * 8 /
16``.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, gwn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-static"
base_config = apply_schedule(GWNConfig(), DATASET)

run_example(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=gwn_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        nhid=Categorical([16, 32]),
    ),
)
