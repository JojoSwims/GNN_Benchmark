#!/usr/bin/env python3
"""GWN on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Uses the
``divvy-bikeshare-static`` registry entry, which restricts to a Loop +
Near North Side + Lincoln Park bbox (N ≈ 515 stations) with a static
haversine graph sparsified at 2 km.  Capacity axis: ``nhid``.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-static"
base_config = apply_schedule(GWNConfig(), DATASET)

run_example(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":      LogUniform(LR_LOW, LR_HIGH),
        "dropout": Categorical([0.0, 0.1, 0.3]),
        "nhid":    Categorical([16, 32]),
    },
)
