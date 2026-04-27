#!/usr/bin/env python3
"""MTGNN on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``conv_channels``.
"""

from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-static"
base_config = apply_schedule(MTGNNConfig(), DATASET)

run_example(
    model_factory=lambda: MTGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":            LogUniform(LR_LOW, LR_HIGH),
        "dropout":       Categorical([0.0, 0.1, 0.3]),
        "conv_channels": Categorical([16, 32]),
    },
)
