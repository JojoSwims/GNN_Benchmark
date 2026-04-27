#!/usr/bin/env python3
"""MTGNN on NOAA buoys — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``conv_channels`` (the TCN/GCN width).  Reg axis: ``dropout``.
Model-critical axis: ``subgraph_size`` (top-K of the learned graph;
bounded by N).
"""

from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "noaa-buoy"
base_config = apply_schedule(MTGNNConfig(), DATASET)

run_example(
    model_factory=lambda: MTGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":             LogUniform(LR_LOW, LR_HIGH),
        "dropout":        Categorical([0.0, 0.1, 0.3, 0.5]),
        "conv_channels":  Categorical([16, 32, 64]),
        "subgraph_size":  Categorical([5, 10, 20]),
    },
)
