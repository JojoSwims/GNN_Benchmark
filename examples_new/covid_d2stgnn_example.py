#!/usr/bin/env python3
"""D2STGNN on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``num_hidden``, capped at 32 because D2STGNN's per-layer N×N dynamic-graph
scores tensor is ~40 MB at fp32 on N=3212 (gradients double that).
"""

from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(D2STGNNConfig(), DATASET)

run_example(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":         LogUniform(LR_LOW, LR_HIGH),
        "dropout":    Categorical([0.0, 0.1, 0.2]),
        "num_hidden": Categorical([16, 32]),
    },
)
