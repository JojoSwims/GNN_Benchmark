#!/usr/bin/env python3
"""D2STGNN on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``num_hidden``, capped at 32 because D2STGNN holds an N×N dynamic-graph
scores tensor per layer (N=859 makes the upper end memory-tight).
"""

from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
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
