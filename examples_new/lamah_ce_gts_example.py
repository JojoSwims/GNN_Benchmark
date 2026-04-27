#!/usr/bin/env python3
"""GTS on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``rnn_units``, capped at 64 because GTS materialises an N×N learned graph
and DCGRU activations are (B, T, N, rnn_units).
"""

from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(GTSConfig(), DATASET)

run_example(
    model_factory=lambda: GTSModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":           LogUniform(LR_LOW, LR_HIGH),
        "weight_decay": Categorical([0.0, 1e-5, 1e-4]),
        "rnn_units":    Categorical([32, 64]),
    },
)
