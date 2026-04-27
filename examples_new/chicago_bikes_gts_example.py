#!/usr/bin/env python3
"""GTS on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``rnn_units``.  Reg axis: ``weight_decay``.  Model-critical axis:
``max_diffusion_step`` (DCGRU diffusion order), capped at 2 for memory.
"""

from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-static"
base_config = apply_schedule(GTSConfig(), DATASET)

run_example(
    model_factory=lambda: GTSModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":                 LogUniform(LR_LOW, LR_HIGH),
        "weight_decay":       Categorical([0.0, 1e-5, 1e-4]),
        "rnn_units":          Categorical([32, 64]),
        "max_diffusion_step": Categorical([1, 2]),
    },
)
