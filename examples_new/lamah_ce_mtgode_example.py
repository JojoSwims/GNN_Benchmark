#!/usr/bin/env python3
"""MTGODE on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``conv_channels``, capped at 32 for the 859-node graph.  Model-critical
axis: ``solver_1`` (CTA ODE solver — Euler vs RK4).  RK4 quadruples the
per-step compute, so wall-clock will spread.
"""

from gnn_benchmark.models import MTGODEConfig, MTGODEModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(MTGODEConfig(), DATASET)

run_example(
    model_factory=lambda: MTGODEModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":            LogUniform(LR_LOW, LR_HIGH),
        "dropout":       Categorical([0.0, 0.1, 0.3]),
        "conv_channels": Categorical([16, 32]),
        "solver_1":      Categorical(["euler", "rk4"]),
    },
)
