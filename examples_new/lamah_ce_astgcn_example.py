#!/usr/bin/env python3
"""ASTGCN on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``nb_chev_filter``, capped at 64.  Reg axis: ``weight_decay``.
"""

from gnn_benchmark.models import ASTGCNConfig, ASTGCNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(ASTGCNConfig(), DATASET)

run_example(
    model_factory=lambda: ASTGCNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":             LogUniform(LR_LOW, LR_HIGH),
        "weight_decay":   Categorical([0.0, 1e-5, 1e-4]),
        "nb_chev_filter": Categorical([32, 64]),
    },
)
