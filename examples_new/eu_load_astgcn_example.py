#!/usr/bin/env python3
"""ASTGCN on EU electricity load — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``nb_chev_filter``.  Reg axis: ``weight_decay``.  Model-critical axis:
``K`` (Chebyshev polynomial order — graph receptive field per layer).
"""

from gnn_benchmark.models import ASTGCNConfig, ASTGCNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "eu-load"
base_config = apply_schedule(ASTGCNConfig(), DATASET)

run_example(
    model_factory=lambda: ASTGCNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":             LogUniform(LR_LOW, LR_HIGH),
        "weight_decay":   Categorical([0.0, 1e-5, 1e-4, 1e-3]),
        "nb_chev_filter": Categorical([32, 64, 128]),
        "K":              Categorical([2, 3, 4]),
    },
)
