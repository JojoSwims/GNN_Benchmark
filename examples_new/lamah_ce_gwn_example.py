#!/usr/bin/env python3
"""GWN on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis: ``nhid``;
the upper end is capped at 32 here because GWN's skip/end channels scale
as ``nhid * 8 / 16`` and N=859 already pushes activation memory.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
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
