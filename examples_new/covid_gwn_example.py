#!/usr/bin/env python3
"""GWN on NYC COVID (county-level) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis: ``nhid``,
capped at 32 (GWN's skip/end channels are 8×/16× nhid; N=3212 makes the
upper end memory-tight).  ``blocks=2`` is fixed for the same reason; each
block doubles the dilation footprint.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(GWNConfig(), DATASET, blocks=2)

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
