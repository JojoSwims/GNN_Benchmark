#!/usr/bin/env python3
"""GWN on NYC COVID (county-level) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``nhid`` is
capped at 32 (skip/end channels are 8×/16× nhid; the large county graph
makes the upper end memory-tight).  ``blocks=2`` is pinned in the base
config (and dropped from the search) because each block doubles the
dilation footprint.
"""

from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, gwn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(GWNConfig(), DATASET, blocks=2)

search_space = gwn_search_space(
    lr_low=LR_LOW,
    lr_high=LR_HIGH,
    nhid=Categorical([16, 32]),
)
del search_space["blocks"]  # pinned in base_config for the N=3212 graph

run_example(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=search_space,
)
