#!/usr/bin/env python3
"""Dyn-GWN on ENTSO-E EU load — fair-protocol hyperparameter tuning.

Uses the ``eu-load-dynamic`` registry entry (the same hourly zonal load
series as ``eu-load`` but with the dataset's hourly directed cross-zone
flow snapshots instead of the static interconnection adjacency).  The
flow values are signed MW; the wrapper's default
``flow_transform="log1p_abs_rownorm"`` collapses sign and tames the
heavy tail before the inner softmax.
"""

from gnn_benchmark.models import DynGWNConfig, DynGWNModel
from gnn_benchmark.tuning import Categorical, dyngwn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "eu-load-dynamic"
base_config = apply_schedule(DynGWNConfig(), DATASET)

run_example(
    model_factory=lambda: DynGWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=dyngwn_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        # ~30 zones: N*N is only 900, so the dynamic-branch activations
        # at nhid=64 are still well under a GB.  Widen the channel grid
        # so search can pick a higher-capacity model when it pays off.
        nhid=Categorical([16, 32, 64]),
    ),
)
