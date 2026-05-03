#!/usr/bin/env python3
"""Dyn-GWN on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

Uses the ``divvy-bikeshare-dynamic`` registry entry (same Chicago-Loop
station subset, winter dropped, 30 % activity-rate threshold as
``divvy-bikeshare-static``) but swaps the haversine static graph for
the hourly directed trip-count snapshots.  Trip counts are non-negative
but extremely heavy-tailed (most station pairs see zero per hour, busy
pairs see hundreds), so the wrapper's default
``flow_transform="log1p_abs_rownorm"`` is essential — raw counts would
saturate Dyn-GWN's destination-softmax onto a single hot edge per row.
"""

from gnn_benchmark.models import DynGWNConfig, DynGWNModel
from gnn_benchmark.tuning import Categorical, dyngwn_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-dynamic"
base_config = apply_schedule(DynGWNConfig(), DATASET)

run_example(
    model_factory=lambda: DynGWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=dyngwn_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        # ~200 stations: cap dilation/skip channels to keep activation
        # memory manageable on a single GPU. ``nhid * 8`` is the
        # skip-channel count — at nhid=32 that's 256 skip channels x
        # 195^2 dynamic-graph cells per timestep, which is plenty.
        nhid=Categorical([16, 32]),
    ),
)
