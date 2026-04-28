#!/usr/bin/env python3
"""GTS on Chicago / Divvy bikeshare — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``rnn_units``
is capped at 64 — DCGRU activations are (B, T, N, rnn_units) and the
N≈195 graph (after bbox + min_active_fraction filtering) fits
comfortably; the cap is kept for consistency with the protocol.
"""

from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import Categorical, gts_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "divvy-bikeshare-static"
base_config = apply_schedule(GTSConfig(), DATASET)

run_example(
    model_factory=lambda: GTSModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=gts_search_space(
        lr_low=LR_LOW,
        lr_high=LR_HIGH,
        rnn_units=Categorical([32, 64]),
    ),
)
