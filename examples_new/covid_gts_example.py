#!/usr/bin/env python3
"""GTS on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``rnn_units``
is capped at 64 — GTS materialises an N×N learned graph, so the rnn
width is the larger memory lever on the large county graph.
"""

from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import Categorical, gts_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
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
