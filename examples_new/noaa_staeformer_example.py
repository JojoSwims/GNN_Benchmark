#!/usr/bin/env python3
"""STAEFormer on NOAA buoys — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  ``num_heads=4``
is fixed in the base config and divides every produced
``model_dim = input_embedding_dim + adaptive_embedding_dim``
(40, 64, 104).
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import staeformer_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "noaa-buoy"
base_config = apply_schedule(STAEFormerConfig(), DATASET, num_heads=4)

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=staeformer_search_space(lr_low=LR_LOW, lr_high=LR_HIGH),
)
