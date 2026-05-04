#!/usr/bin/env python3
"""STAEFormer on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  Spatial
self-attention scales as N²·heads·layers, so ``num_layers=2`` is fixed
in the base config (and dropped from the search) and
``adaptive_embedding_dim`` is capped at 40 (``model_dim`` ∈ {40, 64}).
``num_heads=4`` divides every produced model_dim.
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, staeformer_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce"
base_config = apply_schedule(
    STAEFormerConfig(),
    DATASET,
    num_heads=4,
    num_layers=2,
)

search_space = staeformer_search_space(
    lr_low=LR_LOW,
    lr_high=LR_HIGH,
    adaptive_embedding_dim=Categorical([16, 40]),
)
del search_space["num_layers"]  # pinned in base_config for the N=859 graph

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=search_space,
)
