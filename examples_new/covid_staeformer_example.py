#!/usr/bin/env python3
"""STAEFormer on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  Spatial
self-attention scales as N²·heads·layers, so on N=3212 we pin
``num_layers=1`` and ``feed_forward_dim=128`` in the base config (and
drop both from the search), and cap ``adaptive_embedding_dim`` at 16
(``model_dim`` ∈ {32, 40}).  ``num_heads=4`` divides both.
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, staeformer_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(
    STAEFormerConfig(),
    DATASET,
    num_heads=4,
    num_layers=1,
    feed_forward_dim=128,
)

search_space = staeformer_search_space(
    lr_low=LR_LOW,
    lr_high=LR_HIGH,
    adaptive_embedding_dim=Categorical([8, 16]),
)
# Pinned in base_config for the N=3212 graph.
del search_space["num_layers"]
del search_space["feed_forward_dim"]

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=search_space,
)
