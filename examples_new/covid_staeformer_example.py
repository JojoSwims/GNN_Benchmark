#!/usr/bin/env python3
"""STAEFormer on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``adaptive_embedding_dim``.  Spatial self-attention scales as N²·heads, so
on N=3212 we drop ``num_layers`` to 1 and cap adaptive_embedding_dim at 16
(``model_dim`` ∈ {32, 40}).  ``num_heads=4`` divides both.
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(
    STAEFormerConfig(),
    DATASET,
    num_heads=4,
    num_layers=1,
    feed_forward_dim=128,
)

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":                     LogUniform(LR_LOW, LR_HIGH),
        "dropout":                Categorical([0.0, 0.1, 0.2]),
        "adaptive_embedding_dim": Categorical([8, 16]),
    },
)
