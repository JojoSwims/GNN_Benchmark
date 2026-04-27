#!/usr/bin/env python3
"""STAEFormer on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``adaptive_embedding_dim``, capped at 16 (``model_dim`` ∈ {32, 40}) because
spatial self-attention scales as N²·heads·layers on N=3212.  Reg axis:
``dropout``.  Model-critical axis: ``num_layers`` ∈ {1, 2} for memory.
``num_heads=4`` divides both produced model_dims; ``feed_forward_dim``
is also halved.
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(
    STAEFormerConfig(),
    DATASET,
    num_heads=4,
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
        "num_layers":             Categorical([1, 2]),
    },
)
