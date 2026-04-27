#!/usr/bin/env python3
"""STAEFormer on EU electricity load — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``adaptive_embedding_dim`` (``model_dim`` = 24 + adaptive_emb).  Reg axis:
``dropout``.  ``num_heads=4`` is fixed and divides every model_dim this
search produces (40, 64, 104).
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "eu-load"
base_config = apply_schedule(STAEFormerConfig(), DATASET, num_heads=4)

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":                     LogUniform(LR_LOW, LR_HIGH),
        "dropout":                Categorical([0.0, 0.1, 0.2, 0.3]),
        "adaptive_embedding_dim": Categorical([16, 40, 80]),
    },
)
