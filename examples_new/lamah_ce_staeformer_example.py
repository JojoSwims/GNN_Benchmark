#!/usr/bin/env python3
"""STAEFormer on LamaH-CE (dynamic) — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol.  Capacity axis:
``adaptive_embedding_dim``, capped at 40 here because spatial self-attention
scales as N²·heads on N=859.  Reg axis: ``dropout``.  Model-critical axis:
``num_layers`` (transformer depth), capped at 2 for the same memory reason.
``num_heads=4`` is fixed and divides every produced model_dim (40, 64).
"""

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, LogUniform

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "lamah-ce-dynamic"
base_config = apply_schedule(STAEFormerConfig(), DATASET, num_heads=4)

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space={
        "lr":                     LogUniform(LR_LOW, LR_HIGH),
        "dropout":                Categorical([0.0, 0.1, 0.2]),
        "adaptive_embedding_dim": Categorical([16, 40]),
        "num_layers":             Categorical([1, 2]),
    },
)
