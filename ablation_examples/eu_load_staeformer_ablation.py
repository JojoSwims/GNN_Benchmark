#!/usr/bin/env python3
"""STAEFormer on EU electricity load with an ablated graph.

Single-run benchmark using the winning STAEFormer hyperparameters from the
fair-protocol search in ``examples_new/eu_load_staeformer_example.py``.  See
``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel

from _shared import run_ablation

DATASET = "eu-load"

# Winning STAEFormer config on eu-load (random search, seed=0).
config = STAEFormerConfig(
    adaptive_embedding_dim=80,
    batch_size=32,
    clip_grad=5.0,
    device="auto",
    dropout=0.1,
    early_stop=7,
    eps=1e-8,
    feed_forward_dim=256,
    input_embedding_dim=24,
    lr=0.0032726888367412277,
    lr_decay_ratio=0.5,
    lr_milestones=[10, 15],
    max_epochs=20,
    num_heads=4,
    num_layers=2,
    seed=None,
    steps_per_day=288,
    tod_embedding_dim=0,
    use_lr_scheduler=True,
    weight_decay=3e-4,
    no_graph=True,
)

run_ablation(model=STAEFormerModel(), config=config, dataset_key=DATASET)
