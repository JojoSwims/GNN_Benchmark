#!/usr/bin/env python3
"""ASTGCN on EU electricity load with an ablated graph.

Single-run benchmark using the winning ASTGCN hyperparameters from the
fair-protocol search in ``examples_new/eu_load_astgcn_example.py``.  See
``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import ASTGCNConfig, ASTGCNModel

from _shared import run_ablation

DATASET = "eu-load"

# Winning ASTGCN config on eu-load (random search, seed=0).
config = ASTGCNConfig(
    K=2,
    batch_size=32,
    clip_grad=5.0,
    device="auto",
    early_stop=7,
    eps=1e-8,
    lr=0.00467436955296385,
    lr_decay_ratio=0.5,
    lr_milestones=[10, 15],
    max_epochs=20,
    nb_block=3,
    nb_chev_filter=64,
    nb_time_filter=64,
    seed=None,
    time_strides=1,
    use_lr_scheduler=True,
    weight_decay=1e-5,
    no_graph=True,
)

run_ablation(model=ASTGCNModel(), config=config, dataset_key=DATASET)
