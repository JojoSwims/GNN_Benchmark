#!/usr/bin/env python3
"""MTGNN on LamaH-CE (dynamic) with an ablated graph.

Single-run benchmark using the winning MTGNN hyperparameters from the
fair-protocol search in ``examples_new/lamah_ce_mtgnn_example.py``.  See
``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import MTGNNConfig, MTGNNModel

from _shared import run_ablation

DATASET = "lamah-ce"

# Winning MTGNN config on lamah-ce-dynamic (random search, seed=0).
config = MTGNNConfig(
    batch_size=16,
    buildA_true=True,
    clip_grad=5.0,
    conv_channels=32,
    device="auto",
    dilation_exponential=1,
    dropout=0.1,
    early_stop=7,
    end_channels=128,
    eps=1e-8,
    gcn_depth=2,
    gcn_true=True,
    layer_norm_affline=True,
    layers=3,
    lr=0.0023166431971318405,
    lr_decay_ratio=0.5,
    lr_milestones=[10, 15],
    max_epochs=20,
    node_dim=40,
    propalpha=0.05,
    residual_channels=32,
    seed=None,
    skip_channels=64,
    subgraph_size=20,
    tanhalpha=3.0,
    use_lr_scheduler=True,
    weight_decay=1e-4,
    no_graph=True,
)

run_ablation(model=MTGNNModel(), config=config, dataset_key=DATASET)
