#!/usr/bin/env python3
"""Graph WaveNet on Chicago / Divvy bikeshare with an ablated graph.

Single-run benchmark using the winning GWN hyperparameters from the
fair-protocol search in ``examples_new/chicago_bikes_gwn_example.py``.
See ``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import GWNConfig, GWNModel

from _shared import run_ablation

DATASET = "divvy-bikeshare-static"

# Winning GWN config on divvy-bikeshare-static (random search, seed=0).
config = GWNConfig(
    addaptadj=True,
    adjtype="doubletransition",
    batch_size=16,
    blocks=4,
    clip_grad=5.0,
    device="auto",
    dropout=0.3,
    early_stop=7,
    eps=1e-8,
    gcn_bool=True,
    kernel_size=2,
    layers=2,
    lr=0.0008506042266922006,
    lr_decay_ratio=0.5,
    lr_milestones=[10, 15],
    max_epochs=20,
    nhid=32,
    seed=None,
    use_lr_scheduler=True,
    weight_decay=1e-4,
    no_graph=True,
)

run_ablation(model=GWNModel(), config=config, dataset_key=DATASET)
