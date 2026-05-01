#!/usr/bin/env python3
"""MTGODE on EU electricity load with an ablated graph.

Single-run benchmark using the winning MTGODE hyperparameters from the
fair-protocol search in ``examples_new/eu_load_mtgode_example.py``.  See
``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import MTGODEConfig, MTGODEModel

from _shared import run_ablation

DATASET = "eu-load"

# Winning MTGODE config on eu-load (random search, seed=0).
config = MTGODEConfig(
    adjoint=False,
    alpha=2.0,
    atol=1e-3,
    batch_size=32,
    buildA_true=True,
    clip_grad=5.0,
    conv_channels=64,
    device="auto",
    dilation_exponential=1,
    dropout=0.1,
    early_stop=7,
    end_channels=128,
    eps=1e-8,
    ln_affine=True,
    lr=0.003012343636515041,
    lr_decay_ratio=0.5,
    lr_milestones=[10, 15],
    max_epochs=20,
    node_dim=40,
    perturb=False,
    rtol=1e-4,
    seed=None,
    solver_1="euler",
    solver_2="euler",
    step_1=0.125,
    step_2=0.25,
    subgraph_size=20,
    tanhalpha=3.0,
    time_1=1.0,
    time_2=1.0,
    use_lr_scheduler=True,
    weight_decay=1e-4,
    no_graph=True,
)

run_ablation(model=MTGODEModel(), config=config, dataset_key=DATASET)
