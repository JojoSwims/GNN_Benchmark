#!/usr/bin/env python3
"""STAEFormer on LamaH-CE (dynamic) with an ablated graph.

Single-run benchmark using the winning STAEFormer hyperparameters from the
fair-protocol search in ``examples_new/lamah_ce_staeformer_example.py``.  See
``ablation_examples/_shared.py`` for the ablation contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel

from _shared import run_ablation

DATASET = "lamah-ce"

# Winning STAEFormer config on lamah-ce-dynamic (random search, seed=0).
config = STAEFormerConfig(
    adaptive_embedding_dim=40,
    batch_size=16,
    clip_grad=5.0,
    device="auto",
    dropout=0.0,
    early_stop=7,
    eps=1e-8,
    feed_forward_dim=256,
    input_embedding_dim=24,
    lr=0.002519920701883559,
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
