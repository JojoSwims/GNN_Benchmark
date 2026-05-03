#!/usr/bin/env python3
"""MTGODE on EU electricity load with the adaptive graph disabled.

Single-run benchmark using the winning MTGODE hyperparameters from
``examples_new/eu_load_mtgode_example.py``, but with ``buildA_true=False``
so the model uses the dataset's predefined adjacency instead of the
graph constructor's learned one.  Compare against
``eu_load_mtgode_example.py`` (learned graph) and
``eu_load_mtgode_ablation.py`` (no graph at all) to isolate the
contribution of the learned adjacency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGODEConfig, MTGODEModel

from _shared import WORKSPACE

DATASET = "eu-load"

# Winning MTGODE config on eu-load (random search, seed=0), with the
# learned graph constructor switched off so ``predefined_A`` is used.
config = MTGODEConfig(
    adjoint=False,
    alpha=2.0,
    atol=1e-3,
    batch_size=32,
    buildA_true=False,
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
)

model = MTGODEModel()
print(
    f"[no-adaptive ablation] {model.name} on {DATASET} — "
    "predefined graph only (single run, no tuning)"
)
runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(model, config=config)
print(result.summary())
