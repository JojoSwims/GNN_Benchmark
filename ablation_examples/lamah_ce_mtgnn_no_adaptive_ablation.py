#!/usr/bin/env python3
"""MTGNN on LamaH-CE (dynamic) with the adaptive graph disabled.

Single-run benchmark using the winning MTGNN hyperparameters from
``examples_new/lamah_ce_mtgnn_example.py``, but with ``buildA_true=False``
so the model uses the dataset's predefined adjacency instead of the
graph constructor's learned one.  ``gcn_true`` is left True so spatial
propagation still happens — over the provided graph only.  Compare
against ``lamah_ce_mtgnn_example.py`` (learned graph) and
``lamah_ce_mtgnn_ablation.py`` (no graph at all) to isolate the
contribution of the learned adjacency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGNNConfig, MTGNNModel

from _shared import WORKSPACE

DATASET = "lamah-ce"

# Winning MTGNN config on lamah-ce-dynamic (random search, seed=0), with
# the learned graph constructor switched off so ``predefined_A`` is used.
config = MTGNNConfig(
    batch_size=16,
    buildA_true=False,
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
)

model = MTGNNModel()
print(
    f"[no-adaptive ablation] {model.name} on {DATASET} — "
    "predefined graph only (single run, no tuning)"
)
runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(model, config=config)
print(result.summary())
