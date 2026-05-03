#!/usr/bin/env python3
"""Graph WaveNet on Chicago / Divvy bikeshare with the adaptive graph disabled.

Single-run benchmark using the winning GWN hyperparameters from
``examples_new/chicago_bikes_gwn_example.py``, but with ``addaptadj=False``
so the model relies solely on the dataset's predefined adjacency
(processed via ``adjtype="doubletransition"``).  Compare against
``chicago_bikes_gwn_example.py`` (provided graph + adaptive) and
``chicago_bikes_gwn_ablation.py`` (no graph at all) to isolate the
contribution of the learned adaptive adjacency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel

from _shared import WORKSPACE

DATASET = "divvy-bikeshare-static"

# Winning GWN config on divvy-bikeshare-static (random search, seed=0),
# with the adaptive adjacency switched off.  ``no_graph`` is left at its
# default (False) so the predefined adjacency is still consumed.
config = GWNConfig(
    addaptadj=False,
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
)

model = GWNModel()
print(
    f"[no-adaptive ablation] {model.name} on {DATASET} — "
    "predefined graph only (single run, no tuning)"
)
runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(model, config=config)
print(result.summary())
