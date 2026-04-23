#!/usr/bin/env python3
"""Run STAEFormer on the EU load dataset (no hyperparameter tuning).

Dataset:
    ENTSO-E European zonal electricity load (hourly, 2023-2024).  49 nodes
    (bidding zones), 1 feature ("load", MW) which is also the target.
    Edges are undirected, unweighted cross-zone interconnections.  Loader
    auto-downloads both files from Google Drive on first use.

Hyperparameters (no grid search):
    Sticking with STAEFormer's defaults — for a 49-node graph with a
    single feature, the default num_layers=3 / num_heads=4 transformer
    is well-sized.  num_heads must divide model_dim; with this wrapper
    model_dim = input_embedding_dim (24) + adaptive_embedding_dim (80)
    = 104, and 4 divides 104.  Only the training-budget knobs
    (max_epochs, batch_size, early_stop) are touched: max_epochs is
    bumped to 50 since we run a single trial instead of a grid.

Note:
    STAEFormer does not use the supplied graph — the transformer +
    adaptive embedding ignore ``adj``.

Usage:
    python examples/eu_load_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel

WORKSPACE = "./benchmark_workspace"
DATASET = "eu-load"

print(f"[example] STAEFormer on {DATASET} — workspace={WORKSPACE}")

config = STAEFormerConfig(
    lr=1e-3,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    batch_size=16,
    max_epochs=50,
    early_stop=10,
)

runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
result = runner.run(STAEFormerModel(), config=config)
print(result.summary())
