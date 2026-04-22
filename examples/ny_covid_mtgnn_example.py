#!/usr/bin/env python3
"""Tune MTGNN on NY COVID (county-level), then report test metrics.

Pipeline:
    1. Run a grid search over MTGNN-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Dataset:
    NYT COVID-19 county-level, 7-day-smoothed deaths aligned with the
    14-day-lagged smoothed cases. Target: ``new_deaths_smooth``.
    N = 3,212 counties, T = 1,138 days, D_in = 2.
    Graph: haversine distances with a 150 km cutoff (~128k directed
    edges, avg degree ~20, 17 offshore counties isolated).

Note:
    MTGNN ignores the supplied adjacency (``predefined_A=None`` in
    ``models/mtgnn.py``) and learns its own graph via its graph
    constructor. For N=3,212 the learned node embedding and subgraph
    selection are the main memory terms; we reduce ``node_dim`` and
    ``subgraph_size`` below to keep this in check.

Memory notes:
    MTGNN allocates an N×subgraph_size adjacency at every forward pass
    plus (B, conv_channels, N, T) activations per layer. With N=3,212,
    subgraph_size=10 costs ~32k float32 pairs per batch, which is fine;
    the real memory cost is the (B, C, N, T) activations, so the knobs
    that matter are ``batch_size`` and ``conv_channels``.

Usage:
    python examples/ny_covid_mtgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "nyc-covid"

print(f"[example] MTGNN on {DATASET} — workspace={WORKSPACE}")

# MTGNN-specific memory notes:
#   - batch_size=8        : 4× smaller than LamaH-CE example; (B,C,N,T)
#                           activations scale linearly with B.
#   - conv_channels=16    : halved from default; residual_channels is
#                           tied, so this halves the widest tensors.
#   - subgraph_size=10    : paper default is 20; with 3k nodes the
#                           top-k neighbour mixer still has plenty of
#                           variety, and memory of (N, subgraph_size)
#                           drops to half.
#   - node_dim=20         : halved from 40; affects learned node
#                           embedding only, not forward-pass activations.
#   - layers=2            : default is 3.
base_config = MTGNNConfig(
    max_epochs=10,
    batch_size=8,
    early_stop=5,
    conv_channels=16,
    residual_channels=16,
    skip_channels=32,
    end_channels=64,
    subgraph_size=10,
    node_dim=20,
    layers=2,
)

tuner = HyperparameterTuner(
    model_factory=lambda: MTGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":            Categorical([1e-3, 5e-4]),
        "conv_channels": Categorical([16, 32]),
        "dropout":       Categorical([0.1, 0.3]),
    },
    strategy="grid",
)
print("[example] Starting hyperparameter grid search (8 trials)...")
tuning_result = tuner.run()
print("[example] Tuning complete.")
print(tuning_result.summary())

if tuning_result.best is not None:
    print("[example] Running final evaluation on test set with best config...")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(MTGNNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
