#!/usr/bin/env python3
"""Tune D2STGNN on the EU load dataset, then report test metrics.

Pipeline:
    1. Run a grid search over D2STGNN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    ENTSO-E European zonal electricity load (hourly, 2023-2024).  49 nodes
    (bidding zones), 1 feature ("load", MW) which is also the target.
    Edges are undirected, unweighted cross-zone interconnections.  Loader
    auto-downloads both files from Google Drive on first use.

D2STGNN consumes the interconnection adjacency via doubletransition.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/eu_load_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "eu-load"

print(f"[example] D2STGNN on {DATASET} — workspace={WORKSPACE}")

base_config = D2STGNNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# D2STGNN-specific search space (2 x 3 x 3 = 18 trials).
# - lr         : D2STGNN's default (2e-3) is higher than GWN/MTGNN
# - num_hidden : hidden channel width — the main capacity knob
# - dropout    : regularisation; D2STGNN uses a lower default (0.1)
tuner = HyperparameterTuner(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([2e-3, 1e-3]),
        "num_hidden": Categorical([16, 32, 64]),
        "dropout":    Categorical([0.1, 0.2, 0.3]),
    },
    strategy="grid",
)
print("[example] Starting hyperparameter grid search (18 trials)...")
tuning_result = tuner.run()
print("[example] Tuning complete.")
print(tuning_result.summary())

if tuning_result.best is not None:
    print("[example] Running final evaluation on test set with best config...")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(D2STGNNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
