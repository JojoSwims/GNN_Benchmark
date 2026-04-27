#!/usr/bin/env python3
"""Tune D2STGNN on NY COVID (county-level), then report test metrics.

Pipeline:
    1. Run a grid search over D2STGNN-specific hyperparameters, scored
       by validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Dataset:
    NYT COVID-19 county-level, 7-day-smoothed deaths aligned with the
    14-day-lagged smoothed cases. Target: ``new_deaths_smooth``.
    N = 3,212 counties, T = 1,138 days, D_in = 2.
    Graph: haversine distances with a 150 km cutoff (~128k directed
    edges). D2STGNN DOES use the supplied adjacency (via
    doubletransition) and combines it with learned dynamic graph
    structure.

Memory notes:
    D2STGNN's decoupling block stacks a diffusion conv + dynamic-graph
    conv per layer, with activations of shape (B, T, N, num_hidden).
    The dynamic-graph branch additionally computes an N×N scores
    tensor per layer (3,212² = ~40 MB at fp32), and gradients double
    that. Knobs applied below:
      - batch_size=8             : 4× smaller than LamaH-CE example;
                                   (B, T, N, C) scales linearly with B.
      - num_hidden=16            : halved from 32; residual and conv
                                   stacks all use this width.
      - num_modalities=2         : kept at default — cheaper than adding
                                   hidden width for the same capacity.

Usage:
    python examples/ny_covid_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "nyc-covid"

print(f"[example] D2STGNN on {DATASET} — workspace={WORKSPACE}")

base_config = D2STGNNConfig(
    max_epochs=10,
    batch_size=8,
    early_stop=5,
    num_hidden=16,
)

tuner = HyperparameterTuner(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([2e-3, 1e-3]),
        "num_hidden": Categorical([16, 32]),
        "dropout":    Categorical([0.1, 0.2]),
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
    final = runner.run(D2STGNNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
