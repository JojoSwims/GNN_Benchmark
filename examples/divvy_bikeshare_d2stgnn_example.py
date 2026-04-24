#!/usr/bin/env python3
"""Tune D2STGNN on Divvy bikeshare (static edges), then report test metrics.

Uses the ``divvy-bikeshare-static`` registry entry, which emits only the
static haversine graph sparsified with a 2 km cutoff (no bbox / top-K).
N ≈ 3,658 stations, T ≈ 18,264 hourly steps.

Note: D2STGNN's internal dynamic-graph constructor materializes
``(B, N, k_t * N)`` tensors — at N=3,658, B=32, k_t=10 that is ~17 GB.
Expect OOM on a 16 GB GPU at ``batch_size=32``; lower to 2-4 if needed.

Pipeline:
    1. Grid search over D2STGNN-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/divvy_bikeshare_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "divvy-bikeshare-static"

base_config = D2STGNNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# D2STGNN-specific search space (2 x 3 x 3 = 18 trials).
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
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(D2STGNNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
