#!/usr/bin/env python3
"""Tune Graph WaveNet on Divvy bikeshare (static edges), then report test metrics.

Uses the ``divvy-bikeshare-static`` registry entry, which emits only the
static haversine graph sparsified with a 2 km cutoff (no bbox / top-K).
N ≈ 3,658 stations, T ≈ 18,264 hourly steps.

Note: N=3,658 is an order of magnitude larger than Beijing Air (N=36),
so each fit() is correspondingly slower. Consider lowering
``max_epochs`` and ``batch_size`` if running on CPU / small GPU.

Pipeline:
    1. Grid search over GWN-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/divvy_bikeshare_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "divvy-bikeshare-static"

# Shared base config — fields not listed in the search space use these values.
base_config = GWNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# GWN-specific search space (2 x 3 x 3 = 18 trials).
tuner = HyperparameterTuner(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":      Categorical([1e-3, 5e-4]),
        "dropout": Categorical([0.1, 0.3, 0.5]),
        "nhid":    Categorical([16, 32, 64]),
    },
    strategy="grid",
)
tuning_result = tuner.run()
print(tuning_result.summary())

# Final, unbiased test-set evaluation with the winning config.
if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(GWNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
