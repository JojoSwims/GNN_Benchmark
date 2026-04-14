#!/usr/bin/env python3
"""Tune Graph WaveNet on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over GWN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run — run on a
GPU host if you have one.

Usage:
    python examples/beijing_air_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

# Shared base config — fields not listed in the search space use these values.
base_config = GWNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# GWN-specific search space (2 x 3 x 3 = 18 trials).
# Covers the three knobs that move GWN performance most on small
# spatiotemporal datasets:
# - lr      : training signal (log-scale pair)
# - dropout : regularisation — GWN stacks 4 blocks and overfits easily
# - nhid    : hidden channel width — the main capacity knob
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
