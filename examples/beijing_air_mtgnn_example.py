#!/usr/bin/env python3
"""Tune MTGNN on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over MTGNN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run.

Usage:
    python examples/beijing_air_mtgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

base_config = MTGNNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# MTGNN-specific search space (2 x 3 x 3 = 18 trials).
# - lr            : training signal (log-scale pair)
# - conv_channels : width of the core TCN/GCN stack — the main capacity knob
# - dropout       : regularisation
#
# Note: residual_channels is left at its default (32) for all trials.  The
# MTGNN blocks already support a mismatch between conv_channels and
# residual_channels, so we only vary conv_channels here.
tuner = HyperparameterTuner(
    model_factory=lambda: MTGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":            Categorical([1e-3, 5e-4]),
        "conv_channels": Categorical([16, 32, 64]),
        "dropout":       Categorical([0.1, 0.3, 0.5]),
    },
    strategy="grid",
)
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(MTGNNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
