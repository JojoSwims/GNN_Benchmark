#!/usr/bin/env python3
"""Tune MTGODE on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over MTGODE-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run.

Usage:
    python examples/beijing_air_mtgode_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGODEConfig, MTGODEModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

base_config = MTGODEConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
    use_lr_scheduler=False,
)

# MTGODE-specific search space (2 x 3 x 3 = 18 trials).
# - lr            : training signal (log-scale pair)
# - conv_channels : width of the core TCN/GCN stack — the main capacity knob
# - dropout       : regularisation
tuner = HyperparameterTuner(
    model_factory=lambda: MTGODEModel(),
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
    final = runner.run(MTGODEModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
