#!/usr/bin/env python3
"""Tune GTS on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over GTS-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run.

Usage:
    python examples/beijing_air_gts_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GTSConfig, GTSModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

base_config = GTSConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# GTS-specific search space (2 x 3 x 3 = 18 trials).
# - lr                 : GTS's default (5e-3) is higher than most baselines
# - rnn_units          : width of the DCGRU encoder/decoder — main capacity knob
# - max_diffusion_step : diffusion order for graph convolution inside DCGRU
tuner = HyperparameterTuner(
    model_factory=lambda: GTSModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":                 Categorical([5e-3, 1e-3]),
        "rnn_units":          Categorical([32, 64, 96]),
        "max_diffusion_step": Categorical([1, 2, 3]),
    },
    strategy="grid",
)
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(GTSModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
