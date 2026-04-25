#!/usr/bin/env python3
"""Tune ASTGCN on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over ASTGCN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run.

Usage:
    python examples/beijing_air_astgcn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import ASTGCNConfig, ASTGCNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

base_config = ASTGCNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# ASTGCN-specific search space (2 x 3 x 3 = 18 trials).
# - lr             : training signal (log-scale pair)
# - K              : Chebyshev polynomial order — the spatial receptive
#                    field; each step propagates one extra hop on the graph
# - nb_chev_filter : width of the spatial convolutions, the main capacity
#                    knob (nb_time_filter is left at its default to keep
#                    the residual path width-matched)
tuner = HyperparameterTuner(
    model_factory=lambda: ASTGCNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":             Categorical([1e-3, 5e-4]),
        "K":              Categorical([2, 3, 4]),
        "nb_chev_filter": Categorical([32, 64, 128]),
    },
    strategy="grid",
)
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(ASTGCNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
