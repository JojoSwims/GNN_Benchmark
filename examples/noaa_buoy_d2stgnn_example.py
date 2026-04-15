#!/usr/bin/env python3
"""Tune D2STGNN on NOAA buoys, then report test metrics.

Pipeline:
    1. Run a grid search over D2STGNN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    NOAA NDBC ocean buoy network (hourly, 2022-2025).  4 features = WSPD
    (target, wind speed), WTMP, WVHT, PRES.  WindowConfig sets
    target_columns=["WSPD"], so the model consumes all 4 features but
    predicts a single channel.

    Requires pre-fetched files under datasets/:
        datasets/noaa_buoy_stations.json
        datasets/noaa_buoy_data.parquet

D2STGNN consumes the haversine-distance adjacency via doubletransition.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/noaa_buoy_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "noaa-buoy"

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
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(D2STGNNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
