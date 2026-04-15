#!/usr/bin/env python3
"""Tune Graph WaveNet on NOAA buoys, then report test metrics.

Pipeline:
    1. Run a grid search over GWN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    NOAA NDBC ocean buoy network (hourly, 2022-2025).  4 features = WSPD
    (target, wind speed), WTMP (sea surface temp), WVHT (significant wave
    height), PRES (sea level pressure).  WindowConfig sets
    target_columns=["WSPD"], so the model consumes all 4 features but
    predicts a single channel.

    The loader auto-downloads station metadata and stdmet archives from
    the NDBC public archive on first use and caches them under
    ``~/.cache/gnn_benchmark/noaa_buoy/``.

GWN consumes the haversine-distance adjacency across buoys.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/noaa_buoy_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "noaa-buoy"

base_config = GWNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# GWN-specific search space (2 x 3 x 3 = 18 trials).
# - lr      : training signal (log-scale pair)
# - dropout : regularisation
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

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(GWNModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
