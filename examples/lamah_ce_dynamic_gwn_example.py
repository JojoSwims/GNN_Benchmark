#!/usr/bin/env python3
"""Tune Graph WaveNet on LamaH-CE (dynamic features), then report test metrics.

Pipeline:
    1. Run a grid search over GWN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    LamaH-CE dynamic-only variant (~859 gauges, 9 features = qobs + 8 ERA5
    met forcings).  The WindowConfig pins target_columns=["qobs"], so the
    model consumes all 9 features but predicts a single channel (streamflow).

    The loader auto-downloads the Zenodo archive on first use and caches
    it under ``~/.cache/gnn_benchmark/lamah_ce/``.

Grid is 2 x 3 x 3 = 18 trials.  GWN + the river-network graph is a natural
fit for streamflow: the doubletransition adjacency already encodes
upstream/downstream routing.

Usage:
    python examples/lamah_ce_dynamic_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "lamah-ce-dynamic"

print(f"[example] GWN on {DATASET} — workspace={WORKSPACE}")

base_config = GWNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# GWN-specific search space (2 x 3 x 3 = 18 trials).
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
print("[example] Starting hyperparameter grid search (18 trials)...")
tuning_result = tuner.run()
print("[example] Tuning complete.")
print(tuning_result.summary())

if tuning_result.best is not None:
    print("[example] Running final evaluation on test set with best config...")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(GWNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
