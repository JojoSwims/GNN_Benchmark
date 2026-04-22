#!/usr/bin/env python3
"""Tune D2STGNN on LamaH-CE (dynamic features), then report test metrics.

Pipeline:
    1. Run a grid search over D2STGNN-specific hyperparameters, scored by
       validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    LamaH-CE dynamic-only variant (~859 gauges, 9 features = qobs + 8 ERA5
    met forcings).  The WindowConfig pins target_columns=["qobs"], so the
    model consumes all 9 features but predicts a single channel (streamflow).

    The loader auto-downloads the Zenodo archive on first use and caches
    it under ``~/.cache/gnn_benchmark/lamah_ce/``.

Note:
    D2STGNN *does* use the supplied river-network adjacency (via
    doubletransition). This combines well with its dynamic graph
    learning.

Memory notes:
    Per layer D2STGNN holds (B, T, N, num_hidden) activations and a
    dynamic-graph scores tensor of shape (B, N, N). With N=859 and
    batch_size=32 the scores tensor alone is ~95 MB per layer; gradients
    double that. Dropping ``batch_size`` to 16 and ``num_hidden`` to 16
    cuts memory roughly 4× and resolves most OOMs on 12 GB cards. If
    tight, further reduce ``batch_size`` to 8.

Grid is 2 x 2 x 2 = 8 trials (down from 18) to keep the sweep bounded.

Usage:
    python examples/lamah_ce_dynamic_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "lamah-ce-dynamic"

print(f"[example] D2STGNN on {DATASET} — workspace={WORKSPACE}")

base_config = D2STGNNConfig(
    max_epochs=10,
    batch_size=16,
    early_stop=5,
    num_hidden=16,
)

# D2STGNN-specific search space (2 x 2 x 2 = 8 trials).
# - lr         : D2STGNN's default (2e-3) is higher than GWN/MTGNN
# - num_hidden : hidden channel width — the main capacity knob;
#                (B,T,N,C) activation memory is linear in this.
# - dropout    : regularisation; D2STGNN uses a lower default (0.1)
tuner = HyperparameterTuner(
    model_factory=lambda: D2STGNNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([2e-3, 1e-3]),
        "num_hidden": Categorical([16, 32]),
        "dropout":    Categorical([0.1, 0.2]),
    },
    strategy="grid",
)
print("[example] Starting hyperparameter grid search (8 trials)...")
tuning_result = tuner.run()
print("[example] Tuning complete.")
print(tuning_result.summary())

if tuning_result.best is not None:
    print("[example] Running final evaluation on test set with best config...")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(D2STGNNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
