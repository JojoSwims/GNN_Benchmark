#!/usr/bin/env python3
"""Tune D2STGNN on GDELT Protest Diffusion, then report test metrics.

Pipeline:
    1. Run a grid search over D2STGNN-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline once
       for an unbiased test-set evaluation.

Dataset:
    GDELT Geopolitical Diffusion — daily per-country aggregates over the
    100 most active countries, 2015-02-18 to 2020-01-31 (cut off before the
    COVID news-coverage regime shift). 10 features per node:
    protest_count, threats_issued, coercions_issued, assaults_issued,
    appeals_issued, cooperation_issued, avg_goldstein, avg_tone,
    total_event_count, material_conflict_count.

    The loader's WindowConfig is unpinned, so the model consumes and
    predicts all 10 channels (D_in == D_out == 10).

    Edges come from UNGA voting similarity (Voeten Dataverse): top-10
    diplomatic neighbours per country, ~1,000 directed edges. D2STGNN
    consumes this adjacency via doubletransition and combines it with
    its learned dynamic graph branch.

    The loader auto-downloads pre-fetched parquet caches from Google
    Drive on first use and caches them under
    ``~/.cache/gnn_benchmark/gdelt_protest/``.

Memory notes:
    With N=100 the per-batch tensors are small (the dynamic-graph N×N
    scores tensor is only ~40 kB at fp32), so paper-default architecture
    knobs are fine. The grid below leaves capacity dials at default and
    only sweeps lr / num_hidden / dropout.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/gdelt_protests_d2stgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "gdelt-protest"

print(f"[example] D2STGNN on {DATASET} — workspace={WORKSPACE}")

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
print("[example] Starting hyperparameter grid search (18 trials)...")
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
