#!/usr/bin/env python3
"""Tune MTGNN on GDELT Protest Diffusion, then report test metrics.

Pipeline:
    1. Run a grid search over MTGNN-specific hyperparameters, scored by
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

    Edges come from UNGA voting similarity (Voeten Dataverse) by default,
    but MTGNN ignores the supplied adjacency (``predefined_A=None`` in
    ``models/mtgnn.py``) and learns its own graph via its graph
    constructor — an interesting contrast for GDELT, where the diplomatic
    similarity prior is non-trivially different from anything purely
    data-driven could recover.

    The loader auto-downloads pre-fetched parquet caches from Google
    Drive on first use and caches them under
    ``~/.cache/gnn_benchmark/gdelt_protest/``.

Memory notes:
    With N=100 the learned (N, subgraph_size) adjacency is tiny and the
    (B, conv_channels, N, T) activations are well within budget at
    paper defaults. No aggressive trims needed.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/gdelt_protests_mtgnn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import MTGNNConfig, MTGNNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "gdelt-protest"

print(f"[example] MTGNN on {DATASET} — workspace={WORKSPACE}")

base_config = MTGNNConfig(
    max_epochs=10,
    batch_size=32,
    early_stop=5,
)

# MTGNN-specific search space (2 x 3 x 3 = 18 trials).
# - lr            : training signal (log-scale pair)
# - conv_channels : hidden channel width — main capacity knob
# - dropout       : regularisation
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
print("[example] Starting hyperparameter grid search (18 trials)...")
tuning_result = tuner.run()
print("[example] Tuning complete.")
print(tuning_result.summary())

if tuning_result.best is not None:
    print("[example] Running final evaluation on test set with best config...")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(MTGNNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
