#!/usr/bin/env python3
"""Tune STAEFormer on GDELT Protest Diffusion, then report test metrics.

Pipeline:
    1. Run a grid search over STAEFormer-specific hyperparameters,
       scored by validation loss. The test set is NOT seen during this
       phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

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
    but STAEFormer ignores the supplied adjacency — its spatial mixing
    relies entirely on full self-attention plus the adaptive embedding.

    The loader auto-downloads pre-fetched parquet caches from Google
    Drive on first use and caches them under
    ``~/.cache/gnn_benchmark/gdelt_protest/``.

Memory notes:
    Spatial self-attention is O(N²·d) per head per layer. With N=100,
    N² = 10k entries — trivial — so paper-default architecture knobs
    are fine. No batch / depth trims needed.

num_heads divisibility:
    Default model_dim = input_embedding_dim (24) + adaptive_embedding_dim
    (80) = 104. num_heads must divide 104, so we search {2, 4, 8}.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/gdelt_protests_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "gdelt-protest"

print(f"[example] STAEFormer on {DATASET} — workspace={WORKSPACE}")

base_config = STAEFormerConfig(
    max_epochs=10,
    batch_size=16,
    early_stop=5,
)

# STAEFormer-specific search space (2 x 3 x 3 = 18 trials).
# - lr         : training signal (log-scale pair)
# - num_layers : depth of the stacked spatio-temporal transformer
# - num_heads  : attention heads (must divide model_dim=104)
tuner = HyperparameterTuner(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([1e-3, 5e-4]),
        "num_layers": Categorical([1, 2, 3]),
        "num_heads":  Categorical([2, 4, 8]),
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
    final = runner.run(STAEFormerModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
