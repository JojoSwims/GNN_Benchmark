#!/usr/bin/env python3
"""Tune STAEFormer on NY COVID (county-level), then report test metrics.

Pipeline:
    1. Run a grid search over STAEFormer-specific hyperparameters,
       scored by validation loss. The test set is NOT seen during this
       phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Dataset:
    NYT COVID-19 county-level, 7-day-smoothed deaths aligned with the
    14-day-lagged smoothed cases. Target: ``new_deaths_smooth``.
    N = 3,212 counties, T = 1,138 days, D_in = 2.
    Graph: haversine distances with a 150 km cutoff — ignored by
    STAEFormer (the transformer + adaptive embedding do not use ``adj``).

Memory notes:
    Spatial self-attention in STAEFormer is O(N²·d) per head per layer.
    With N=3,212 that is ~10M attention entries per head; at
    ``model_dim``=104 and ``num_heads``=4 the per-layer attention tensor
    is ~4·32·3212² ≈ 1.3 GB in fp32 for a batch of 32, before any of
    the usual transformer constants. This is the single biggest OOM
    risk in this benchmark. Knobs applied below:
      - batch_size=2             : spatial attention scales with B·N²
      - num_layers=1             : default is 3; each layer stacks
                                   another N² attention.
      - adaptive_embedding_dim=40: halved from 80. model_dim drops from
                                   104 to 64, so num_heads {2, 4, 8}
                                   all divide cleanly.
      - feed_forward_dim=128     : halved from 256 for the same reason.

    If you still OOM, lower ``batch_size`` to 1 or drop
    ``adaptive_embedding_dim`` to 16.

num_heads divisibility:
    With this config, model_dim = input_embedding_dim (24) +
    adaptive_embedding_dim (40) = 64. num_heads must divide 64, so we
    search {2, 4, 8}.

Usage:
    python examples/ny_covid_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "nyc-covid"

print(f"[example] STAEFormer on {DATASET} — workspace={WORKSPACE}")

base_config = STAEFormerConfig(
    max_epochs=10,
    batch_size=2,
    early_stop=5,
    adaptive_embedding_dim=40,
    feed_forward_dim=128,
    num_layers=1,
)

tuner = HyperparameterTuner(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([1e-3, 5e-4]),
        "num_layers": Categorical([1, 2]),
        "num_heads":  Categorical([2, 4]),
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
    final = runner.run(STAEFormerModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
