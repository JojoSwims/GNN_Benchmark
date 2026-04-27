#!/usr/bin/env python3
"""Tune STAEFormer on Divvy bikeshare (static edges), then report test metrics.

Uses the ``divvy-bikeshare-static`` registry entry, which emits only the
static haversine graph sparsified with a 2 km cutoff (no bbox / top-K).
N ≈ 3,658 stations, T ≈ 18,264 hourly steps. STAEFormer ignores the
adjacency entirely (pure spatial transformer), so the 2 km cutoff does
not affect this model's inputs — it only speeds up workspace I/O.

Note on num_heads: in this wrapper tod/dow/spatial embedding dims are 0,
so model_dim = input_embedding_dim (24) + adaptive_embedding_dim (80) =
104. num_heads must divide 104, which is why we search {2, 4, 8}
(104 / 2 = 52, 104 / 4 = 26, 104 / 8 = 13). Do not add values that
aren't divisors of 104 without also adjusting the embedding dims.

Pipeline:
    1. Grid search over STAEFormer-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.

Usage:
    python examples/divvy_bikeshare_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "divvy-bikeshare-static"

base_config = STAEFormerConfig(
    max_epochs=10,
    batch_size=16,
    early_stop=5,
)

# STAEFormer-specific search space (2 x 3 x 3 = 18 trials).
tuner = HyperparameterTuner(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":         Categorical([1e-3, 5e-4]),
        "num_layers": Categorical([2, 3, 4]),
        "num_heads":  Categorical([2, 4, 8]),
    },
    strategy="grid",
)
tuning_result = tuner.run()
print(tuning_result.summary())

if tuning_result.best is not None:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])
    final = runner.run(STAEFormerModel(), config=tuning_result.best.config)
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
