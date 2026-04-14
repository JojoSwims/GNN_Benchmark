#!/usr/bin/env python3
"""Tune STAEFormer on the Beijing Air dataset, then report test metrics.

Pipeline:
    1. Run a grid search over STAEFormer-specific hyperparameters, scored
       by validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Grid is 2 x 3 x 3 = 18 trials.  Each trial performs a full fit() on
Beijing Air, so the total cost is ~18x a single training run.

Note on num_heads: in this wrapper tod/dow/spatial embedding dims are 0,
so model_dim = input_embedding_dim (24) + adaptive_embedding_dim (80) =
104.  num_heads must divide 104, which is why we search {2, 4, 8}
(104 / 2 = 52, 104 / 4 = 26, 104 / 8 = 13).  Do not add values that
aren't divisors of 104 without also adjusting the embedding dims.

Usage:
    python examples/beijing_air_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "beijing-air"

base_config = STAEFormerConfig(
    max_epochs=10,
    batch_size=16,
    early_stop=5,
)

# STAEFormer-specific search space (2 x 3 x 3 = 18 trials).
# - lr         : training signal (log-scale pair)
# - num_layers : transformer depth — the main capacity knob
# - num_heads  : attention heads; all values must divide model_dim (104)
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
