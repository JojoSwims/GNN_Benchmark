#!/usr/bin/env python3
"""Tune Graph WaveNet on NY COVID (county-level), then report test metrics.

Pipeline:
    1. Run a grid search over GWN-specific hyperparameters, scored by
       validation loss. The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Dataset:
    NYT COVID-19 county-level, 7-day-smoothed deaths aligned with the
    14-day-lagged smoothed cases. Target: ``new_deaths_smooth``.
    N = 3,212 counties (~4× LamaH-CE), T = 1,138 days, D_in = 2.
    Graph: haversine distances with a 150 km cutoff (~128k directed
    edges, avg degree ~20, 17 offshore counties isolated).

Memory notes (see 'OOM mitigations' at the bottom):
    With 3,212 nodes the per-batch activation tensors are roughly four
    times larger than for LamaH-CE. GWN stacks 4 residual blocks × 2
    layers of dilated conv by default; each intermediate tensor is
    (B, nhid*8, N, T) at the skip connection. The config below halves
    the hidden width and uses a small batch to fit comfortably on
    12 GB cards; scale ``nhid`` and ``batch_size`` up once you confirm
    headroom.

Usage:
    python examples/ny_covid_gwn_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import GWNConfig, GWNModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "nyc-covid"

print(f"[example] GWN on {DATASET} — workspace={WORKSPACE}")

# Base config is intentionally lean for this dataset. Starting points
# for the knobs that dominate memory:
#   - batch_size=8   : 4× smaller than the LamaH-CE GWN example because
#                      N is ~4× bigger; activation memory is ~linear in B·N.
#   - nhid=16        : half the GWN default; skip/end channels derive
#                      from nhid so this halves the widest tensors.
#   - blocks=2       : default is 4; each block doubles the dilation
#                      footprint.
# Raise these back up once you confirm headroom on your GPU.
base_config = GWNConfig(
    max_epochs=10,
    batch_size=8,
    early_stop=5,
    nhid=16,
    blocks=2,
)

# Smaller 2×2×2 = 8-trial grid (vs 18 for LamaH-CE) so the whole sweep
# stays within a reasonable wall-clock budget on a single GPU.
tuner = HyperparameterTuner(
    model_factory=lambda: GWNModel(),
    base_config=base_config,
    dataset_key=DATASET,
    workspace_dir=WORKSPACE,
    search_space={
        "lr":      Categorical([1e-3, 5e-4]),
        "dropout": Categorical([0.1, 0.3]),
        "nhid":    Categorical([16, 32]),
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
    final = runner.run(GWNModel(), config=tuning_result.best.config)
    print("[example] Final evaluation complete.")
    print(final.summary())
else:
    print("No successful trials — skipping final evaluation.")
