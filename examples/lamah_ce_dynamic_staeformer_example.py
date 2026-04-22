#!/usr/bin/env python3
"""Tune STAEFormer on LamaH-CE (dynamic features), then report test metrics.

Pipeline:
    1. Run a grid search over STAEFormer-specific hyperparameters, scored
       by validation loss.  The test set is NOT seen during this phase.
    2. Take the winning config and run the standard benchmark pipeline
       once for an unbiased test-set evaluation.

Dataset:
    LamaH-CE dynamic-only variant (~859 gauges, 9 features = qobs + 8 ERA5
    met forcings).  The WindowConfig pins target_columns=["qobs"], so the
    model consumes all 9 features but predicts a single channel (streamflow).

    The loader auto-downloads the Zenodo archive on first use and caches
    it under ``~/.cache/gnn_benchmark/lamah_ce/``.

Memory notes:
    Spatial self-attention scales with N² per head per layer. With
    N=859 gauges, model_dim=104, num_heads=4 and the default 3 layers
    this comes out to ~47M fp32 entries per batch of 16 just for the
    attention scores — tight on 12 GB cards and a common OOM source.
    We halve ``adaptive_embedding_dim`` (80 → 40, model_dim drops to
    64) and drop ``num_layers`` to 2 to cut attention memory ~4×.
    If you still OOM, lower ``batch_size`` to 4 or set
    ``adaptive_embedding_dim=16``.

    With model_dim=64 the valid ``num_heads`` divisors are {2, 4, 8};
    the search grid below stays inside that set.

STAEFormer does not use the supplied graph — the transformer + adaptive
embedding ignore `adj`.

Grid is 2 x 2 x 2 = 8 trials (down from 18) to keep the sweep bounded
while the per-trial memory footprint is smaller.

Usage:
    python examples/lamah_ce_dynamic_staeformer_example.py
"""

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, HyperparameterTuner

WORKSPACE = "./benchmark_workspace"
DATASET = "lamah-ce-dynamic"

print(f"[example] STAEFormer on {DATASET} — workspace={WORKSPACE}")

base_config = STAEFormerConfig(
    max_epochs=10,
    batch_size=8,
    early_stop=5,
    adaptive_embedding_dim=40,
    feed_forward_dim=128,
    num_layers=2,
)

# STAEFormer-specific search space (2 x 2 x 2 = 8 trials).
# - lr         : training signal (log-scale pair)
# - num_layers : transformer depth; kept low because each layer doubles
#                the spatial-attention activation footprint.
# - num_heads  : attention heads; all values must divide model_dim (64
#                with the config above).
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
