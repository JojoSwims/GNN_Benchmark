"""Shared protocol for the fair hyperparameter-tuning examples.

Every example in ``examples_new/`` follows the same protocol so results are
comparable across models on a given dataset and across datasets for a given
model.  The constants below are the entire protocol — model- and
dataset-specific tweaks live in the example files themselves.

Why
---
Earlier examples (now under ``examples_old/``) had a few fairness problems:

  * **Inconsistent trial counts.**  Most examples ran 18 trials, but several
    models on the larger datasets (LamaH-CE, NYC COVID) ran 8.  The hardest
    models on the largest graphs got the *least* exploration.
  * **`max_epochs=10` with default LR milestones at `[20, 40, 60, 80]`.**
    The scheduler never fired, so all comparisons were at constant LR.  Models
    that converge slowly (transformers, ODE) were penalised relative to fast
    convolutional ones.
  * **`lr` only sampled at 2 fixed values.**  Learning rate is the most
    important hyperparameter and the existing grid had no room to find a
    winner outside `{1e-3, 5e-4}`.
  * **Per-dataset memory tweaks were scattered across files** with no single
    source of truth — easy for two examples on the same dataset to drift.
  * **Search-space *shape* differed across models** (e.g. STAEFormer tuned
    depth + heads while everyone else tuned width + dropout).

Fair protocol used here
-----------------------
  * **Random search, n_trials=16, seed=0** for every (model, dataset) pair.
  * **Training schedule depends only on the dataset** (memory budget):
    `batch_size`, `max_epochs`, `early_stop`, `lr_milestones`, and
    `lr_decay_ratio` are pinned in :data:`DATASET_SCHEDULE`, not in any
    single example.  ``lr_decay_ratio=0.5`` is uniform across all models —
    this overrides STAEFormer's and GTS's paper default of 0.1 so scheduler
    aggressiveness is the same on every model.
  * **Each model tunes 4 axes** with the same shape but a model-specific
    fourth knob:

      1. ``lr`` — universal ``LogUniform(1e-4, 5e-3)``.
      2. **Regularization** — ``dropout`` for models that have it,
         ``weight_decay`` for ASTGCN/GTS which don't.
      3. **Capacity** — the model's principal width parameter.
      4. **Model-critical knob** — the highest-leverage non-width
         architecture knob for that model: ``blocks`` (GWN),
         ``subgraph_size`` (MTGNN), ``solver_1`` (MTGODE), ``k_t``
         (D2STGNN), ``num_layers`` (STAEFormer), ``K`` (ASTGCN),
         ``max_diffusion_step`` (GTS).

    Earlier ``examples_old/*`` already tuned several of these (``K``,
    ``max_diffusion_step``, ``num_layers``); collapsing them all to
    ``(lr, reg, capacity)`` lost real signal, so they're back here.
  * **`lr` is sampled log-uniform**, the natural distribution for LR
    search.

Limits we don't claim to solve
------------------------------
  * Per-fit wall-clock isn't equalised — some models are 5× slower per epoch
    than others at the same width.  We equalise the *protocol*, not the
    seconds.  Each example prints
    ``tuning_result.total_compute_time_sec`` so users can audit it.
  * No successive halving / Hyperband — every trial pays for the full fit.
    This keeps trials independent and matches the existing tuner.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

# ---------------------------------------------------------------------------
# Search budget — identical for every (model, dataset) pair.
# ---------------------------------------------------------------------------

WORKSPACE = "./benchmark_workspace"
N_TRIALS = 16
SEED = 0
STRATEGY = "random"

# Common LR range, log-uniform.  Spans roughly the published defaults across
# the wrapped models (1e-3 for GWN/MTGNN/STAEFormer/ASTGCN/MTGODE, 2e-3 for
# D2STGNN, 5e-3 for GTS), with one decade of headroom on either side.
LR_LOW = 1e-4
LR_HIGH = 5e-3


# ---------------------------------------------------------------------------
# Per-dataset training schedule.  Same numbers for every model.
# ---------------------------------------------------------------------------
#
# ``batch_size`` shrinks with N because (B, T, N, C) activation memory is
# linear in B·N.  ``max_epochs`` is set generously enough that the LR
# scheduler (`lr_milestones=[10, 15]`, `lr_decay_ratio=0.5`) actually fires
# twice during training; ``early_stop`` cuts trials that have plateaued.

DATASET_SCHEDULE: dict[str, dict[str, Any]] = {
    "noaa-buoy": dict(
        batch_size=32, max_epochs=20, early_stop=7,
        lr_milestones=[10, 15], lr_decay_ratio=0.5,
    ),
    "eu-load": dict(
        batch_size=32, max_epochs=20, early_stop=7,
        lr_milestones=[10, 15], lr_decay_ratio=0.5,
    ),
    "lamah-ce-dynamic": dict(
        batch_size=16, max_epochs=20, early_stop=7,
        lr_milestones=[10, 15], lr_decay_ratio=0.5,
    ),
    "nyc-covid": dict(
        batch_size=8, max_epochs=20, early_stop=7,
        lr_milestones=[10, 15], lr_decay_ratio=0.5,
    ),
    "divvy-bikeshare-static": dict(
        batch_size=16, max_epochs=20, early_stop=7,
        lr_milestones=[10, 15], lr_decay_ratio=0.5,
    ),
}


def apply_schedule(base_config: Any, dataset_key: str, **extra: Any) -> Any:
    """Return ``base_config`` overridden with the dataset's training schedule.

    ``extra`` lets a single example file pass through model-specific
    fixed-architecture overrides (e.g. STAEFormer's smaller
    ``adaptive_embedding_dim`` on big graphs) on top of the shared schedule.
    """
    schedule = DATASET_SCHEDULE[dataset_key]
    return replace(base_config, **schedule, **extra)


# ---------------------------------------------------------------------------
# A tiny ``run_example`` helper so each example file holds only its
# (model, dataset, search-space) specifics.
# ---------------------------------------------------------------------------

def run_example(
    *,
    model_factory,
    base_config,
    dataset_key: str,
    search_space,
) -> None:
    """Run the fair-protocol pipeline for one (model, dataset) example.

    Steps:
      1. Random search with ``N_TRIALS`` trials and ``SEED`` for reproducibility.
      2. If a winner exists, run the standard benchmark pipeline once on the
         test set with the winning config for an unbiased final number.
      3. Print the tuning summary, the total tuning compute, and the final
         benchmark summary.
    """
    # Imported lazily so simply importing this module doesn't pull in torch.
    from gnn_benchmark.benchmark import BenchmarkRunner
    from gnn_benchmark.tuning import HyperparameterTuner

    print(
        f"[example] {model_factory().name} on {dataset_key} — "
        f"{N_TRIALS} random trials (seed={SEED})"
    )
    tuner = HyperparameterTuner(
        model_factory=model_factory,
        base_config=base_config,
        dataset_key=dataset_key,
        workspace_dir=WORKSPACE,
        search_space=search_space,
        strategy=STRATEGY,
        n_trials=N_TRIALS,
        seed=SEED,
    )
    tuning_result = tuner.run()
    print(tuning_result.summary())
    # Compute audit — the protocol equalises trial count, not wall-clock.
    print(
        f"[example] tuning compute used: "
        f"{tuning_result.total_compute_time_sec:.2f}s across "
        f"{len(tuning_result.trials)} trial(s)."
    )

    if tuning_result.best is None:
        print("[example] No successful trials — skipping final evaluation.")
        return

    print("[example] Running final test-set evaluation with best config …")
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[dataset_key])
    final = runner.run(model_factory(), config=tuning_result.best.config)
    print(final.summary())
