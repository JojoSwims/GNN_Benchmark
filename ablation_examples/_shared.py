"""Shared protocol for the *graph-ablation* examples.

These examples answer the question: how much of each GNN's accuracy comes
from the dataset's adjacency matrix vs. from its temporal/architectural
machinery?  Each example runs a model with its *winning* hyperparameters
(taken from the random-search benchmark in ``examples_new/``) but swaps the
real adjacency for an ablated one, so the same model is graded twice — once
in ``examples_new/`` with the true graph, and once here with the ablation.

Ablation
--------
The default ablation is the **identity adjacency** (``np.eye(N)``): every
node only sees itself across the graph dimension.  Models that *learn* an
adjacency (GWN with ``addaptadj``, MTGNN/MTGODE with ``buildA_true``) can
still recover cross-node structure from data, but they no longer get the
real interconnection prior — exactly the comparison we want.

Switch ``ABLATION_MODE`` below to ``"zero"`` for a fully blank adjacency,
or ``"random"`` for a density-matched random graph (seeded for repeatability).

No tuning
---------
Unlike ``examples_new/``, these scripts skip the random-search step.  They
take the winning config as a literal and run a single deterministic
``BenchmarkRunner`` pass — that's the "use the winning hyperparameters"
contract from the project doc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np


WORKSPACE = "./benchmark_workspace"

# "identity" → np.eye(N); "zero" → np.zeros((N, N)); "random" → density-matched
# Erdős-Rényi graph using ABLATION_SEED.  Identity is the canonical no-graph
# ablation (self-loops only, no neighbours).
ABLATION_MODE: Literal["identity", "zero", "random"] = "identity"
ABLATION_SEED = 0


def ablate_adjacency(adj: np.ndarray | None) -> np.ndarray | None:
    """Return an ablated copy of ``adj`` according to ``ABLATION_MODE``.

    ``adj`` may be ``None`` for datasets without an edge list; we return
    ``None`` unchanged so model wrappers that handle missing graphs keep
    working.
    """
    if adj is None:
        return None

    n = adj.shape[0]
    if ABLATION_MODE == "identity":
        return np.eye(n, dtype=adj.dtype)
    if ABLATION_MODE == "zero":
        return np.zeros((n, n), dtype=adj.dtype)
    if ABLATION_MODE == "random":
        # Density-matched, symmetric, zero diagonal, seeded for reproducibility.
        density = float((adj != 0).sum()) / max(n * n - n, 1)
        rng = np.random.default_rng(ABLATION_SEED)
        upper = rng.random((n, n)) < density
        upper = np.triu(upper, k=1)
        m = (upper | upper.T).astype(adj.dtype)
        return m
    raise ValueError(f"Unknown ABLATION_MODE: {ABLATION_MODE!r}")


def run_ablation(
    *,
    model,
    config: Any,
    dataset_key: str,
) -> None:
    """Run a single benchmark pass with the graph replaced by the ablation.

    The runner subclass below intercepts ``prepare_dataset`` so the model's
    ``fit`` and ``predict`` calls receive the ablated adjacency instead of
    the dataset's real one.  Every other piece of the pipeline (windowing,
    splits, metrics) is identical to ``BenchmarkRunner``, which keeps the
    final numbers directly comparable to ``examples_new/``.
    """
    from gnn_benchmark.benchmark import BenchmarkRunner

    class AblatedBenchmarkRunner(BenchmarkRunner):
        def prepare_dataset(self, workspace, dataset_key):  # type: ignore[override]
            prepared = super().prepare_dataset(workspace, dataset_key)
            prepared.adj = ablate_adjacency(prepared.adj)
            return prepared

    print(
        f"[ablation] {model.name} on {dataset_key} — "
        f"adjacency = {ABLATION_MODE} (single run, no tuning)"
    )
    runner = AblatedBenchmarkRunner(workspace_dir=WORKSPACE, datasets=[dataset_key])
    result = runner.run(model, config=config)
    print(result.summary())


__all__ = [
    "ABLATION_MODE",
    "ABLATION_SEED",
    "WORKSPACE",
    "ablate_adjacency",
    "run_ablation",
]
