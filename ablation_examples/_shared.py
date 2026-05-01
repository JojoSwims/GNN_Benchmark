"""Shared protocol for the *graph-ablation* examples.

These examples answer the question: how much of each GNN's accuracy comes
from the dataset's adjacency vs. from its temporal/architectural machinery?
Each example runs a model with its *winning* hyperparameters (taken from
the random-search benchmark in ``examples_new/``) but with the per-model
``no_graph=True`` ablation flag set, so the spatial component can no longer
mix across nodes while the rest of the architecture is untouched.

Per-model ablation semantics
----------------------------
The ``no_graph`` flag is implemented inside each model wrapper (see commit
"Add no_graph ablation flag to GWN, MTGNN, STAEformer, ASTGCN, MTGODE"):

* GWN — forces ``gcn_bool=False`` / ``addaptadj=False``.
* MTGNN — forces ``gcn_true=False`` / ``buildA_true=False``.
* STAEformer — empties the spatial-attention stack.
* ASTGCN — replaces Chebyshev polynomials with identity matrices.
* MTGODE — collapses graph propagation to a no-op via ``predefined_A=I``.

No tuning
---------
Unlike ``examples_new/``, these scripts skip the random-search step.  They
take the winning config as a literal and run a single deterministic
``BenchmarkRunner`` pass — that's the "use the winning hyperparameters"
contract from the project doc.
"""

from __future__ import annotations

from typing import Any


WORKSPACE = "./benchmark_workspace"


def run_ablation(
    *,
    model,
    config: Any,
    dataset_key: str,
) -> None:
    """Run a single benchmark pass with ``config.no_graph=True``."""
    from gnn_benchmark.benchmark import BenchmarkRunner

    if not getattr(config, "no_graph", False):
        raise ValueError(
            "run_ablation expects config.no_graph=True; got "
            f"no_graph={getattr(config, 'no_graph', None)!r}."
        )

    print(
        f"[ablation] {model.name} on {dataset_key} — "
        f"no_graph=True (single run, no tuning)"
    )
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[dataset_key])
    result = runner.run(model, config=config)
    print(result.summary())


__all__ = ["WORKSPACE", "run_ablation"]
