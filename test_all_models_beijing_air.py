#!/usr/bin/env python3
"""Smoke test: every model runs one epoch on the Beijing Air dataset.

No hyperparameter tuning. Each trainable model is configured with
``max_epochs=1`` and ``early_stop`` set high enough that the run reflects
exactly one epoch. ``LastValueModel`` has no epochs and uses ``config=None``.

Usage:
    python test_all_models_beijing_air.py

Exits non-zero if any model fails. Per-model status is printed at the end.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from typing import Any, Callable

from gnn_benchmark.benchmark import BenchmarkRunner
from gnn_benchmark.models import (
    ASTGCNConfig,
    ASTGCNModel,
    BenchmarkModel,
    GWNConfig,
    GWNModel,
    LastValueModel,
    MLPMultivariateConfig,
    MLPMultivariateModel,
    MTGNNConfig,
    MTGNNModel,
    MTGODEConfig,
    MTGODEModel,
    STAEFormerConfig,
    STAEFormerModel,
)


DATASET = "beijing-air"
WORKSPACE = "./benchmark_workspace"


def _one_epoch(cfg: Any) -> Any:
    """Pin every trainable Config to a single training epoch."""
    return replace(cfg, max_epochs=1, early_stop=10**6)


CASES: list[tuple[str, Callable[[], BenchmarkModel], Any]] = [
    ("LastValue",        lambda: LastValueModel(),        None),
    ("MLPMultivariate",  lambda: MLPMultivariateModel(),  _one_epoch(MLPMultivariateConfig())),
    ("ASTGCN",           lambda: ASTGCNModel(),           _one_epoch(ASTGCNConfig())),
    ("GWN",              lambda: GWNModel(),              _one_epoch(GWNConfig())),
    ("MTGNN",            lambda: MTGNNModel(),            _one_epoch(MTGNNConfig())),
    ("MTGODE",           lambda: MTGODEModel(),           _one_epoch(MTGODEConfig())),
    ("STAEFormer",       lambda: STAEFormerModel(),       _one_epoch(STAEFormerConfig())),
]


def main() -> int:
    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=[DATASET])

    statuses: list[tuple[str, str, str]] = []

    for label, factory, cfg in CASES:
        print(f"\n[smoke] === {label} on {DATASET} ===")
        try:
            result = runner.run(factory(), config=cfg)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            statuses.append((label, "FAIL", f"{type(exc).__name__}: {exc}"))
            continue

        dr = result.dataset_results.get(DATASET)
        if dr is None or dr.error:
            err = dr.error if dr else "no result returned"
            statuses.append((label, "FAIL", err))
        else:
            statuses.append(
                (label, "PASS", f"MAE={dr.mae:.4f} RMSE={dr.rmse:.4f}")
            )

    print("\n[smoke] === Summary ===")
    width = max(len(label) for label, _, _ in statuses)
    failed = 0
    for label, status, info in statuses:
        if status == "FAIL":
            failed += 1
        print(f"  [{status}] {label:<{width}}  {info}")

    if failed:
        print(f"\n[smoke] {failed}/{len(statuses)} model(s) FAILED on {DATASET}.")
        return 1
    print(f"\n[smoke] All {len(statuses)} models passed on {DATASET}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
