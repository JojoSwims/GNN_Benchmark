#!/usr/bin/env python3
"""Smoke test: ``LastValueModel`` runs end-to-end on every registered dataset.

No hyperparameter tuning. The runner iterates ``DATASET_REGISTRY`` and
catches per-dataset failures so one bad loader doesn't abort the rest.

Usage:
    python test_lastvalue_all_datasets.py

Exits non-zero if any dataset fails.
"""

from __future__ import annotations

import sys

from gnn_benchmark.benchmark import DATASET_REGISTRY, BenchmarkRunner
from gnn_benchmark.models import LastValueModel


WORKSPACE = "./benchmark_workspace"


def main() -> int:
    dataset_keys = list(DATASET_REGISTRY.keys())
    print(
        f"[smoke] LastValue on {len(dataset_keys)} dataset(s): "
        f"{', '.join(dataset_keys)}"
    )

    runner = BenchmarkRunner(workspace_dir=WORKSPACE, datasets=dataset_keys)
    result = runner.run(LastValueModel(), config=None)

    print("\n[smoke] === Summary ===")
    width = max(len(k) for k in dataset_keys)
    failed = 0
    for key in dataset_keys:
        dr = result.dataset_results.get(key)
        if dr is None:
            failed += 1
            print(f"  [FAIL] {key:<{width}}  no result returned")
            continue
        if dr.error:
            failed += 1
            print(f"  [FAIL] {key:<{width}}  {dr.error}")
        else:
            print(
                f"  [PASS] {key:<{width}}  "
                f"MAE={dr.mae:.4f} RMSE={dr.rmse:.4f} MAPE={dr.mape:.2f}"
            )

    if failed:
        print(
            f"\n[smoke] {failed}/{len(dataset_keys)} dataset(s) FAILED for "
            "LastValue."
        )
        return 1
    print(f"\n[smoke] LastValue passed on all {len(dataset_keys)} datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
