#!/usr/bin/env python3
"""
Test script for LamaHCELoader.

Usage:
    python test_lamah.py
    python test_lamah.py --data-root /path/to/extracted/LamaH-CE

The data_root should be the folder that contains D_gauges/, A_basins_total_upstrm/, etc.
When --data-root is omitted in normal loader usage, LamaHCELoader downloads the
Google Drive mirror with gdown and caches it under ~/.cache/gnn_benchmark/lamah_ce/.
"""

import argparse
import sys
import types
from pathlib import Path

# Bootstrap: register a lightweight gnn_benchmark package stub so that
# submodule imports (core.workspace, datasets.lamah_ce) work without
# triggering the root __init__.py, which pulls in torch via models/.
_pkg = types.ModuleType("gnn_benchmark")
_pkg.__path__ = [str(Path(__file__).parent)]
_pkg.__package__ = "gnn_benchmark"
sys.modules.setdefault("gnn_benchmark", _pkg)

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.datasets.lamah_ce import LamaHCELoader


def main(data_root: Path | None, resolution: str, max_gap_fraction: float) -> None:
    print("=" * 60)
    print("LamaH-CE Dataset Test")
    print("=" * 60)
    print(f"  data_root        : {data_root}")
    print(f"  resolution       : {resolution}")
    print(f"  max_gap_fraction : {max_gap_fraction}")
    print()

    loader = LamaHCELoader(
        data_root=data_root,
        resolution=resolution,
        max_gap_fraction=max_gap_fraction,
    )

    workspace = DataWorkspace(Path("./workspace_lamah_test"))
    print("Preparing dataset (may take a few minutes on first run)...")
    ir = loader.prepare(workspace)

    print()
    print("=" * 60)
    print("IR Summary")
    print("=" * 60)
    print(ir)

    T, N, C = ir.shape
    print(f"  T (timesteps) : {T}")
    print(f"  N (nodes)     : {N}")
    print(f"  C (features)  : {C}")
    print(f"  feature cols  : {ir.metadata.feature_columns}")
    print(f"  frequency     : {ir.metadata.frequency}")
    print()

    # ── Date range ──────────────────────────────────────────────────────────
    print("Date range:")
    print(f"  first : {ir.series['ts'].min()}")
    print(f"  last  : {ir.series['ts'].max()}")
    print()

    # ── Sample rows ─────────────────────────────────────────────────────────
    print("Sample rows (first 10):")
    print(ir.series.head(10).to_string(index=False))
    print()

    # ── Missing value rate ───────────────────────────────────────────────────
    missing_pct = ir.series["qobs"].isna().mean() * 100
    print(f"Overall missing qobs: {missing_pct:.2f}%")
    print()

    # ── Value range (sanity check units: m3/s) ───────────────────────────────
    print("qobs statistics (m3/s):")
    print(ir.series["qobs"].describe().round(3).to_string())
    print()

    # ── Nodes ────────────────────────────────────────────────────────────────
    print(f"First 10 node IDs : {ir.metadata.node_order[:10]}")
    print(f"Last  10 node IDs : {ir.metadata.node_order[-10:]}")
    print()

    # ── Edges ────────────────────────────────────────────────────────────────
    if ir.edges is not None:
        print(f"Edges: {len(ir.edges)} directed edges")
        print("Sample edges (first 10):")
        print(ir.edges.head(10).to_string(index=False))
        print()
        print("Edge cost (km) statistics:")
        print(ir.edges["cost"].describe().round(2).to_string())
    else:
        print("Edges: None")
    print()

    # ── Adjacency matrix shape check ─────────────────────────────────────────
    if ir.edges is not None:
        adj = ir.get_adjacency_matrix()
        print(f"Adjacency matrix shape : {adj.shape}")
        nonzero = (adj > 0).sum()
        print(f"Non-zero entries       : {nonzero} / {adj.size}")
    print()

    print("=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LamaHCELoader")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to extracted LamaH-CE root directory (contains D_gauges/, "
            "etc.). Omit to download the Google Drive mirror with gdown."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=["daily", "hourly"],
        default="daily",
        help="Time series resolution (default: daily)",
    )
    parser.add_argument(
        "--max-gap-fraction",
        type=float,
        default=0.05,
        help="Max allowed gap fraction per gauge (default: 0.05)",
    )
    args = parser.parse_args()
    main(args.data_root, args.resolution, args.max_gap_fraction)
