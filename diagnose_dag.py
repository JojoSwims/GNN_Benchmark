#!/usr/bin/env python3
"""
Diagnostic script for the LamaH-CE river network DAG.

Checks for isolated nodes, disconnected components, missing edges,
and compares raw hierarchy data against the filtered graph.

Usage:
    python diagnose_dag.py --data-root datasets/2_LamaH-CE_daily
"""

import argparse
from pathlib import Path

import pandas as pd


def main(data_root: Path, max_gap_fraction: float):
    print("=" * 70)
    print("LamaH-CE River Network DAG — Diagnostic Report")
    print("=" * 70)

    # ── Load raw data ────────────────────────────────────────────────────
    ga = pd.read_csv(
        data_root / "D_gauges" / "1_attributes" / "Gauge_attributes.csv",
        sep=";",
    )
    ga.columns = ga.columns.str.strip()
    ga["ID"] = ga["ID"].astype(int)
    print(f"\n1. RAW DATA")
    print(f"   D_gauges total gauges: {len(ga)}")

    hier = pd.read_csv(
        data_root / "B_basins_intermediate_all" / "1_attributes" / "Gauge_hierarchy.csv",
        sep=";",
    )
    hier.columns = hier.columns.str.strip()
    hier["ID"] = hier["ID"].astype(int)
    hier["NEXTDOWNID"] = pd.to_numeric(hier["NEXTDOWNID"], errors="coerce")
    hier["HIERARCHY"] = pd.to_numeric(hier["HIERARCHY"], errors="coerce")
    print(f"   B_basins hierarchy entries: {len(hier)}")

    dist_path = data_root / "B_basins_intermediate_all" / "1_attributes" / "Stream_dist.csv"
    dist_df = pd.read_csv(dist_path, sep=";") if dist_path.exists() else None
    if dist_df is not None:
        dist_df.columns = dist_df.columns.str.strip()
        dist_df["ID"] = dist_df["ID"].astype(int)
    print(f"   Stream_dist entries: {len(dist_df) if dist_df is not None else 'N/A'}")

    # ── Gap filter ───────────────────────────────────────────────────────
    print(f"\n2. GAP FILTER (max_gap_fraction={max_gap_fraction})")
    all_gauge_ids = set(ga["ID"].tolist())
    if "gaps_post" in ga.columns:
        passed = ga[ga["gaps_post"] <= max_gap_fraction]
        failed = ga[ga["gaps_post"] > max_gap_fraction]
        print(f"   Passed gap filter: {len(passed)}")
        print(f"   Failed gap filter: {len(failed)}")
    else:
        passed = ga
        failed = pd.DataFrame()
        print(f"   No gaps_post column — all {len(ga)} pass")

    valid_gauge_ids = set(passed["ID"].tolist())

    # ── B-basin coverage ─────────────────────────────────────────────────
    print(f"\n3. B-BASIN COVERAGE")
    b_ids = set(hier["ID"].tolist())
    gauges_with_b = valid_gauge_ids & b_ids
    gauges_without_b = valid_gauge_ids - b_ids
    print(f"   Valid gauges with B-basin entry: {len(gauges_with_b)}")
    print(f"   Valid gauges WITHOUT B-basin entry (non-delineated): {len(gauges_without_b)}")
    if gauges_without_b:
        print(f"   Non-delineated IDs: {sorted(gauges_without_b)}")

    # After dropping non-delineated (same as lamah_ce.py logic)
    final_node_ids = gauges_with_b
    print(f"\n   Final node set (after dropping non-delineated): {len(final_node_ids)}")

    # ── Met file coverage ────────────────────────────────────────────────
    print(f"\n4. MET FILE COVERAGE")
    met_dir = data_root / "B_basins_intermediate_all" / "2_timeseries" / "daily"
    if met_dir.exists():
        met_files = {int(f.stem.split("_")[1]) for f in met_dir.glob("ID_*.csv")}
        nodes_with_met = final_node_ids & met_files
        nodes_without_met = final_node_ids - met_files
        print(f"   Met files found: {len(met_files)}")
        print(f"   Final nodes with met file: {len(nodes_with_met)}")
        print(f"   Final nodes WITHOUT met file: {len(nodes_without_met)}")
        if nodes_without_met:
            print(f"   Missing met IDs: {sorted(nodes_without_met)}")
    else:
        print(f"   Met directory not found: {met_dir}")

    # ── Hierarchy analysis ───────────────────────────────────────────────
    print(f"\n5. HIERARCHY STRUCTURE (raw, all 859 B-basin entries)")
    id_to_next: dict[int, int] = {}
    for _, row in hier.iterrows():
        nxt = row["NEXTDOWNID"]
        if pd.notna(nxt) and int(nxt) > 0:
            id_to_next[int(row["ID"])] = int(nxt)

    outlets_raw = set(hier["ID"].tolist()) - set(id_to_next.keys())
    nextdown_zero = hier[hier["NEXTDOWNID"].fillna(0).astype(int) <= 0]
    print(f"   Gauges with NEXTDOWNID > 0: {len(id_to_next)}")
    print(f"   Gauges with NEXTDOWNID <= 0 or NaN (outlets): {len(nextdown_zero)}")
    print(f"   Outlet IDs: {sorted(nextdown_zero['ID'].tolist())}")

    hier_counts = hier["HIERARCHY"].value_counts().sort_index()
    print(f"\n   Hierarchy depth distribution:")
    for depth, count in hier_counts.items():
        bar = "#" * min(int(count / 5), 60)
        print(f"     depth {int(depth):2d}: {count:4d} gauges  {bar}")

    # ── Edge building with look-ahead (replicate lamah_ce.py logic) ──────
    print(f"\n6. EDGE BUILDING (with transitive look-ahead)")

    edges = []
    bridged = 0
    no_downstream = 0
    isolated = 0

    for src in sorted(final_node_ids):
        if src not in id_to_next:
            isolated += 1
            continue

        cursor = id_to_next[src]
        visited = {src}
        hops = 0

        while cursor > 0 and cursor not in final_node_ids:
            if cursor in visited:
                break
            visited.add(cursor)
            nxt = id_to_next.get(cursor)
            if nxt is None:
                cursor = 0
                break
            cursor = nxt
            hops += 1

        if cursor > 0 and cursor in final_node_ids:
            edges.append((src, cursor, hops))
            if hops > 0:
                bridged += 1
        else:
            no_downstream += 1

    nodes_with_outgoing = {e[0] for e in edges}
    nodes_with_incoming = {e[1] for e in edges}
    nodes_in_edges = nodes_with_outgoing | nodes_with_incoming
    truly_isolated = final_node_ids - nodes_in_edges

    print(f"   Total edges: {len(edges)}")
    print(f"   Direct edges (0 hops): {len(edges) - bridged}")
    print(f"   Bridged edges (skipped filtered nodes): {bridged}")
    if bridged > 0:
        hop_counts = {}
        for _, _, h in edges:
            if h > 0:
                hop_counts[h] = hop_counts.get(h, 0) + 1
        for h in sorted(hop_counts):
            print(f"     {hop_counts[h]} edges bridged over {h} node(s)")

    print(f"   Nodes with no NEXTDOWNID (hierarchy outlets): {isolated}")
    print(f"   Nodes whose downstream chain ends outside valid set: {no_downstream}")

    print(f"\n7. CONNECTIVITY ANALYSIS")
    print(f"   Nodes participating in at least one edge: {len(nodes_in_edges)}")
    print(f"   Truly isolated nodes (no edges at all): {len(truly_isolated)}")
    print(f"   Isolated fraction: {len(truly_isolated) / len(final_node_ids) * 100:.1f}%")

    if truly_isolated:
        print(f"\n   Isolated node details:")
        for gid in sorted(truly_isolated):
            reasons = []
            if gid not in id_to_next:
                reasons.append("outlet (NEXTDOWNID=0)")
            else:
                reasons.append(f"NEXTDOWNID={id_to_next[gid]}")
                # Check if anyone points TO this node
                upstream_of = [k for k, v in id_to_next.items() if v == gid]
                if not upstream_of:
                    reasons.append("headwater (no upstream gauge)")
                else:
                    valid_upstream = [u for u in upstream_of if u in final_node_ids]
                    filtered_upstream = [u for u in upstream_of if u not in final_node_ids]
                    if valid_upstream:
                        reasons.append(f"has valid upstream: {valid_upstream}")
                    if filtered_upstream:
                        reasons.append(f"upstream filtered out: {filtered_upstream}")

            h = hier[hier["ID"] == gid]["HIERARCHY"].values
            h_str = f"hierarchy={int(h[0])}" if len(h) > 0 else "no hierarchy"
            print(f"     ID {gid:4d} ({h_str}): {', '.join(reasons)}")

    # ── Connected components ─────────────────────────────────────────────
    print(f"\n8. CONNECTED COMPONENTS (treating edges as undirected)")
    from collections import deque

    adj: dict[int, set[int]] = {n: set() for n in final_node_ids}
    for src, dst, _ in edges:
        adj[src].add(dst)
        adj[dst].add(src)

    visited_global: set[int] = set()
    components: list[set[int]] = []
    for node in sorted(final_node_ids):
        if node in visited_global:
            continue
        comp: set[int] = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if n in comp:
                continue
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in comp:
                    queue.append(nb)
        components.append(comp)
        visited_global |= comp

    components.sort(key=len, reverse=True)
    print(f"   Total components: {len(components)}")
    print(f"   Largest component: {len(components[0])} nodes")
    print(f"   Component size distribution:")
    size_counts: dict[int, int] = {}
    for c in components:
        s = len(c)
        size_counts[s] = size_counts.get(s, 0) + 1
    for size in sorted(size_counts.keys(), reverse=True):
        count = size_counts[size]
        print(f"     size {size:4d}: {count} component(s)")

    # ── Filtered-out nodes that broke edges ──────────────────────────────
    print(f"\n9. IMPACT OF GAP FILTER ON CONNECTIVITY")
    failed_ids = set(failed["ID"].tolist()) if not failed.empty else set()
    failed_in_b = failed_ids & b_ids
    print(f"   Gauges removed by gap filter: {len(failed_ids)}")
    print(f"   Of those, in B-basin hierarchy: {len(failed_in_b)}")

    # Check how many were "bridge" nodes (had both upstream and downstream)
    bridge_nodes = 0
    for fid in failed_in_b:
        has_downstream = fid in id_to_next
        has_upstream = any(v == fid for v in id_to_next.values())
        if has_downstream and has_upstream:
            bridge_nodes += 1
    print(f"   Bridge nodes removed (had both up+downstream): {bridge_nodes}")
    print(f"   (These are the nodes the look-ahead bridges over)")

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose LamaH-CE DAG")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-gap-fraction", type=float, default=0.05)
    args = parser.parse_args()
    main(args.data_root, args.max_gap_fraction)
