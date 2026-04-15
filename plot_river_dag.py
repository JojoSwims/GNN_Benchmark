#!/usr/bin/env python3
"""
Visualise the LamaH-CE river network as a directed acyclic graph.

Nodes are plotted at their real geographic positions (EPSG:3035) with
directed edges showing upstream → downstream flow.

Usage:
    python plot_river_dag.py --data-root datasets/2_LamaH-CE_daily

Options:
    --max-gap-fraction  Gap filter threshold (default 0.05)
    --color-by          Node colouring: "hierarchy" | "region" | "elevation"
                        (default: hierarchy)
    --save              Save to PNG instead of showing interactively
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd


def load_data(data_root: Path, max_gap_fraction: float):
    """Load gauge attrs, catchment attrs, hierarchy and stream distances."""

    # ── Gauge attributes (positions, gap filter) ─────────────────────────
    ga = pd.read_csv(
        data_root / "D_gauges" / "1_attributes" / "Gauge_attributes.csv",
        sep=";",
    )
    ga.columns = ga.columns.str.strip()
    ga["ID"] = ga["ID"].astype(int)

    if "gaps_post" in ga.columns:
        ga = ga[ga["gaps_post"] <= max_gap_fraction].copy()

    # ── Catchment attributes (elevation for colouring) ───────────────────
    ca_path = (
        data_root
        / "B_basins_intermediate_all"
        / "1_attributes"
        / "Catchment_attributes.csv"
    )
    ca = pd.read_csv(ca_path, sep=";")
    ca.columns = ca.columns.str.strip()
    ca["ID"] = ca["ID"].astype(int)

    # ── Hierarchy (edges + depth) ────────────────────────────────────────
    hier = pd.read_csv(
        data_root
        / "B_basins_intermediate_all"
        / "1_attributes"
        / "Gauge_hierarchy.csv",
        sep=";",
    )
    hier.columns = hier.columns.str.strip()
    hier["ID"] = hier["ID"].astype(int)
    hier["NEXTDOWNID"] = pd.to_numeric(hier["NEXTDOWNID"], errors="coerce")
    hier["HIERARCHY"] = pd.to_numeric(hier["HIERARCHY"], errors="coerce")

    # ── Stream distances ─────────────────────────────────────────────────
    dist_path = (
        data_root
        / "B_basins_intermediate_all"
        / "1_attributes"
        / "Stream_dist.csv"
    )
    dist_df = None
    if dist_path.exists():
        dist_df = pd.read_csv(dist_path, sep=";")
        dist_df.columns = dist_df.columns.str.strip()
        dist_df["ID"] = dist_df["ID"].astype(int)
        dist_df["dist_hdn"] = pd.to_numeric(
            dist_df["dist_hdn"], errors="coerce"
        ).fillna(1.0)

    return ga, ca, hier, dist_df


def build_graph(ga, ca, hier, dist_df, color_by):
    """Build a networkx DiGraph with positions and colours."""

    valid_ids = set(ga["ID"].tolist())

    # Only keep gauges that also have a B-basin entry (excludes 23 non-delineated)
    b_ids = set(hier["ID"].tolist())
    valid_ids = valid_ids & b_ids

    # Build lookup tables
    ga_idx = ga.set_index("ID")
    ca_idx = ca.set_index("ID")
    hier_idx = hier.set_index("ID")

    id_to_dist = {}
    if dist_df is not None:
        id_to_dist = dict(zip(dist_df["ID"], dist_df["dist_hdn"]))

    # ── Transitive look-ahead (same logic as lamah_ce.py) ────────────────
    id_to_next: dict[int, int] = {}
    for _, row in hier.iterrows():
        nxt = row["NEXTDOWNID"]
        if pd.notna(nxt) and int(nxt) > 0:
            id_to_next[int(row["ID"])] = int(nxt)

    G = nx.DiGraph()
    pos = {}
    node_colors = []
    edge_colors = []

    # Add nodes
    for gid in sorted(valid_ids):
        G.add_node(gid)

        # Position from gauge lon/lat (EPSG:3035)
        if gid in ga_idx.index:
            x = float(ga_idx.loc[gid, "lon"])
            y = float(ga_idx.loc[gid, "lat"])
            pos[gid] = (x, y)
        else:
            pos[gid] = (0, 0)

        # Node colour value
        if color_by == "hierarchy" and gid in hier_idx.index:
            node_colors.append(float(hier_idx.loc[gid, "HIERARCHY"]))
        elif color_by == "elevation" and gid in ca_idx.index:
            node_colors.append(float(ca_idx.loc[gid, "elev_mean"]))
        elif color_by == "region" and gid in ga_idx.index:
            node_colors.append(float(ga_idx.loc[gid, "region"]))
        else:
            node_colors.append(0.0)

    # Add edges with transitive look-ahead
    for src in sorted(valid_ids):
        if src not in id_to_next:
            continue

        cursor = id_to_next[src]
        curr_dist = id_to_dist.get(src, 1.0)
        visited = {src}

        while cursor > 0 and cursor not in valid_ids:
            if cursor in visited:
                break
            visited.add(cursor)
            curr_dist += id_to_dist.get(cursor, 1.0)
            nxt = id_to_next.get(cursor)
            if nxt is None:
                cursor = 0
                break
            cursor = nxt

        if cursor > 0 and cursor in valid_ids:
            G.add_edge(src, cursor, dist=curr_dist)
            edge_colors.append(curr_dist)

    return G, pos, node_colors, edge_colors


def plot_dag(G, pos, node_colors, edge_colors, color_by, save_path=None):
    """Render the river network."""

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#f0f4f8")

    # ── Edges ────────────────────────────────────────────────────────────
    if edge_colors:
        edge_norm = mcolors.Normalize(
            vmin=min(edge_colors), vmax=max(edge_colors)
        )
        edge_cmap = cm.Blues
        edge_rgba = [edge_cmap(edge_norm(c)) for c in edge_colors]
    else:
        edge_rgba = "gray"

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_rgba,
        width=0.6,
        alpha=0.5,
        arrows=True,
        arrowsize=4,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.05",
    )

    # ── Nodes ────────────────────────────────────────────────────────────
    color_label = {
        "hierarchy": "Hierarchy depth",
        "elevation": "Elevation (m a.s.l.)",
        "region": "River region",
    }.get(color_by, color_by)

    node_cmap = cm.viridis
    node_norm = mcolors.Normalize(
        vmin=min(node_colors), vmax=max(node_colors)
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=12,
        node_color=node_colors,
        cmap=node_cmap,
        vmin=node_norm.vmin,
        vmax=node_norm.vmax,
        edgecolors="black",
        linewidths=0.2,
    )

    # ── Colour bars ──────────────────────────────────────────────────────
    cbar_node = plt.colorbar(
        cm.ScalarMappable(norm=node_norm, cmap=node_cmap),
        ax=ax,
        fraction=0.025,
        pad=0.02,
        label=color_label,
    )

    if edge_colors:
        cbar_edge = plt.colorbar(
            cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap),
            ax=ax,
            fraction=0.025,
            pad=0.06,
            label="Flow-path distance (km)",
        )

    # ── Stats annotation ─────────────────────────────────────────────────
    n_components = nx.number_weakly_connected_components(G)
    stats = (
        f"Nodes: {G.number_of_nodes()}  |  "
        f"Edges: {G.number_of_edges()}  |  "
        f"Components: {n_components}"
    )
    ax.set_title(
        f"LamaH-CE River Network DAG\n{stats}",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Easting (EPSG:3035)", fontsize=10)
    ax.set_ylabel("Northing (EPSG:3035)", fontsize=10)
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise LamaH-CE river network DAG"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to extracted LamaH-CE root directory",
    )
    parser.add_argument(
        "--max-gap-fraction",
        type=float,
        default=0.05,
        help="Gap filter threshold (default: 0.05)",
    )
    parser.add_argument(
        "--color-by",
        choices=["hierarchy", "region", "elevation"],
        default="hierarchy",
        help="Node colouring scheme (default: hierarchy)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save to file instead of showing (e.g. river_dag.png)",
    )
    args = parser.parse_args()

    print("Loading data...")
    ga, ca, hier, dist_df = load_data(args.data_root, args.max_gap_fraction)

    print("Building graph...")
    G, pos, node_colors, edge_colors = build_graph(
        ga, ca, hier, dist_df, args.color_by
    )

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Weakly connected components: "
          f"{nx.number_weakly_connected_components(G)}")

    print("Plotting...")
    plot_dag(G, pos, node_colors, edge_colors, args.color_by, args.save)


if __name__ == "__main__":
    main()
