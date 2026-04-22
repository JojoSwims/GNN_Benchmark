#!/usr/bin/env python3
"""Exploratory analysis of the NY COVID-19 county-level dataset.

This script does NOT train a model. It loads the prepared NY_Covid IR
(already 7-day smoothed and 14-day lag-aligned: each row pairs deaths at
day t with cases from day t-14) and generates a set of figures + a
printed summary aimed at the questions you need to answer before
modelling:

    - When does the dataset actually start / end per county?
    - How many counties are reporting on any given day?
    - How long are the runs of zero cases / deaths per county?
      (long zero stretches are the main worry for small counties.)
    - What fraction of days are zeros per county?
    - How skewed is the per-county total? (a few huge counties dominate.)
    - Day-of-week pattern — should be flat after 7-day smoothing.
    - Residual single-day spikes after smoothing.
    - A few representative county time series for a sanity check.
    - Graph connectivity as a function of distance cutoff — what's the
      smallest cutoff that leaves no isolated nodes, what's the smallest
      cutoff that makes the graph fully connected, and every cutoff
      between those two at which the number of connected components
      drops by one.
    - A diagnostics section flagging major issues that would make
      training models on this dataset problematic.

Outputs go to ``./ny_covid_explore_out/`` as PNGs plus a text summary.

Usage:
    python examples/ny_covid_explore.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.datasets.ny_covid import NYCovidLoader

# ── Config ────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path("./benchmark_workspace")
OUT_DIR = Path("./ny_covid_explore_out")
NYC_FIPS = "99999"

# Column names in the aligned IR. Cases are already shifted back 14 days
# at the loader level, so CASES_COL[t] = smoothed raw cases at day t-14.
CASES_COL = "new_cases_smooth_lag14"
DEATHS_COL = "new_deaths_smooth"

# Top-K counties (by total cases) plotted individually
TOP_K_SAMPLE = 3
# Bottom-K counties (by total cases, excluding all-zero) plotted individually
BOT_K_SAMPLE = 3
# Spike threshold: a day's value exceeding (median * MULT) of that county.
# After 7-day smoothing real spikes > 50x median are essentially impossible,
# so any such spike is a diagnostic red flag rather than normal behaviour.
SPIKE_MULT = 50.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def longest_zero_run(values: np.ndarray) -> int:
    """Length of the longest contiguous run of zeros in `values`."""
    if values.size == 0:
        return 0
    is_zero = values == 0
    # Run-length via change-points
    changes = np.diff(is_zero.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def per_county_stats(
    panel: pd.DataFrame, col: str
) -> pd.DataFrame:
    """Per-county stats for one feature column.

    Returns a DataFrame indexed by node_id with columns:
      total, mean, max, zero_frac, longest_zero_run, first_nonzero, last_nonzero
    """
    rows: list[dict] = []
    for nid, sub in panel.groupby("node_id", sort=True):
        v = sub[col].to_numpy()
        nonzero_idx = np.flatnonzero(v > 0)
        if nonzero_idx.size:
            first = sub["ts"].iloc[int(nonzero_idx[0])]
            last = sub["ts"].iloc[int(nonzero_idx[-1])]
        else:
            first = pd.NaT
            last = pd.NaT
        rows.append(
            {
                "node_id": nid,
                "total": float(v.sum()),
                "mean": float(v.mean()),
                "max": float(v.max()),
                "zero_frac": float((v == 0).mean()),
                "longest_zero_run": longest_zero_run(v),
                "first_nonzero": first,
                "last_nonzero": last,
            }
        )
    return pd.DataFrame(rows).set_index("node_id")


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_aggregate_daily(daily: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(daily.index, daily[CASES_COL], color="C0", linewidth=0.9)
    axes[0].set_ylabel("Total smoothed cases / day (14-day lag)")
    axes[0].set_title("Aggregate smoothed cases (sum across all counties)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(daily.index, daily[DEATHS_COL], color="C3", linewidth=0.9)
    axes[1].set_ylabel("Total smoothed deaths / day")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Aggregate smoothed deaths (sum across all counties)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reporting_coverage(panel: pd.DataFrame, out_path: Path) -> None:
    """Number of counties with non-zero smoothed cases per day."""
    coverage = (
        panel.assign(reporting=(panel[CASES_COL] > 0).astype(int))
        .groupby("ts")["reporting"]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(coverage.index, coverage.values, color="C2", linewidth=0.9)
    ax.set_ylabel(f"# counties with {CASES_COL} > 0")
    ax.set_xlabel("Date")
    ax.set_title("Reporting coverage over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_zero_fraction_hist(
    cases_stats: pd.DataFrame, deaths_stats: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cases_stats["zero_frac"], bins=40, color="C0", edgecolor="black")
    axes[0].set_title(f"Per-county fraction of days with {CASES_COL} = 0")
    axes[0].set_xlabel("Zero-day fraction")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(deaths_stats["zero_frac"], bins=40, color="C3", edgecolor="black")
    axes[1].set_title(f"Per-county fraction of days with {DEATHS_COL} = 0")
    axes[1].set_xlabel("Zero-day fraction")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_longest_zero_streak_hist(
    cases_stats: pd.DataFrame, deaths_stats: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(
        cases_stats["longest_zero_run"], bins=40, color="C0", edgecolor="black"
    )
    axes[0].set_title(f"Longest contiguous run of {CASES_COL} = 0 per county")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        deaths_stats["longest_zero_run"], bins=40, color="C3", edgecolor="black"
    )
    axes[1].set_title(f"Longest contiguous run of {DEATHS_COL} = 0 per county")
    axes[1].set_xlabel("Days")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_total_per_county_hist(
    cases_stats: pd.DataFrame, deaths_stats: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(
        np.log10(cases_stats["total"].clip(lower=1.0)),
        bins=40,
        color="C0",
        edgecolor="black",
    )
    axes[0].set_title(f"Per-county total {CASES_COL} (log10)")
    axes[0].set_xlabel("log10(total cases)")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        np.log10(deaths_stats["total"].clip(lower=1.0)),
        bins=40,
        color="C3",
        edgecolor="black",
    )
    axes[1].set_title(f"Per-county total {DEATHS_COL} (log10)")
    axes[1].set_xlabel("log10(total deaths)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dow_pattern(panel: pd.DataFrame, out_path: Path) -> None:
    dow = panel["ts"].dt.day_name()
    by_dow = (
        panel.assign(dow=dow)
        .groupby("dow")[[CASES_COL, DEATHS_COL]]
        .mean()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(by_dow.index, by_dow[CASES_COL], color="C0", edgecolor="black")
    axes[0].set_title(f"Mean {CASES_COL} per (county, day) by weekday")
    axes[0].set_ylabel("Mean")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(by_dow.index, by_dow[DEATHS_COL], color="C3", edgecolor="black")
    axes[1].set_title(f"Mean {DEATHS_COL} per (county, day) by weekday")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sample_counties(
    panel: pd.DataFrame, sample_ids: list[str], out_path: Path
) -> None:
    fig, axes = plt.subplots(
        len(sample_ids), 1, figsize=(12, 2.2 * len(sample_ids)), sharex=True
    )
    if len(sample_ids) == 1:
        axes = [axes]
    for ax, nid in zip(axes, sample_ids):
        sub = panel[panel["node_id"] == nid].sort_values("ts")
        ax.plot(sub["ts"], sub[CASES_COL], color="C0", linewidth=0.7, label="cases (t-14)")
        ax.plot(sub["ts"], sub[DEATHS_COL], color="C3", linewidth=0.7, label="deaths")
        ax.set_ylabel(f"FIPS {nid}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Sample county time series (smoothed, cases lagged 14d)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spike_distribution(
    panel: pd.DataFrame, out_path: Path
) -> None:
    """For each county compute max/median ratio (excl. zero medians)."""
    ratios: list[float] = []
    for _, sub in panel.groupby("node_id", sort=False):
        v = sub[CASES_COL].to_numpy()
        nz = v[v > 0]
        if nz.size < 10:
            continue
        med = np.median(nz)
        if med <= 0:
            continue
        ratios.append(float(v.max() / med))
    fig, ax = plt.subplots(figsize=(8, 4))
    if ratios:
        ax.hist(np.log10(ratios), bins=40, color="C4", edgecolor="black")
    ax.set_title(f"Per-county max({CASES_COL}) / median(non-zero) (log10)")
    ax.set_xlabel("log10(spike ratio)")
    ax.set_ylabel("# counties")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    workspace = DataWorkspace(WORKSPACE_DIR)
    loader = NYCovidLoader()
    print("Preparing NY COVID IR (this downloads on first run)...")
    ir = loader.prepare(workspace)

    panel = ir.series.copy()
    panel["ts"] = pd.to_datetime(panel["ts"])
    panel["node_id"] = panel["node_id"].astype(str)

    nodes = ir.nodes
    T = panel["ts"].nunique()
    N = len(nodes)
    n_edges = 0 if ir.edges is None else len(ir.edges)

    # ── Aggregate daily series ────────────────────────────────────────────────
    daily = panel.groupby("ts")[[CASES_COL, DEATHS_COL]].sum().sort_index()

    # ── Per-county stats ──────────────────────────────────────────────────────
    print("Computing per-county statistics...")
    cases_stats = per_county_stats(panel, CASES_COL)
    deaths_stats = per_county_stats(panel, DEATHS_COL)

    # Pick sample counties: top-K, bottom-K (>0), and NYC if present
    by_total = cases_stats["total"].sort_values()
    nonzero = by_total[by_total > 0]
    sample_ids: list[str] = []
    sample_ids += list(by_total.index[-TOP_K_SAMPLE:][::-1])  # top-K desc
    sample_ids += list(nonzero.index[:BOT_K_SAMPLE])  # bottom-K
    if NYC_FIPS in cases_stats.index and NYC_FIPS not in sample_ids:
        sample_ids.append(NYC_FIPS)
    # Dedup, preserve order
    seen = set()
    sample_ids = [x for x in sample_ids if not (x in seen or seen.add(x))]

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Writing plots to", OUT_DIR.resolve())
    plot_aggregate_daily(daily, OUT_DIR / "01_aggregate_daily.png")
    plot_reporting_coverage(panel, OUT_DIR / "02_reporting_coverage.png")
    plot_zero_fraction_hist(
        cases_stats, deaths_stats, OUT_DIR / "03_zero_fraction_hist.png"
    )
    plot_longest_zero_streak_hist(
        cases_stats, deaths_stats, OUT_DIR / "04_longest_zero_streak_hist.png"
    )
    plot_total_per_county_hist(
        cases_stats, deaths_stats, OUT_DIR / "05_total_per_county_hist.png"
    )
    plot_dow_pattern(panel, OUT_DIR / "06_dow_pattern.png")
    plot_sample_counties(panel, sample_ids, OUT_DIR / "07_sample_counties.png")
    plot_spike_distribution(panel, OUT_DIR / "08_spike_distribution.png")

    # ── Connectivity analysis ────────────────────────────────────────────────
    print("Running connectivity analysis on distance graph...")
    conn = connectivity_analysis(ir.edges, nodes)
    plot_connectivity_curve(conn, OUT_DIR / "09_connectivity_vs_cutoff.png")
    plot_isolation_curve(
        conn["first_neighbor_dist"], OUT_DIR / "10_isolated_nodes_vs_cutoff.png"
    )

    # ── Per-feature stats summary ────────────────────────────────────────────
    n_all_zero_cases = int((cases_stats["total"] == 0).sum())
    n_all_zero_deaths = int((deaths_stats["total"] == 0).sum())
    p99_zero_cases = float(cases_stats["longest_zero_run"].quantile(0.99))
    p99_zero_deaths = float(deaths_stats["longest_zero_run"].quantile(0.99))

    worst_cases_zeros = cases_stats["longest_zero_run"].sort_values(ascending=False).head(10)
    worst_deaths_zeros = deaths_stats["longest_zero_run"].sort_values(ascending=False).head(10)

    # ── Diagnostics: major issues that would bite during training ────────────
    spike_counties = detect_spike_counties(panel, SPIKE_MULT)
    dense_edge_ratio = n_edges / max(N * (N - 1), 1)
    diagnostics = run_diagnostics(
        T=T,
        N=N,
        n_edges=n_edges,
        dense_edge_ratio=dense_edge_ratio,
        cases_stats=cases_stats,
        deaths_stats=deaths_stats,
        spike_counties=spike_counties,
        panel=panel,
    )

    # ── Text summary ──────────────────────────────────────────────────────────
    lines = [
        "NY_Covid exploratory summary",
        "=" * 60,
        f"Days (T):              {T}",
        f"Counties (N):          {N}",
        f"Edges (directed):      {n_edges}",
        f"Date range:            {panel['ts'].min().date()} → {panel['ts'].max().date()}",
        f"Total {CASES_COL}: {daily[CASES_COL].sum():,.0f}",
        f"Total {DEATHS_COL}: {daily[DEATHS_COL].sum():,.0f}",
        "",
        f"Synthetic NYC FIPS ({NYC_FIPS}) present in node_order: "
        f"{NYC_FIPS in nodes}",
        "",
        f"Zero-day fractions ({CASES_COL}):",
        f"  median {cases_stats['zero_frac'].median():.2f}  "
        f"mean {cases_stats['zero_frac'].mean():.2f}  "
        f"p95 {cases_stats['zero_frac'].quantile(0.95):.2f}",
        f"Zero-day fractions ({DEATHS_COL}):",
        f"  median {deaths_stats['zero_frac'].median():.2f}  "
        f"mean {deaths_stats['zero_frac'].mean():.2f}  "
        f"p95 {deaths_stats['zero_frac'].quantile(0.95):.2f}",
        "",
        f"Counties with ALL-ZERO {CASES_COL}:  {n_all_zero_cases}",
        f"Counties with ALL-ZERO {DEATHS_COL}: {n_all_zero_deaths}",
        "",
        f"Longest zero-cases streak (p99):   {p99_zero_cases:.0f} days",
        f"Longest zero-deaths streak (p99):  {p99_zero_deaths:.0f} days",
        "",
        "Top 10 longest zero-cases streaks (FIPS, days):",
        worst_cases_zeros.to_string(),
        "",
        "Top 10 longest zero-deaths streaks (FIPS, days):",
        worst_deaths_zeros.to_string(),
        "",
        f"Sample counties plotted (FIPS): {sample_ids}",
        "",
        "Connectivity vs distance cutoff (haversine, meters)",
        "-" * 60,
        format_connectivity_report(conn),
        "",
        "Isolated-node count at sub-d_min cutoffs",
        "-" * 60,
        format_isolation_report(conn["first_neighbor_dist"]),
        "",
        "Diagnostics — major issues flagged for training",
        "=" * 60,
        diagnostics,
    ]

    summary = "\n".join(lines)
    print()
    print(summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print(f"\nSummary also written to {(OUT_DIR / 'summary.txt').resolve()}")


# ── Connectivity analysis ─────────────────────────────────────────────────────


class _UnionFind:
    """Tiny union-find for Kruskal-style connectivity sweeps."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = int(self.parent[a])
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def connectivity_analysis(
    edges: pd.DataFrame | None, nodes: list[str]
) -> dict:
    """
    Sweep edges in ascending distance order and record, at every distance
    that reduces the number of connected components, the (distance,
    remaining components) pair. Also record the smallest cutoff such that
    every node has at least one neighbour (i.e., no isolated nodes).

    Returns a dict with:
      - n_nodes:           int
      - d_min_degree:      float, smallest cutoff with no isolated node
      - d_full_connected:  float, smallest cutoff that makes graph connected
      - events:            list[(distance, components_after)], sorted by
                           ascending distance, one entry per component merge
      - n_events:          int, number of merges (always n_nodes - 1)
      - first_neighbor_dist: np.ndarray of length n_nodes, distance to each
                           node's nearest neighbour (so # isolated at cutoff
                           c is just (first_neighbor_dist > c).sum()).
    """
    if edges is None or edges.empty or len(nodes) <= 1:
        return {
            "n_nodes": len(nodes),
            "d_min_degree": float("nan"),
            "d_full_connected": float("nan"),
            "events": [],
            "n_events": 0,
            "first_neighbor_dist": np.array([], dtype=np.float64),
        }

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # Keep only one direction of each undirected edge (src < dst lexically)
    # to halve the work. Source edges are symmetric haversine distances.
    und = edges[edges["src"].astype(str) < edges["dst"].astype(str)].copy()
    # Sort edges by ascending cost once (stable) → kruskal-like sweep.
    und = und.sort_values("cost", kind="mergesort").reset_index(drop=True)

    src_idx = und["src"].astype(str).map(node_to_idx).to_numpy()
    dst_idx = und["dst"].astype(str).map(node_to_idx).to_numpy()
    cost = und["cost"].to_numpy()

    uf = _UnionFind(N)
    components = N
    events: list[tuple[float, int]] = []

    # Track the distance at which each node first acquires a neighbour.
    # Since edges are sorted ascending, the first edge incident to a node
    # gives that node's nearest-neighbour distance.
    first_neighbor_dist = np.full(N, np.inf, dtype=np.float64)

    for i in range(len(und)):
        s = int(src_idx[i])
        d = int(dst_idx[i])
        c = float(cost[i])
        if first_neighbor_dist[s] == np.inf:
            first_neighbor_dist[s] = c
        if first_neighbor_dist[d] == np.inf:
            first_neighbor_dist[d] = c
        if uf.union(s, d):
            components -= 1
            events.append((c, components))
            if components == 1:
                break

    # d_min_degree: smallest cutoff so every node has >= 1 neighbour.
    # This equals the max over nodes of their nearest-neighbour distance.
    # If any node has no neighbour at all (shouldn't happen with a fully
    # connected edge list, but defensively), this is +inf.
    if np.isinf(first_neighbor_dist).any():
        d_min_degree = float("inf")
    else:
        d_min_degree = float(first_neighbor_dist.max())

    d_full_connected = events[-1][0] if events else float("nan")

    return {
        "n_nodes": N,
        "d_min_degree": d_min_degree,
        "d_full_connected": d_full_connected,
        "events": events,
        "n_events": len(events),
        "first_neighbor_dist": first_neighbor_dist,
    }


# Cutoffs (in metres) at which the isolation count is tabulated. Chosen
# as a geometric sweep across the sub-d_min_degree range to expose where
# the number of isolated nodes starts "exploding" — i.e. the elbow of the
# isolation-vs-cutoff curve. The largest value is just above the
# observed d_min_degree (~1263 km) so the sweep brackets the full range.
ISOLATION_CUTOFFS_KM = (
    25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1300,
)


def plot_isolation_curve(
    first_neighbor_dist: np.ndarray, out_path: Path
) -> None:
    """Plot # isolated nodes vs distance cutoff.

    A node is "isolated" at cutoff c iff its nearest-neighbour distance
    exceeds c. The curve is the complementary CDF of first-neighbour
    distances, shown on linear and log-log axes side by side so the
    explosion point (the elbow where isolation count grows rapidly as the
    cutoff drops) is visible on either scale.
    """
    if first_neighbor_dist.size == 0:
        return

    # Use every sorted first-neighbour distance as a curve point — at a
    # cutoff equal to the k-th smallest first-neighbour distance, exactly
    # N - k nodes are isolated (the ones with larger nearest-neighbour
    # distance). Prepend 0 so the curve starts at N (cutoff=0 → every
    # node isolated). Drop any infinite values defensively.
    finite = first_neighbor_dist[np.isfinite(first_neighbor_dist)]
    sorted_fnd = np.sort(finite)
    N = first_neighbor_dist.size
    # At cutoff = sorted_fnd[k] (inclusive), nodes with fnd > cutoff are
    # isolated; that is N - (k + 1) nodes (because sorted_fnd[k] is
    # included).
    cutoffs_m = np.concatenate([[0.0], sorted_fnd])
    isolated = np.concatenate([[N], N - np.arange(1, sorted_fnd.size + 1)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear axes
    axes[0].plot(cutoffs_m / 1000.0, isolated, color="C0", linewidth=1.2)
    axes[0].set_xlabel("Distance cutoff (km)")
    axes[0].set_ylabel("# isolated nodes")
    axes[0].set_title("Isolated nodes vs cutoff (linear)")
    axes[0].grid(alpha=0.3)

    # Log-log axes — compress the "almost everyone isolated" range so the
    # elbow is easy to see.
    axes[1].plot(cutoffs_m / 1000.0, isolated, color="C0", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Distance cutoff (km, log)")
    axes[1].set_ylabel("# isolated nodes (log)")
    axes[1].set_title("Isolated nodes vs cutoff (log-log)")
    axes[1].grid(alpha=0.3, which="both")

    # Mark the discrete tabulated cutoffs on both axes.
    for km in ISOLATION_CUTOFFS_KM:
        n_iso = int((first_neighbor_dist > km * 1000.0).sum())
        for ax in axes:
            ax.axvline(km, color="C7", alpha=0.25, linewidth=0.6)
        axes[0].annotate(
            f"{km}km\n{n_iso}",
            xy=(km, n_iso),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=7,
            color="C3",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_isolation_report(first_neighbor_dist: np.ndarray) -> str:
    """Table of # isolated nodes at a geometric sweep of cutoffs,
    plus an 'explosion' hint computed from the steepest log-log drop."""
    if first_neighbor_dist.size == 0:
        return "  (no edges — isolation sweep skipped)"

    rows = [
        "  cutoff (km) |  # isolated  |  % isolated  |  Δ since prev",
        "  " + "-" * 56,
    ]
    N = first_neighbor_dist.size
    prev = None
    for km in ISOLATION_CUTOFFS_KM:
        n_iso = int((first_neighbor_dist > km * 1000.0).sum())
        pct = 100.0 * n_iso / N
        delta = "" if prev is None else f"  {prev - n_iso:+d}"
        rows.append(
            f"  {km:>10d} |  {n_iso:>10d}  |  {pct:>9.2f}%  |{delta}"
        )
        prev = n_iso

    # Heuristic elbow: the cutoff where the biggest *relative* jump in
    # isolation happens as the cutoff decreases. Using finite differences
    # on the tabulated grid so the signal comes from the requested
    # cutoffs rather than the dense per-node curve.
    iso_by_cutoff = [
        int((first_neighbor_dist > km * 1000.0).sum())
        for km in ISOLATION_CUTOFFS_KM
    ]
    # Δlog(iso) between adjacent grid points, computed as we walk the
    # cutoff downward (so positive Δlog = isolation count grew when
    # cutoff shrank, i.e. the "explosion" direction).
    best_ratio = 0.0
    best_pair = None
    for (km_hi, iso_hi), (km_lo, iso_lo) in zip(
        list(zip(ISOLATION_CUTOFFS_KM, iso_by_cutoff))[1:],
        list(zip(ISOLATION_CUTOFFS_KM, iso_by_cutoff))[:-1],
    ):
        if iso_hi <= 0:
            continue
        ratio = iso_lo / max(iso_hi, 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pair = (km_lo, iso_lo, km_hi, iso_hi)

    hint = ""
    if best_pair is not None:
        km_lo, iso_lo, km_hi, iso_hi = best_pair
        hint = (
            f"\n  Steepest relative growth between tabulated cutoffs: "
            f"{km_hi}→{km_lo} km, isolated {iso_hi}→{iso_lo} "
            f"(x{best_ratio:.2f}). Below this the isolation count starts "
            "dominating; treat this as the first hint of where to NOT "
            "set the cutoff."
        )

    return "\n".join(rows) + hint


def plot_connectivity_curve(conn: dict, out_path: Path) -> None:
    """Plot number of connected components vs distance cutoff."""
    events = conn["events"]
    if not events:
        return
    # Build a staircase: before the first event components = N; after event
    # i components = events[i][1]. We plot with a step function.
    dists = np.array([e[0] for e in events])
    comps = np.array([e[1] for e in events])
    # Prepend start (0 distance, N components) so the curve begins at N.
    dists = np.concatenate([[0.0], dists])
    comps = np.concatenate([[conn["n_nodes"]], comps])

    fig, ax = plt.subplots(figsize=(10, 5))
    # Convert to km for readability
    ax.step(dists / 1000.0, comps, where="post", color="C2", linewidth=1.2)
    ax.set_xlabel("Distance cutoff (km)")
    ax.set_ylabel("# connected components")
    ax.set_yscale("log")
    ax.set_title(
        "Connected components vs distance cutoff (haversine, inclusive cutoff)"
    )
    ax.grid(alpha=0.3, which="both")

    d1_km = conn["d_min_degree"] / 1000.0
    d_full_km = conn["d_full_connected"] / 1000.0
    ax.axvline(
        d1_km, color="C1", linestyle="--", linewidth=1.0,
        label=f"no isolated nodes (d={d1_km:.1f} km)",
    )
    ax.axvline(
        d_full_km, color="C3", linestyle="--", linewidth=1.0,
        label=f"fully connected (d={d_full_km:.1f} km)",
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_connectivity_report(conn: dict) -> str:
    """Short textual summary of the connectivity sweep, with every merge
    event between d_min_degree and d_full_connected listed inline."""
    if not conn["events"]:
        return "  (no edges — connectivity analysis skipped)"

    d1 = conn["d_min_degree"]
    d_full = conn["d_full_connected"]
    events = conn["events"]

    # Filter the in-between events: distance strictly > d1 and <= d_full.
    # (d1 itself is always a merge event since it is where the last
    # singleton stops being isolated.)
    between = [(d, c) for (d, c) in events if d1 < d <= d_full]

    header = [
        f"  nodes:                              {conn['n_nodes']}",
        f"  lowest cutoff, no isolated nodes:   {d1:,.1f} m "
        f"({d1 / 1000:,.2f} km)",
        f"  lowest cutoff, fully connected:     {d_full:,.1f} m "
        f"({d_full / 1000:,.2f} km)",
        f"  merge events total:                 {len(events)}",
        f"  merge events strictly after d_min:  {len(between)}",
        "",
        "  Every merge event (distance_m, components_after):",
    ]

    # To avoid dumping thousands of events to stdout when N is huge, sample
    # if needed; but we still want the user to be able to pick a cutoff, so
    # print all merges from d1 onwards (the user specifically asked for
    # them) and downsample the pre-d1 history.
    pre_d1 = [(d, c) for (d, c) in events if d < d1]
    # Take at most 20 evenly-spaced pre-d1 samples so the printout stays
    # readable without hiding the tail information the user actually asked
    # for.
    if len(pre_d1) > 20:
        step = max(1, len(pre_d1) // 20)
        pre_d1_shown = pre_d1[::step]
        header.append(
            f"    (pre-d_min events downsampled: {len(pre_d1)} total, "
            f"showing every {step}th)"
        )
    else:
        pre_d1_shown = pre_d1

    rows: list[str] = []
    for d, c in pre_d1_shown:
        rows.append(f"    {d:>14,.1f} m   → {c} components")
    rows.append(
        f"    {d1:>14,.1f} m   → "
        f"{next(c for (dd, c) in events if dd == d1)} components  "
        f"[no isolated nodes]"
    )
    for d, c in between:
        tag = "  [fully connected]" if d == d_full else ""
        rows.append(f"    {d:>14,.1f} m   → {c} components{tag}")

    return "\n".join(header + rows)


# ── Diagnostics ───────────────────────────────────────────────────────────────


def detect_spike_counties(panel: pd.DataFrame, mult: float) -> list[str]:
    """Return FIPS of counties whose max cases > mult * median(non-zero)."""
    offenders: list[str] = []
    for nid, sub in panel.groupby("node_id", sort=False):
        v = sub[CASES_COL].to_numpy()
        nz = v[v > 0]
        if nz.size < 10:
            continue
        med = float(np.median(nz))
        if med > 0 and float(v.max()) > mult * med:
            offenders.append(str(nid))
    return offenders


def run_diagnostics(
    *,
    T: int,
    N: int,
    n_edges: int,
    dense_edge_ratio: float,
    cases_stats: pd.DataFrame,
    deaths_stats: pd.DataFrame,
    spike_counties: list[str],
    panel: pd.DataFrame,
) -> str:
    """Flag conditions that would make training GNN models problematic.

    Each line is prefixed with '[!]' for an issue or '[ok]' for a
    sanity-check pass. Thresholds are chosen to be conservative — when in
    doubt the check fires so the user sees the issue.
    """
    findings: list[str] = []

    # 1. Graph scale — an N×N dense adjacency costs 4·N² bytes (fp32) and
    # most message-passing layers scale with |E|. With ~3k counties this
    # gives ~9M directed edges, ~36 MB adjacency, quadratic memory during
    # fitting. This is the motivation for the cutoff analysis above.
    if N > 1000:
        findings.append(
            f"[!] N={N} counties with a fully connected graph → "
            f"{n_edges:,} directed edges. Many GNN backbones will OOM or "
            "run very slowly. Apply a distance cutoff before training."
        )
    else:
        findings.append(f"[ok] N={N} counties; graph scale is manageable.")

    if dense_edge_ratio > 0.9:
        findings.append(
            f"[!] Edge density {dense_edge_ratio:.2%} — graph is ~complete; "
            "without a cutoff, neighbour-mean features are just global "
            "means and the graph carries almost no inductive bias."
        )

    # 2. NYC synthetic FIPS — worth surfacing because the '99999' code and
    # the Manhattan coordinate are both arbitrary and affect edge
    # distances involving the five boroughs collapsed into a single node.
    if NYC_FIPS in cases_stats.index:
        findings.append(
            f"[!] NYC (FIPS {NYC_FIPS}) is a synthetic node collapsing "
            "the 5 boroughs into one county, geolocated at Manhattan. "
            "Its population and distances are not borough-weighted."
        )

    # 3. All-zero counties — after smoothing, a county that never reported
    # anything has constant 0 as input AND target, which inflates metrics
    # (trivially perfect prediction) and contributes nothing during
    # training. Worth either dropping or keeping as a sanity-check row.
    n_all_zero_cases = int((cases_stats["total"] == 0).sum())
    n_all_zero_deaths = int((deaths_stats["total"] == 0).sum())
    if n_all_zero_cases > 0 or n_all_zero_deaths > 0:
        findings.append(
            f"[!] {n_all_zero_cases} counties have all-zero cases and "
            f"{n_all_zero_deaths} have all-zero deaths across the entire "
            "aligned series. They degenerate into trivially-correct "
            "predictions; consider filtering them out."
        )
    else:
        findings.append("[ok] No all-zero counties in cases or deaths.")

    # 4. Long zero streaks — even after smoothing, small rural counties
    # can have hundreds of consecutive zero days. Not wrong per se, but
    # models trained with MAE/MSE will be biased toward zero and the
    # effective training signal is concentrated in high-count counties.
    p99_zero_cases = float(cases_stats["longest_zero_run"].quantile(0.99))
    p99_zero_deaths = float(deaths_stats["longest_zero_run"].quantile(0.99))
    if p99_zero_cases > 60:
        findings.append(
            f"[!] 99th-percentile longest zero-cases run is "
            f"{p99_zero_cases:.0f} days; small counties dominate the "
            "low-signal regime and may drag aggregate loss toward 0."
        )
    if p99_zero_deaths > 180:
        findings.append(
            f"[!] 99th-percentile longest zero-deaths run is "
            f"{p99_zero_deaths:.0f} days; many counties have deaths "
            "concentrated in short bursts separated by long zero gaps."
        )

    # 5. Residual single-day spikes even after 7-day smoothing — these
    # usually indicate data dumps (multi-month backfills). They wreck
    # normalisation statistics and training stability.
    if spike_counties:
        preview = ", ".join(spike_counties[:5])
        extra = "" if len(spike_counties) <= 5 else f" (+{len(spike_counties) - 5} more)"
        findings.append(
            f"[!] {len(spike_counties)} counties still have a daily spike > "
            f"{SPIKE_MULT:g}× their non-zero median after 7-day smoothing. "
            f"Probable data-dump days. FIPS (sample): {preview}{extra}"
        )
    else:
        findings.append(
            f"[ok] No residual spikes > {SPIKE_MULT:g}× median after smoothing."
        )

    # 6. Heavy-tailed per-county totals — a few counties dominate. This
    # forces per-node normalisation (or log-transform), otherwise the
    # loss is driven by NYC / LA / Cook County while the rest of the
    # graph is essentially noise.
    totals = cases_stats["total"]
    if totals.sum() > 0:
        top10_share = totals.sort_values(ascending=False).head(10).sum() / totals.sum()
        if top10_share > 0.25:
            findings.append(
                f"[!] Top-10 counties account for {top10_share:.1%} of total "
                "cases — strong heavy-tail. Use per-node normalisation "
                "or log1p, not a global z-score."
            )

    # 7. Day-of-week residue. After a 7-day moving average, weekday means
    # should be essentially equal. If they are not, the smoothing did not
    # fully absorb reporting cadence (e.g. because of ragged county
    # start dates interacting with the window).
    dow = panel["ts"].dt.day_name()
    by_dow = panel.assign(dow=dow).groupby("dow")[CASES_COL].mean()
    if by_dow.max() > 0:
        dow_spread = (by_dow.max() - by_dow.min()) / by_dow.max()
        if dow_spread > 0.10:
            findings.append(
                f"[!] Day-of-week spread in smoothed cases is "
                f"{dow_spread:.1%} — smoothing did not fully absorb the "
                "reporting cadence."
            )
        else:
            findings.append(
                f"[ok] Day-of-week spread in smoothed cases is "
                f"{dow_spread:.1%} (smoothing absorbed the cadence)."
            )

    # 8. Coverage of aligned series. With 7-day smoothing + 14-day lag, the
    # aligned series starts 20 days after the first observation; verify.
    if T < 100:
        findings.append(
            f"[!] Aligned series has only {T} days — very short for "
            "temporal forecasting with long input/horizon windows."
        )
    else:
        findings.append(
            f"[ok] Aligned series length T={T} days is long enough for "
            "sliding-window forecasting."
        )

    return "\n".join(findings)


if __name__ == "__main__":
    main()
