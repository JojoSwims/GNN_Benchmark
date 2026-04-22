#!/usr/bin/env python3
"""Exploratory analysis of the NY COVID-19 county-level dataset.

This script does NOT train a model. It loads the prepared NY_Covid IR and
generates a set of figures + a printed summary aimed at the questions you
need to answer before modelling:

    - When does the dataset actually start / end per county?
    - How many counties are reporting on any given day?
    - How long are the runs of zero new_cases / zero new_deaths per
      county? (long zero stretches are the main worry.)
    - What fraction of days are zeros per county?
    - How skewed is the per-county total? (a few huge counties dominate.)
    - Is there a day-of-week reporting cadence (Monday dumps, weekend
      under-reporting)?
    - Are there single-day spikes that look like data dumps rather than
      real epidemiology?
    - A few representative county time series for a sanity check.

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

# Top-K counties (by total new_cases) plotted individually
TOP_K_SAMPLE = 3
# Bottom-K counties (by total new_cases, excluding all-zero) plotted individually
BOT_K_SAMPLE = 3
# Spike threshold: a day's value exceeding (median * MULT) of that county
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
    axes[0].plot(daily.index, daily["new_cases"], color="C0", linewidth=0.9)
    axes[0].set_ylabel("Total new cases / day")
    axes[0].set_title("Aggregate daily new cases (sum across all counties)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(daily.index, daily["new_deaths"], color="C3", linewidth=0.9)
    axes[1].set_ylabel("Total new deaths / day")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Aggregate daily new deaths (sum across all counties)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reporting_coverage(panel: pd.DataFrame, out_path: Path) -> None:
    """Number of counties with non-zero new_cases per day."""
    coverage = (
        panel.assign(reporting=(panel["new_cases"] > 0).astype(int))
        .groupby("ts")["reporting"]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(coverage.index, coverage.values, color="C2", linewidth=0.9)
    ax.set_ylabel("# counties with new_cases > 0")
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
    axes[0].set_title("Per-county fraction of days with new_cases = 0")
    axes[0].set_xlabel("Zero-day fraction")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(deaths_stats["zero_frac"], bins=40, color="C3", edgecolor="black")
    axes[1].set_title("Per-county fraction of days with new_deaths = 0")
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
    axes[0].set_title("Longest contiguous run of new_cases = 0 per county")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        deaths_stats["longest_zero_run"], bins=40, color="C3", edgecolor="black"
    )
    axes[1].set_title("Longest contiguous run of new_deaths = 0 per county")
    axes[1].set_xlabel("Days")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_total_per_county_hist(
    cases_stats: pd.DataFrame, deaths_stats: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # log1p so a zero-total county still maps to bin 0
    axes[0].hist(
        np.log10(cases_stats["total"].clip(lower=1.0)),
        bins=40,
        color="C0",
        edgecolor="black",
    )
    axes[0].set_title("Per-county total new_cases (log10)")
    axes[0].set_xlabel("log10(total cases)")
    axes[0].set_ylabel("# counties")
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        np.log10(deaths_stats["total"].clip(lower=1.0)),
        bins=40,
        color="C3",
        edgecolor="black",
    )
    axes[1].set_title("Per-county total new_deaths (log10)")
    axes[1].set_xlabel("log10(total deaths)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dow_pattern(panel: pd.DataFrame, out_path: Path) -> None:
    dow = panel["ts"].dt.day_name()
    by_dow = (
        panel.assign(dow=dow)
        .groupby("dow")[["new_cases", "new_deaths"]]
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
    axes[0].bar(by_dow.index, by_dow["new_cases"], color="C0", edgecolor="black")
    axes[0].set_title("Mean new_cases per (county, day) by weekday")
    axes[0].set_ylabel("Mean")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(by_dow.index, by_dow["new_deaths"], color="C3", edgecolor="black")
    axes[1].set_title("Mean new_deaths per (county, day) by weekday")
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
        ax.plot(sub["ts"], sub["new_cases"], color="C0", linewidth=0.7, label="cases")
        ax.plot(sub["ts"], sub["new_deaths"], color="C3", linewidth=0.7, label="deaths")
        ax.set_ylabel(f"FIPS {nid}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Sample county time series")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spike_distribution(
    panel: pd.DataFrame, out_path: Path
) -> None:
    """For each county compute max/median ratio (excl. zero medians)."""
    ratios: list[float] = []
    for _, sub in panel.groupby("node_id", sort=False):
        v = sub["new_cases"].to_numpy()
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
    ax.set_title("Per-county max(new_cases) / median(non-zero new_cases) (log10)")
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
    daily = panel.groupby("ts")[["new_cases", "new_deaths"]].sum().sort_index()

    # ── Per-county stats ──────────────────────────────────────────────────────
    print("Computing per-county statistics...")
    cases_stats = per_county_stats(panel, "new_cases")
    deaths_stats = per_county_stats(panel, "new_deaths")

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

    # ── Text summary ──────────────────────────────────────────────────────────
    n_all_zero_cases = int((cases_stats["total"] == 0).sum())
    n_all_zero_deaths = int((deaths_stats["total"] == 0).sum())
    p99_zero_cases = float(cases_stats["longest_zero_run"].quantile(0.99))
    p99_zero_deaths = float(deaths_stats["longest_zero_run"].quantile(0.99))

    worst_cases_zeros = cases_stats["longest_zero_run"].sort_values(ascending=False).head(10)
    worst_deaths_zeros = deaths_stats["longest_zero_run"].sort_values(ascending=False).head(10)

    lines = [
        "NY_Covid exploratory summary",
        "=" * 60,
        f"Days (T):              {T}",
        f"Counties (N):          {N}",
        f"Edges:                 {n_edges}",
        f"Date range:            {panel['ts'].min().date()} → {panel['ts'].max().date()}",
        f"Total new_cases:       {int(daily['new_cases'].sum()):,}",
        f"Total new_deaths:      {int(daily['new_deaths'].sum()):,}",
        "",
        f"Synthetic NYC FIPS ({NYC_FIPS}) present in node_order: "
        f"{NYC_FIPS in nodes}",
        "",
        "Zero-day fractions (cases):",
        f"  median {cases_stats['zero_frac'].median():.2f}  "
        f"mean {cases_stats['zero_frac'].mean():.2f}  "
        f"p95 {cases_stats['zero_frac'].quantile(0.95):.2f}",
        "Zero-day fractions (deaths):",
        f"  median {deaths_stats['zero_frac'].median():.2f}  "
        f"mean {deaths_stats['zero_frac'].mean():.2f}  "
        f"p95 {deaths_stats['zero_frac'].quantile(0.95):.2f}",
        "",
        f"Counties with ALL-ZERO new_cases:  {n_all_zero_cases}",
        f"Counties with ALL-ZERO new_deaths: {n_all_zero_deaths}",
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
    ]

    summary = "\n".join(lines)
    print()
    print(summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print(f"\nSummary also written to {(OUT_DIR / 'summary.txt').resolve()}")


if __name__ == "__main__":
    main()
