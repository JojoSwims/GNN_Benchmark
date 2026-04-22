"""NY COVID-19 county-level dataset loader."""

import json
import tempfile
import urllib.request
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.geo import haversine_distance

YEARS = [2020, 2021, 2022, 2023]

NYT_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-{year}.csv"
)
FIPS_MAP_RAW_URL = (
    "https://raw.githubusercontent.com/josh-byster/fips_lat_long/master/fips_map.json"
)

# 7-day backward rolling window for smoothing daily noise (weekend/Monday
# reporting cadence, single-day backfills).
SMOOTH_WINDOW = 7
# 14-day lag: cases at t-14 are aligned with deaths at t (epidemiological
# lag between infection reporting and death).
CASES_LAG_DAYS = 14
# Distance cutoff for the county adjacency graph (haversine, metres).
# 150 km was chosen from the isolated-node / connected-component sweep in
# examples/ny_covid_explore.py: it keeps the fraction of isolated nodes
# under ~0.5% (17/3212 counties — mostly offshore territories like Hawaii
# and Puerto Rico) while cutting the edge count by roughly two orders of
# magnitude relative to the fully connected graph. Counties beyond this
# distance from every other county are emitted with no edges, which the
# adjacency-matrix builder already handles as a zero row/column.
EDGE_DISTANCE_CUTOFF_M = 150_000.0

# Target (deaths) is listed first so model wrappers that slice std[:D_out]
# and assume the target is the first feature column work correctly.
_FEATURES = ["new_deaths_smooth", "new_cases_smooth_lag14"]
_UNITS = {
    "new_deaths_smooth": "deaths/day (7-day mean)",
    "new_cases_smooth_lag14": "cases/day (7-day mean, 14-day lag)",
}


@dataclass
class NYCovidLoader(DatasetLoader):
    """
    Loader for NY Times COVID-19 county-level dataset (Jan 2020 – Mar 2023).

    Downloads cumulative case and death counts for US counties, geolocated via
    FIPS coordinates, and differences them into daily new cases / new deaths.
    Both series are then 7-day backward-smoothed and the cases series is
    shifted forward by 14 days so that each row pairs deaths at day t with
    cases from day t-14 (the epidemiological lag). new_deaths_smooth is the
    prediction target, so it is the leading feature column (model wrappers
    slice std[:D_out] and assume the target is first).

    Node order is determined at download time from the intersection of NYT
    county records and the FIPS coordinate map, stored as sorted FIPS strings,
    and shared between the series tensor and the graph adjacency so that both
    index into the same nodes.
    """

    _node_order: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="ny_covid",
            url=NYT_RAW_URL_TEMPLATE.format(year=YEARS[0]),
            frequency="1D",
            node_order=list(self._node_order),
            feature_columns=_FEATURES,
            units=_UNITS,
            window_config=WindowConfig(target_columns=["new_deaths_smooth"]),
            description=(
                "NYT COVID-19 US county-level daily new cases and deaths, "
                "Jan 2020 – Mar 2023 (NYT stopped county-level updates on "
                "2023-03-23). Both series are smoothed with a 7-day backward "
                "rolling mean; cases are shifted forward by 14 days so that "
                "cases_smooth_lag14[t] = smoothed_cases[t-14], mirroring the "
                "epidemiological lag between cases and deaths. The first 14 "
                "days of deaths and the last 14 days of cases are cut so "
                "every retained row has both aligned signals. Target: "
                "new_deaths_smooth. Edges: haversine distance with a "
                f"{EDGE_DISTANCE_CUTOFF_M / 1000:.0f} km cutoff (isolated "
                "counties keep zero edges)."
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download and convert NY COVID-19 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # --- Download source files ---
            print("Downloading NY COVID-19 data...")
            csv_paths: list[Path] = []
            for year in YEARS:
                path = tmpdir / f"us-counties-{year}.csv"
                urllib.request.urlretrieve(
                    NYT_RAW_URL_TEMPLATE.format(year=year), path
                )
                csv_paths.append(path)

            fips_map_path = tmpdir / "fips_map.json"
            urllib.request.urlretrieve(FIPS_MAP_RAW_URL, fips_map_path)

            # --- Load NYT county data ---
            frames = []
            for p in csv_paths:
                df = pd.read_csv(
                    p, usecols=["date", "county", "state", "fips", "cases", "deaths"]
                )
                df["fips"] = pd.to_numeric(df["fips"], errors="coerce").astype("Int64")
                frames.append(df)

            covid = pd.concat(frames, ignore_index=True)
            covid["date"] = pd.to_datetime(covid["date"], format="%Y-%m-%d")

            # NYC patch: NYT aggregates the five boroughs into a single
            # "New York City" county row with no FIPS code. We assign the
            # synthetic FIPS 99999 (outside the real 5-digit FIPS range) so
            # NYC becomes one node in the graph. The coordinate below is
            # Manhattan (40.7128, -74.0060), not a borough-weighted centroid.
            nyc_mask = (covid["county"] == "New York City") & (
                covid["state"] == "New York"
            )
            covid.loc[nyc_mask, "fips"] = 99999

            # --- Load FIPS coordinates ---
            with open(fips_map_path, "r", encoding="utf-8") as f:
                raw_fips_map = json.load(f)

            coords = pd.DataFrame(
                [
                    {"fips": int(k), "lat": float(v["lat"]), "lon": float(v["long"])}
                    for k, v in raw_fips_map.items()
                ]
            ).drop_duplicates(subset=["fips"], keep="first")

            # Add NYC synthetic coordinate
            coords = pd.concat(
                [
                    coords,
                    pd.DataFrame([{"fips": 99999, "lat": 40.7128, "lon": -74.0060}]),
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["fips"], keep="first")
            coords["fips"] = coords["fips"].astype("int64")

            # --- Merge and keep geolocated rows ---
            merged = covid.merge(coords, on="fips", how="left")
            merged = merged[merged["lat"].notna() & merged["lon"].notna()].copy()
            merged = merged[["date", "fips", "cases", "deaths"]].copy()
            merged["fips"] = merged["fips"].astype("int64")

            all_fips = sorted(merged["fips"].unique())

            # `all_fips` is derived from `merged` *after* dropping rows with
            # no coordinate, so it:
            #   - excludes every FIPS we filtered out (no coord in the map),
            #   - includes the synthetic NYC FIPS (99999) since we inserted
            #     its coordinate into `coords` above.
            # The series tensor and the adjacency both index through this
            # list, so edges built from the same `all_fips` stay aligned.
            self._node_order = [str(f) for f in all_fips]

            # --- Build series DataFrame ---
            series_df = self._build_series(merged, all_fips)

            # --- Build edges DataFrame ---
            edges_df = self._compute_edges(coords, all_fips)

            return series_df, edges_df

    def _build_series(
        self, merged: pd.DataFrame, all_fips: list[int]
    ) -> pd.DataFrame:
        """Build densified series panel of smoothed, lag-aligned cases/deaths.

        NYT reports cumulative counts per county. We reindex onto the full
        daily grid, forward-fill the cumulative within each FIPS (a missing
        report means the cumulative did not change), fill leading gaps with 0
        (pre-first-report days had no cases), then difference per FIPS to get
        daily new counts. Negative deltas from data revisions are clipped to
        0, so the raw daily series is non-negative and NaN-free.

        The daily series is noisy (weekend under-reporting, Monday backfills,
        single-day data dumps), so we apply a 7-day backward rolling mean per
        county (min_periods=7 → the first 6 days per county are dropped).

        To predict deaths from cases with a 2-week epidemiological lag, each
        output row at date t carries the smoothed cases from t-14 alongside
        the smoothed deaths at t. This drops the first 14 days (no lag source
        for cases) and effectively drops the last 14 days of cases (they
        would predict deaths beyond the dataset end). Combined with the
        6-day smoothing warm-up, the first 20 days of the series are dropped.
        """
        all_dates = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")

        full_index = pd.MultiIndex.from_product(
            [all_dates, all_fips], names=["date", "fips"]
        )
        panel = (
            merged.set_index(["date", "fips"])[["cases", "deaths"]]
            .reindex(full_index)
            .sort_index()
        )

        # Forward-fill cumulative per FIPS; pre-first-report days → 0.
        panel = panel.groupby(level="fips").ffill().fillna(0.0)

        # Difference to daily new counts per FIPS; clip revisions to 0.
        new = (
            panel.groupby(level="fips")
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
            .astype(float)
            .rename(columns={"cases": "new_cases", "deaths": "new_deaths"})
        )

        # Re-shape to (date, fips) long form and sort within each county so
        # rolling / shift operations respect time order.
        daily = new.reset_index().rename(columns={"date": "ts", "fips": "node_id"})
        daily["node_id"] = daily["node_id"].astype(str)
        daily = daily.sort_values(["node_id", "ts"]).reset_index(drop=True)

        # 7-day backward rolling mean per county. min_periods=7 leaves the
        # first 6 days per county NaN (user: "ignore this for the first 6 days").
        def _smooth(s: pd.Series) -> pd.Series:
            return s.rolling(window=SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()

        grouped = daily.groupby("node_id", sort=False)
        daily["new_deaths_smooth"] = grouped["new_deaths"].transform(_smooth)
        # Shift smoothed cases forward by 14 days so row at date t holds the
        # smoothed cases from t-14 (still grouped by county).
        daily["new_cases_smooth_lag14"] = grouped["new_cases"].transform(
            lambda s: _smooth(s).shift(CASES_LAG_DAYS)
        )

        # Drop rows where either aligned signal is still NaN. In practice
        # this removes the first 20 days per county (6-day smoothing warm-up
        # + 14-day lag). The unused tail cases (last 14 days) are simply not
        # present in the aligned panel.
        aligned = daily.dropna(
            subset=["new_deaths_smooth", "new_cases_smooth_lag14"]
        ).copy()

        out = aligned[
            ["ts", "node_id", "new_deaths_smooth", "new_cases_smooth_lag14"]
        ]
        return out.sort_values(["ts", "node_id"]).reset_index(drop=True)

    @staticmethod
    def _compute_edges(
        coords: pd.DataFrame, all_fips: list[int]
    ) -> pd.DataFrame:
        """Compute symmetric haversine distance edges, keeping only pairs
        within ``EDGE_DISTANCE_CUTOFF_M`` (inclusive).

        Counties whose nearest neighbour is beyond the cutoff (e.g., Hawaii,
        Puerto Rico) contribute no edges. The adjacency matrix builder
        treats missing edges as zero, so isolated nodes simply have a zero
        row/column.
        """
        county_coords = (
            coords[coords["fips"].isin(all_fips)]
            .copy()
            .sort_values("fips")
            .reset_index(drop=True)
        )

        rows_list: list[tuple[str, str, float]] = []
        records = county_coords.to_dict("records")
        for a, b in combinations(records, 2):
            dist_m = haversine_distance(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist_m > EDGE_DISTANCE_CUTOFF_M:
                continue
            dist = round(dist_m, 1)
            fa, fb = str(int(a["fips"])), str(int(b["fips"]))
            rows_list.append((fa, fb, dist))
            rows_list.append((fb, fa, dist))

        return pd.DataFrame(rows_list, columns=["src", "dst", "cost"])
