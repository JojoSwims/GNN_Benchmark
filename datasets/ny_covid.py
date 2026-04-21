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

_FEATURES = ["new_cases", "new_deaths"]
_UNITS = {"new_cases": "cases/day", "new_deaths": "deaths/day"}


@dataclass
class NYCovidLoader(DatasetLoader):
    """
    Loader for NY Times COVID-19 county-level dataset (Jan 2020 – Mar 2023).

    Downloads cumulative case and death counts for US counties, geolocated via
    FIPS coordinates, and differences them into daily new cases / new deaths.
    new_cases is the prediction target, so it is the leading feature column
    (model wrappers slice std[:D_out] and assume the target is first).

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
            window_config=WindowConfig(target_columns=["new_cases"]),
            description=(
                "NYT COVID-19 US county-level daily new cases and deaths, "
                "Jan 2020 – Mar 2023 (NYT stopped county-level updates on "
                "2023-03-23). Target: new_cases. Edges: fully connected, "
                "weighted by haversine distance."
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

            # Freeze node order (sorted FIPS strings). The series tensor and
            # the adjacency matrix both index through this list, so building
            # edges from the same all_fips guarantees alignment.
            self._node_order = [str(f) for f in all_fips]

            # --- Build series DataFrame ---
            series_df = self._build_series(merged, all_fips)

            # --- Build edges DataFrame ---
            edges_df = self._compute_edges(coords, all_fips)

            return series_df, edges_df

    def _build_series(
        self, merged: pd.DataFrame, all_fips: list[int]
    ) -> pd.DataFrame:
        """Build densified series panel of daily new cases / new deaths.

        NYT reports cumulative counts per county. We reindex onto the full
        daily grid, forward-fill the cumulative within each FIPS (a missing
        report means the cumulative did not change), fill leading gaps with 0
        (pre-first-report days had no cases), then difference per FIPS to get
        daily new counts. Negative deltas from data revisions are clipped to
        0, so the resulting series is non-negative and NaN-free.
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
        )

        out = new.reset_index().rename(
            columns={
                "date": "ts",
                "fips": "node_id",
                "cases": "new_cases",
                "deaths": "new_deaths",
            }
        )
        out["node_id"] = out["node_id"].astype(str)
        out = out[["ts", "node_id", "new_cases", "new_deaths"]]
        return out.sort_values(["ts", "node_id"]).reset_index(drop=True)

    @staticmethod
    def _compute_edges(
        coords: pd.DataFrame, all_fips: list[int]
    ) -> pd.DataFrame:
        """Compute symmetric haversine distance edges between all counties."""
        county_coords = (
            coords[coords["fips"].isin(all_fips)]
            .copy()
            .sort_values("fips")
            .reset_index(drop=True)
        )

        rows_list: list[tuple[str, str, float]] = []
        records = county_coords.to_dict("records")
        for a, b in combinations(records, 2):
            fa, fb = str(int(a["fips"])), str(int(b["fips"]))
            dist = round(
                haversine_distance(a["lat"], a["lon"], b["lat"], b["lon"]), 1
            )
            rows_list.append((fa, fb, dist))
            rows_list.append((fb, fa, dist))

        return pd.DataFrame(rows_list, columns=["src", "dst", "cost"])
