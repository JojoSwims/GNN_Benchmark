"""NY COVID-19 county-level dataset loader."""

import json
import tempfile
import urllib.request
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.graph import haversine_distance

YEARS = [2020, 2021, 2022, 2023]

NYT_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-{year}.csv"
)
FIPS_MAP_RAW_URL = (
    "https://raw.githubusercontent.com/josh-byster/fips_lat_long/master/fips_map.json"
)


@dataclass
class NYCovidLoader(DatasetLoader):
    """
    Loader for NY Times COVID-19 county-level dataset (2020-2023).

    Downloads daily case and death counts for US counties, geolocated via
    FIPS coordinates.  Produces a symmetric haversine-distance edge set.

    Node order is determined at download time from the intersection of NYT
    county records and the FIPS coordinate map.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="ny_covid",
            url=NYT_RAW_URL_TEMPLATE.format(year=YEARS[0]),
            frequency="1D",
            # Node order determined from source data at download time.
            node_order=[],
            feature_columns=["cases", "deaths"],
            units={"cases": "count", "deaths": "count"},
            description="NYT COVID-19 US county data 2020-2023 (daily)",
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

            # NYC patch: assign synthetic FIPS
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

            # --- Build series DataFrame ---
            series_df = self._build_series(merged, all_fips)

            # --- Build edges DataFrame ---
            edges_df = self._compute_edges(coords, all_fips)

            return series_df, edges_df

    def _build_series(
        self, merged: pd.DataFrame, all_fips: list[int]
    ) -> pd.DataFrame:
        """Build densified series panel from merged COVID data."""
        all_dates = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")

        full_index = pd.MultiIndex.from_product(
            [all_dates, all_fips], names=["date", "fips"]
        )
        panel = merged.set_index(["date", "fips"]).reindex(full_index).reset_index()

        # Convert to standard IR format
        out = pd.DataFrame(
            {
                "ts": panel["date"],
                "node_id": panel["fips"].astype(str),
                "cases": pd.array(panel["cases"], dtype="Int64"),
                "deaths": pd.array(panel["deaths"], dtype="Int64"),
            }
        )
        out = out.sort_values(["ts", "node_id"]).reset_index(drop=True)
        return out

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
