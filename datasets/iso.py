"""NYISO Integrated Load dataset loader."""

import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

BASE_URL = "https://mis.nyiso.com/public/csv/palIntegrated"
START_YEAR = 2012
END_YEAR = 2017

# Keep this even if we refactor, it is currently not used but will be used
PTID_TO_NAME_DICT = {
    61757: "CAPITL",
    61754: "CENTRL",
    61760: "DUNWOD",
    61753: "GENESE",
    61758: "HUD VL",
    61762: "LONGIL",
    61756: "MHK VL",
    61759: "MILLWD",
    61761: "N.Y.C.",
    61755: "NORTH",
    61752: "WEST",
}

NYISO_NODE_ORDER = [str(ptid) for ptid in sorted(PTID_TO_NAME_DICT.keys())]


@dataclass
class NYISOLoader(DatasetLoader):
    """
    Loader for NYISO Integrated Load dataset (2012-2017).

    Downloads hourly electricity load data for 11 regions in the
    New York Independent System Operator territory.
    No graph structure is provided for this dataset.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="nyiso",
            url=BASE_URL,
            frequency="1h",
            node_order=NYISO_NODE_ORDER,
            feature_columns=["load"],
            units={"load": "MW"},
            description="NYISO Integrated Load 2012-2017 (11 regions, hourly)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert NYISO data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frames = []

            # Download all monthly ZIPs, then read all daily CSVs inside
            print("Downloading NYISO data...")
            for year in range(START_YEAR, END_YEAR + 1):
                for month in range(1, 13):
                    fname = f"{year}{month:02d}01palIntegrated_csv.zip"
                    url = f"{BASE_URL}/{fname}"
                    zpath = tmpdir / fname

                    urllib.request.urlretrieve(url, zpath)

                    with zipfile.ZipFile(zpath) as zf:
                        for member in sorted(
                            n for n in zf.namelist() if n.lower().endswith(".csv")
                        ):
                            with zf.open(member) as f:
                                frames.append(pd.read_csv(f))

            series_df = self._convert_series(frames)
            return series_df, None

    def _convert_series(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Convert downloaded CSV frames to series DataFrame."""
        df = pd.concat(frames, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]

        # Parse timestamps
        df["Time Stamp"] = pd.to_datetime(
            df["Time Stamp"], format="%m/%d/%Y %H:%M:%S"
        )
        df["Time Zone"] = df["Time Zone"].astype(str).str.strip()

        # Keep only requested local-date range
        start_local = pd.Timestamp(f"{START_YEAR}-01-01 00:00:00")
        end_local = pd.Timestamp(f"{END_YEAR}-12-31 23:00:00")
        df = df[
            (df["Time Stamp"] >= start_local) & (df["Time Stamp"] <= end_local)
        ].copy()

        # Build canonical UTC timestamp from local timestamp + explicit EST/EDT tag
        tz_to_offset = {"EST": "-0500", "EDT": "-0400"}
        ts_str = df["Time Stamp"].dt.strftime("%m/%d/%Y %H:%M:%S")
        df["ts"] = pd.to_datetime(
            ts_str + " " + df["Time Zone"].map(tz_to_offset),
            format="%m/%d/%Y %H:%M:%S %z",
            utc=True,
        )

        # Clean load values
        load = pd.to_numeric(
            df["Integrated Load"]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        df["load"] = load

        # Use PTID as string node_id
        df["node_id"] = df["PTID"].astype(str)

        # Select and sort
        out = df[["ts", "node_id", "load"]].copy()
        out = out.sort_values(["ts", "node_id"]).reset_index(drop=True)

        # Densify to full grid
        full_ts = pd.date_range(out["ts"].min(), out["ts"].max(), freq="1h")
        node_idx = pd.Index(NYISO_NODE_ORDER, dtype=str)

        grid = pd.MultiIndex.from_product(
            [full_ts, node_idx], names=["ts", "node_id"]
        ).to_frame(index=False)

        out = (
            grid.merge(out, on=["ts", "node_id"], how="left")
            .sort_values(["ts", "node_id"])
            .reset_index(drop=True)
        )

        return out
