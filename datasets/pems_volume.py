"""PEMS Volume dataset loaders (PEMS04 and PEMS08)."""

import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

PEMS04_NODE_ORDER = list(range(307))
PEMS08_NODE_ORDER = list(range(170))

URL = "https://zenodo.org/api/records/7816008/files-archive"


@dataclass
class PEMS04Loader(DatasetLoader):
    """
    Loader for PEMS04 traffic volume dataset.

    Contains 5-minute traffic data from 307 sensors with 3 features:
    flow, occupancy, and speed.

    Data period: 2018-01-01 to 2018-02-28
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="pems04",
            url=URL,
            frequency="5min",
            node_order=[str(n) for n in PEMS04_NODE_ORDER],
            feature_columns=["flow", "occupancy", "speed"],
            units={"flow": "vehicles", "occupancy": "percent", "speed": "mph"},
            description="PEMS04 traffic data (307 sensors, 3 features)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert PEMS04 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "data.zip"
            extract_dir = tmpdir / "extracted"

            # Download
            print("Downloading PEMS04 data from Zenodo...")
            urllib.request.urlretrieve(URL, str(zip_path))

            # Extract
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Find data files
            npz_path = self._find_file(extract_dir, "PEMS04.npz")
            csv_path = self._find_file(extract_dir, "PEMS04.csv")

            # Convert to IR format
            series_df = self._convert_series(npz_path, "2018-01-01 00:00:00")
            edges_df = self._convert_edges(csv_path)

            return series_df, edges_df

    def _find_file(self, extract_dir: Path, filename: str) -> Path:
        """Find a file in extracted directory."""
        for pattern in [filename, f"*/{filename}", f"*/*/{filename}"]:
            matches = list(extract_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Could not find {filename} in extracted files")

    def _convert_series(self, npz_path: Path, start_date: str) -> pd.DataFrame:
        """Convert NPZ data to series DataFrame."""
        with np.load(npz_path) as data:
            arr = data["data"]  # Shape: (T, N, 3)

        N = arr.shape[1]
        idx = pd.date_range(start_date, periods=arr.shape[0], freq="5min")

        # Split into 3 feature DataFrames
        names = ["flow", "occupancy", "speed"]
        dfs = [pd.DataFrame(arr[:, :, i], index=idx) for i in range(3)]

        # Replace zeros with NaN
        dfs = [df.where(df != 0) for df in dfs]

        # Convert to long format and combine
        longs = []
        for i, df in enumerate(dfs):
            long = (
                df.stack()
                .rename(names[i])
                .rename_axis(["ts", "node_id"])
                .reset_index()
            )
            long = long.set_index(["ts", "node_id"])
            longs.append(long)

        # Combine all features
        df = pd.concat(longs, axis=1).reset_index()

        # Densify to full grid
        full_grid = pd.MultiIndex.from_product(
            [idx, range(N)], names=["ts", "node_id"]
        )
        df = df.set_index(["ts", "node_id"]).reindex(full_grid).reset_index()

        # Convert node_id to string
        df["node_id"] = df["node_id"].astype(str)
        df = df.sort_values(["ts", "node_id"]).reset_index(drop=True)

        return df

    def _convert_edges(self, csv_path: Path) -> pd.DataFrame:
        """Convert CSV edges file to DataFrame."""
        # Read and clean the edges file
        content = csv_path.read_text()
        lines = [ln for ln in content.splitlines() if ln.strip()]

        # Parse the CSV content
        from io import StringIO
        edges = pd.read_csv(StringIO("\n".join(lines)))

        # Rename columns to standard format
        if len(edges.columns) >= 3:
            edges.columns = ["src", "dst", "cost"]
        else:
            # If only 2 columns (src, dst), add cost of 1
            edges.columns = ["src", "dst"]
            edges["cost"] = 1.0

        edges["src"] = edges["src"].astype(str)
        edges["dst"] = edges["dst"].astype(str)

        return edges


@dataclass
class PEMS08Loader(DatasetLoader):
    """
    Loader for PEMS08 traffic volume dataset.

    Contains 5-minute traffic data from 170 sensors with 3 features:
    flow, occupancy, and speed.

    Data period: 2016-07-01 to 2016-08-31
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="pems08",
            url=URL,
            frequency="5min",
            node_order=[str(n) for n in PEMS08_NODE_ORDER],
            feature_columns=["flow", "occupancy", "speed"],
            units={"flow": "vehicles", "occupancy": "percent", "speed": "mph"},
            description="PEMS08 traffic data (170 sensors, 3 features)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert PEMS08 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "data.zip"
            extract_dir = tmpdir / "extracted"

            # Download
            print("Downloading PEMS08 data from Zenodo...")
            urllib.request.urlretrieve(URL, str(zip_path))

            # Extract
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Find data files
            npz_path = self._find_file(extract_dir, "PEMS08.npz")
            csv_path = self._find_file(extract_dir, "PEMS08.csv")

            # Convert to IR format
            series_df = self._convert_series(npz_path, "2016-07-01 00:00:00")
            edges_df = self._convert_edges(csv_path)

            return series_df, edges_df

    def _find_file(self, extract_dir: Path, filename: str) -> Path:
        """Find a file in extracted directory."""
        for pattern in [filename, f"*/{filename}", f"*/*/{filename}"]:
            matches = list(extract_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Could not find {filename} in extracted files")

    def _convert_series(self, npz_path: Path, start_date: str) -> pd.DataFrame:
        """Convert NPZ data to series DataFrame."""
        with np.load(npz_path) as data:
            arr = data["data"]  # Shape: (T, N, 3)

        N = arr.shape[1]
        idx = pd.date_range(start_date, periods=arr.shape[0], freq="5min")

        # Split into 3 feature DataFrames
        names = ["flow", "occupancy", "speed"]
        dfs = [pd.DataFrame(arr[:, :, i], index=idx) for i in range(3)]

        # Replace zeros with NaN
        dfs = [df.where(df != 0) for df in dfs]

        # Convert to long format and combine
        longs = []
        for i, df in enumerate(dfs):
            long = (
                df.stack()
                .rename(names[i])
                .rename_axis(["ts", "node_id"])
                .reset_index()
            )
            long = long.set_index(["ts", "node_id"])
            longs.append(long)

        # Combine all features
        df = pd.concat(longs, axis=1).reset_index()

        # Densify to full grid
        full_grid = pd.MultiIndex.from_product(
            [idx, range(N)], names=["ts", "node_id"]
        )
        df = df.set_index(["ts", "node_id"]).reindex(full_grid).reset_index()

        # Convert node_id to string
        df["node_id"] = df["node_id"].astype(str)
        df = df.sort_values(["ts", "node_id"]).reset_index(drop=True)

        return df

    def _convert_edges(self, csv_path: Path) -> pd.DataFrame:
        """Convert CSV edges file to DataFrame."""
        # Read and clean the edges file
        content = csv_path.read_text()
        lines = [ln for ln in content.splitlines() if ln.strip()]

        # Parse the CSV content
        from io import StringIO
        edges = pd.read_csv(StringIO("\n".join(lines)))

        # Rename columns to standard format
        if len(edges.columns) >= 3:
            edges.columns = ["src", "dst", "cost"]
        else:
            edges.columns = ["src", "dst"]
            edges["cost"] = 1.0

        edges["src"] = edges["src"].astype(str)
        edges["dst"] = edges["dst"].astype(str)

        return edges
