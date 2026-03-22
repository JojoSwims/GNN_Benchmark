"""Melbourne Pedestrian Counts dataset loader."""

import pickle
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.geo import haversine_distance

URL = "https://github.com/uctb/Urban-Dataset/raw/main/Public_Datasets/Pedestrian/60_minutes/Pedestrian_Melbourne.zip"

NODE_ORDER = [
    1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 17, 19, 20, 21, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 36, 37, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51,
    53, 54, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 75,
]

MELPEDS_NODE_ORDER = [str(n) for n in NODE_ORDER]

START = "2021-01-01 00:00:00"
END = "2022-11-01 00:00:00"
FREQ = "60min"


@dataclass
class MelPedsLoader(DatasetLoader):
    """
    Loader for Melbourne Pedestrian Counts dataset (2021-2022).

    Downloads hourly pedestrian traffic data for 55 counting stations in
    Melbourne. Produces a symmetric haversine-distance edge set.

    Source: UCTB Urban Dataset
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="mel_peds",
            url=URL,
            frequency="1h",
            node_order=MELPEDS_NODE_ORDER,
            feature_columns=["count"],
            units={"count": "pedestrians"},
            description="Melbourne Pedestrian Counts 2021-2022 (55 sensors, hourly)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download and convert Melbourne pedestrian data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "data.zip"
            extract_dir = tmpdir / "extracted"

            # Download and extract
            print("Downloading Melbourne Pedestrian data...")
            urllib.request.urlretrieve(URL, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            pkl_file = next(extract_dir.rglob("*.pkl"))
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            node = data["Node"]
            X_all = node["TrafficNode"]  # [T, N_total]
            station_info = node["StationInfo"]  # N_total entries: [id, date, lat, lon, name]

            # Alignment check
            T_all, N_all = X_all.shape
            if N_all != len(station_info):
                raise ValueError(
                    f"Alignment fail: TrafficNode N={N_all} != len(StationInfo)={len(station_info)}"
                )

            # Build stations table
            stations_all = pd.DataFrame(
                station_info, columns=["node_id", "date", "lat", "lon", "name"]
            )
            stations_all["node_id"] = stations_all["node_id"].astype(int)

            # Map station_id -> column index
            id2col = {
                sid: i for i, sid in enumerate(stations_all["node_id"].tolist())
            }

            # Enforce deterministic order, keep only present nodes
            node_ids = [sid for sid in NODE_ORDER if sid in id2col]
            cols = [id2col[sid] for sid in node_ids]

            # Subset data and stations
            X = X_all[:, cols]
            stations = (
                stations_all.set_index("node_id").loc[node_ids].reset_index()
            )

            series_df = self._convert_series(X, node_ids)
            edges_df = self._compute_edges(stations)

            return series_df, edges_df

    @staticmethod
    def _convert_series(X: np.ndarray, node_ids: list[int]) -> pd.DataFrame:
        """Convert traffic array to series DataFrame."""
        ts = pd.date_range(START, END, freq=FREQ, inclusive="left")
        if len(ts) != X.shape[0]:
            ts_both = pd.date_range(START, END, freq=FREQ, inclusive="both")
            raise ValueError(
                f"Timestamp count mismatch: TrafficNode T={X.shape[0]}, "
                f"ts(left)={len(ts)}, ts(both)={len(ts_both)}"
            )

        T, N = X.shape
        str_ids = np.array([str(i) for i in node_ids], dtype=object)

        series = pd.DataFrame(
            {
                "ts": np.repeat(ts.to_numpy(), N),
                "node_id": np.tile(str_ids, T),
                "count": X.reshape(-1),
            }
        )
        series = series.sort_values(["ts", "node_id"]).reset_index(drop=True)
        return series

    @staticmethod
    def _compute_edges(stations: pd.DataFrame) -> pd.DataFrame:
        """Compute symmetric haversine distance edges between all stations."""
        rows: list[tuple[str, str, float]] = []
        for i in range(len(stations)):
            for j in range(len(stations)):
                if i == j:
                    continue
                d = haversine_distance(
                    float(stations.at[i, "lat"]),
                    float(stations.at[i, "lon"]),
                    float(stations.at[j, "lat"]),
                    float(stations.at[j, "lon"]),
                )
                rows.append(
                    (str(int(stations.at[i, "node_id"])),
                     str(int(stations.at[j, "node_id"])),
                     d)
                )
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])
