"""NOAA NDBC ocean buoy dataset loader.

Reads pre-fetched station metadata and hourly stdmet observations that
have been curated into two files hosted on Google Drive:

    noaa_buoy_stations.json   station_id -> [lat, lon]
    noaa_buoy_data.parquet    long-format rows (ts, station_id, WSPD, WTMP,
                              WVHT, PRES) already sentinel-cleaned.

``prepare()`` downloads both files on first use with ``gdown`` and caches
them under ``~/.cache/gnn_benchmark/noaa_buoy/``, so subsequent runs are
offline and fast.

Graph construction uses a two-pass approach:
  1. Base threshold: add edges for all pairs within ``edge_threshold_km``.
     Any node still isolated after this step is connected to its single
     nearest neighbour (adaptive fallback).
  2. Sparsification: greedily remove the longest edges while keeping each
     node's degree at most ``max_degree``.  An edge is only removed when
     it is not a bridge — i.e. both endpoints remain reachable without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.geo import haversine_distance

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

# Google Drive file IDs for the pre-fetched NOAA buoy data.
_STATIONS_GDRIVE_ID = "1brU5qePbtB1fSyAKlGK2LF4nwIqago61"
_DATA_GDRIVE_ID = "1ER228VCA8Yi92UD5gLQGKYpFlHdkYr5T"

_YEARS = (2022, 2023, 2024, 2025)

# WSPD (wind speed) is the prediction target, so it must come first: the
# model wrappers' inverse-transform slices ``std[:D_out]`` and assumes target
# columns are the leading entries of ``feature_columns``.
_FEATURES = ["WSPD", "WTMP", "WVHT", "PRES"]
_UNITS = {"WSPD": "m/s", "WTMP": "degC", "WVHT": "m", "PRES": "hPa"}

# Raw 0 in these features is a sensor artifact (vacuum pressure, perfectly
# calm open-ocean wave/wind are unrealistic). Reclassify as NaN so the
# harness scoring mask excludes them.
_ZERO_AS_MISSING = ("WSPD", "WVHT", "PRES")


def _default_cache_dir() -> Path:
    """Persistent cache shared across workspaces."""
    return Path.home() / ".cache" / "gnn_benchmark" / "noaa_buoy"


def _distance_matrix_km(
    coord_map: dict[str, tuple[float, float]],
    station_ids: list[str],
) -> "np.ndarray":
    """Return an NxN float32 array of haversine distances in km."""
    import numpy as np

    n = len(station_ids)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lat_i, lon_i = coord_map[station_ids[i]]
        for j in range(i + 1, n):
            lat_j, lon_j = coord_map[station_ids[j]]
            d = haversine_distance(lat_i, lon_i, lat_j, lon_j) / 1_000.0
            D[i, j] = d
            D[j, i] = d
    return D


def _bfs_connected(adj: list[set[int]], src: int, dst: int) -> bool:
    """Return True if src and dst are connected in the adjacency-list graph."""
    visited = {src}
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return False


# ── Loader ────────────────────────────────────────────────────────────────────


@dataclass
class NOAABuoyLoader(DatasetLoader):
    """Loader for NOAA NDBC ocean buoy stdmet data (pre-fetched).

    Features
    --------
    WSPD  wind speed                (m/s)   — prediction target
    WTMP  sea surface temperature   (degC)
    WVHT  significant wave height   (m)
    PRES  sea level pressure        (hPa)

    Args:
        max_missing: Drop a buoy if its WSPD series has more than this
            fraction of missing values on the full hourly grid.
        edge_threshold_km: Base distance cutoff for the first edge-building
            pass.  Pairs beyond this distance receive no edge unless a buoy
            would otherwise be completely isolated (adaptive fallback).
        max_degree: Maximum neighbours per node after the sparsification
            pass.  Long edges are removed greedily; edges that would
            disconnect a subgraph (bridges) are kept even if they push a
            node above this limit.
        cache_dir: Override the default cache location
            (``~/.cache/gnn_benchmark/noaa_buoy/``).

    Graph
    -----
    Two-pass sparse graph:
      Pass 1 — keep pairs within ``edge_threshold_km``; connect isolated
               nodes to their nearest neighbour.
      Pass 2 — greedily drop the longest edges until every node has at
               most ``max_degree`` neighbours, skipping bridges.
    """

    max_missing: float = 0.50
    edge_threshold_km: float = 500.0
    max_degree: int = 5
    cache_dir: Path | None = None

    _node_order: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="noaa_buoy",
            url=f"https://drive.google.com/uc?id={_DATA_GDRIVE_ID}",
            frequency="1H",
            node_order=list(self._node_order),
            feature_columns=_FEATURES,
            units=_UNITS,
            window_config=WindowConfig(target_columns=["WSPD"]),
            description=(
                f"NOAA NDBC ocean buoy network, Western Hemisphere, "
                f"{_YEARS[0]}-{_YEARS[-1]} (hourly). Features: WSPD "
                f"(target, wind speed), WTMP (sea surface temp), WVHT "
                f"(significant wave height), PRES (sea level pressure). "
                f"Quality filter: WSPD <= {self.max_missing*100:.0f}% missing. "
                f"Missing values kept as NaN. Edges: base threshold "
                f"{self.edge_threshold_km:.0f} km + adaptive fallback + "
                f"sparsification to max {self.max_degree} neighbours; "
                f"weighted by haversine distance (km)."
            ),
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cache = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        stations_path = cache / "noaa_buoy_stations.json"
        data_path = cache / "noaa_buoy_data.parquet"

        # 1. Download cached artifacts from Google Drive on first use.
        self._ensure_file(stations_path, _STATIONS_GDRIVE_ID, "station metadata")
        self._ensure_file(data_path, _DATA_GDRIVE_ID, "hourly observations")

        # 2. Load station coordinates
        stations_map: dict[str, tuple[float, float]] = {
            sid: (v[0], v[1])
            for sid, v in json.loads(stations_path.read_text()).items()
        }

        # 3. Load hourly observations
        df = pd.read_parquet(data_path)
        df["ts"] = pd.to_datetime(df["ts"])
        station_ids = sorted(df["station_id"].unique())
        print(f"[NOAA] Loaded {len(station_ids)} stations from {data_path.name}.")

        # 4. Reclassify raw-0 sensor artifacts as NaN
        for col in _ZERO_AS_MISSING:
            df[col] = df[col].where(df[col] != 0)

        # 5. WSPD-missingness quality filter on the full hourly grid
        full_index = pd.date_range(
            f"{_YEARS[0]}-01-01",
            f"{_YEARS[-1]}-12-31 23:00",
            freq="1h",
        )
        good: list[str] = []
        filtered: dict[str, pd.DataFrame] = {}
        for sid in station_ids:
            sid_df = df[df["station_id"] == sid].set_index("ts")[_FEATURES]
            sid_df = sid_df.reindex(full_index)
            if sid_df["WSPD"].isna().mean() <= self.max_missing:
                good.append(sid)
                filtered[sid] = sid_df

        good = sorted(good)
        self._node_order = good
        print(
            f"[NOAA] {len(good)} buoys passed quality filter "
            f"(WSPD <= {self.max_missing*100:.0f}% missing)."
        )
        if not good:
            raise RuntimeError(
                "No buoys passed the WSPD quality filter. "
                "Consider widening max_missing."
            )

        # 6. Build outputs
        series_df = self._build_series(filtered, good)
        coord_map = {sid: stations_map[sid] for sid in good if sid in stations_map}
        edges_df = self._build_edges(
            coord_map, good, self.edge_threshold_km, self.max_degree
        )
        return series_df, edges_df

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ensure_file(path: Path, gdrive_id: str, label: str) -> None:
        if path.exists():
            return
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required to fetch the NOAA buoy dataset. "
                "Install with: pip install gdown"
            )
        print(f"[NOAA] Downloading {label} from Google Drive -> {path}")
        tmp = path.with_suffix(path.suffix + ".part")
        gdown.download(id=gdrive_id, output=str(tmp), quiet=False)
        tmp.rename(path)

    @staticmethod
    def _build_series(
        frames: dict[str, pd.DataFrame],
        station_ids: list[str],
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for sid in station_ids:
            chunk = frames[sid][_FEATURES].copy()
            chunk.index.name = "ts"
            chunk = chunk.reset_index()
            chunk["node_id"] = sid
            parts.append(chunk[["ts", "node_id", *_FEATURES]])
        series = pd.concat(parts, ignore_index=True)
        return series.sort_values(["ts", "node_id"]).reset_index(drop=True)

    @staticmethod
    def _build_edges(
        coord_map: dict[str, tuple[float, float]],
        station_ids: list[str],
        edge_threshold_km: float,
        max_degree: int,
    ) -> pd.DataFrame:
        import numpy as np

        n = len(station_ids)

        # ── Distance matrix (km) ─────────────────────────────────────────────
        D = _distance_matrix_km(coord_map, station_ids)

        # ── Pass 1a: base threshold ───────────────────────────────────────────
        adj: list[set[int]] = [
            {j for j in range(n) if j != i and D[i, j] <= edge_threshold_km}
            for i in range(n)
        ]

        # ── Pass 1b: adaptive fallback for isolated nodes ─────────────────────
        D_masked = D.copy()
        np.fill_diagonal(D_masked, np.inf)
        for i in range(n):
            if not adj[i]:
                j = int(np.argmin(D_masked[i]))
                adj[i].add(j)
                adj[j].add(i)

        # ── Pass 2: sparsification — remove long edges, skip bridges ─────────
        all_edges = sorted(
            ((i, j) for i in range(n) for j in adj[i] if i < j),
            key=lambda e: D[e[0], e[1]],
            reverse=True,
        )
        for u, v in all_edges:
            if v not in adj[u]:
                continue
            if len(adj[u]) <= max_degree and len(adj[v]) <= max_degree:
                continue
            adj[u].discard(v)
            adj[v].discard(u)
            if not _bfs_connected(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)

        # ── Emit symmetric edge list ──────────────────────────────────────────
        rows: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in adj[i]:
                d_km = round(D[i, j], 1)
                rows.append((station_ids[i], station_ids[j], d_km))
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])
