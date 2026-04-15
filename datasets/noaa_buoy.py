"""NOAA NDBC ocean buoy dataset loader.

Fetches station metadata and historical ``stdmet`` hourly observation files
directly from the NOAA NDBC public archive.  No preprocessing scripts are
required — calling ``prepare()`` downloads, parses, filters, and returns
the IR in one go, with all raw files cached under
``~/.cache/gnn_benchmark/noaa_buoy/`` so subsequent runs are fast.

NDBC endpoints
--------------
Station table:
    https://www.ndbc.noaa.gov/data/stations/station_table.txt
Per-station historical stdmet file (one per year):
    https://www.ndbc.noaa.gov/data/historical/stdmet/{STATION}h{YEAR}.txt.gz

Pipeline
--------
1. Download the station table, parse station IDs and lat/lon.
2. Filter to the requested longitude band and to buoy-type stations.
3. For every (station, year) pair in ``years``, try to download the
   historical stdmet gzip.  404s are expected and recorded via a sentinel
   file so they are not re-requested.
4. Parse each file, coerce NDBC missing-value sentinels (99/99.0/9999…) to
   NaN, resample to hourly, and concatenate.
5. Apply the WSPD-missingness quality filter.
6. Build the series DataFrame and a fully connected haversine graph.
"""

from __future__ import annotations

import gzip
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.geo import haversine_distance

# ── Constants ─────────────────────────────────────────────────────────────────

# Target feature comes first so the model wrappers' inverse_transform
# (which slices std[:D_out]) picks the target's mean/std.
_FEATURES = ["WSPD", "WTMP", "WVHT", "PRES"]
_UNITS = {"WSPD": "m/s", "WTMP": "degC", "WVHT": "m", "PRES": "hPa"}

# Raw 0 in these features is a sensor artifact; reclassify as NaN so the
# harness scoring mask excludes it.
_ZERO_AS_MISSING = ("WSPD", "WVHT", "PRES")

# NDBC missing-value sentinels in the raw stdmet files.
_NDBC_MISSING = {
    "WSPD": 99.0,
    "WVHT": 99.0,
    "PRES": 9999.0,
    "WTMP": 999.0,
}

_DEFAULT_YEARS = (2022, 2023, 2024, 2025)
_DEFAULT_LON_MIN = -180.0   # Western Hemisphere
_DEFAULT_LON_MAX = -50.0

_NDBC_STATION_TABLE_URL = (
    "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
)
_NDBC_STDMET_URL_TEMPLATE = (
    "https://www.ndbc.noaa.gov/data/historical/stdmet/{sid}h{year}.txt.gz"
)

# LOCATION field in station_table.txt looks like:
#     "35.740 N 75.330 W (35°44'23\" N 75°19'47\" W)"
_LOCATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([NS])\s*(\d+(?:\.\d+)?)\s*([EW])"
)


def _default_cache_dir() -> Path:
    """Persistent cache shared across workspaces."""
    return Path.home() / ".cache" / "gnn_benchmark" / "noaa_buoy"


# ── Loader ────────────────────────────────────────────────────────────────────


@dataclass
class NOAABuoyLoader(DatasetLoader):
    """Loader for NOAA NDBC ocean buoy stdmet data.

    Features
    --------
    WSPD  wind speed                (m/s)   — prediction target
    WTMP  sea surface temperature   (degC)
    WVHT  significant wave height   (m)
    PRES  sea level pressure        (hPa)

    Args:
        years: Calendar years to download (inclusive set).
        lon_min / lon_max: Longitude bounds of the buoys to consider.
        max_missing: Drop a buoy if its WSPD series has more than this
            fraction of missing values on the full hourly grid.
        max_workers: Thread pool size for parallel NDBC downloads.
        cache_dir: Override the default cache location
            (``~/.cache/gnn_benchmark/noaa_buoy/``).

    Graph
    -----
    Fully connected graph: every buoy pair gets an edge weighted by
    haversine distance in km.
    """

    years: tuple[int, ...] = _DEFAULT_YEARS
    lon_min: float = _DEFAULT_LON_MIN
    lon_max: float = _DEFAULT_LON_MAX
    max_missing: float = 0.50
    max_workers: int = 16
    cache_dir: Path | None = None

    _node_order: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="noaa_buoy",
            url=_NDBC_STATION_TABLE_URL,
            frequency="1H",
            node_order=list(self._node_order),
            feature_columns=_FEATURES,
            units=_UNITS,
            window_config=WindowConfig(target_columns=["WSPD"]),
            description=(
                f"NOAA NDBC ocean buoy network, longitude band "
                f"[{self.lon_min}, {self.lon_max}], years "
                f"{self.years[0]}-{self.years[-1]} (hourly). Features: WSPD "
                f"(target, wind speed), WTMP (sea surface temp), WVHT "
                f"(significant wave height), PRES (sea level pressure). "
                f"Quality filter: WSPD <= {self.max_missing*100:.0f}% missing. "
                f"Missing values kept as NaN. Edges: fully connected, "
                f"weighted by haversine distance (km)."
            ),
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cache = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        # 1. Station metadata → lat/lon
        stations = self._fetch_station_table(cache)
        stations = stations[
            (stations["lon"] >= self.lon_min)
            & (stations["lon"] <= self.lon_max)
        ].reset_index(drop=True)
        print(
            f"[NOAA] {len(stations)} candidate stations in lon "
            f"[{self.lon_min}, {self.lon_max}]"
        )
        if stations.empty:
            raise RuntimeError(
                "No NDBC stations matched the longitude filter; "
                "relax lon_min/lon_max."
            )

        # 2. Fetch stdmet files in parallel
        raw = self._fetch_all_stdmet(stations["station_id"].tolist(), cache)
        if not raw:
            raise RuntimeError(
                "No NDBC stdmet data could be fetched — is the NDBC archive "
                "reachable? First-run downloads require network access."
            )
        df = pd.concat(raw, ignore_index=True)

        # 3. Reclassify raw-0 sensor artifacts
        for col in _ZERO_AS_MISSING:
            df[col] = df[col].where(df[col] != 0)

        # 4. WSPD-missingness quality filter on the full hourly grid
        full_index = pd.date_range(
            f"{self.years[0]}-01-01",
            f"{self.years[-1]}-12-31 23:00",
            freq="1h",
        )
        frames: dict[str, pd.DataFrame] = {}
        good: list[str] = []
        for sid, sid_df in df.groupby("station_id"):
            s = (
                sid_df.set_index("ts")[_FEATURES]
                .sort_index()
                .reindex(full_index)
            )
            if s["WSPD"].isna().mean() <= self.max_missing:
                good.append(sid)
                frames[sid] = s
        good = sorted(good)
        self._node_order = good
        print(
            f"[NOAA] {len(good)} buoys passed quality filter "
            f"(WSPD <= {self.max_missing*100:.0f}% missing)."
        )
        if not good:
            raise RuntimeError(
                "No buoys passed the WSPD quality filter. "
                "Consider widening max_missing or years."
            )

        # 5. Build outputs
        series_df = self._build_series(frames, good)
        coord_map = {
            row["station_id"]: (row["lat"], row["lon"])
            for _, row in stations.iterrows()
            if row["station_id"] in good
        }
        edges_df = self._build_edges(coord_map, good)
        return series_df, edges_df

    # ── Private: station table ────────────────────────────────────────────────

    def _fetch_station_table(self, cache: Path) -> pd.DataFrame:
        path = cache / "station_table.txt"
        if not path.exists():
            print(f"[NOAA] Downloading station table from {_NDBC_STATION_TABLE_URL}")
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(_NDBC_STATION_TABLE_URL, str(tmp))
            tmp.rename(path)

        rows: list[dict] = []
        with path.open("r", errors="replace") as f:
            for line in f:
                if line.startswith("#") or "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                # Schema: STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD |
                #         LOCATION | TIMEZONE | FORECAST | NOTE
                if len(parts) < 7:
                    continue
                sid = parts[0]
                ttype = parts[2].lower()
                m = _LOCATION_RE.search(parts[6])
                if not m:
                    continue
                lat = float(m.group(1)) * (1.0 if m.group(2).upper() == "N" else -1.0)
                lon = float(m.group(3)) * (1.0 if m.group(4).upper() == "E" else -1.0)
                # Keep buoy-class platforms with anemometers; NDBC "ttype"
                # values include 'buoy', 'dart', 'tao', 'oilrig', 'cman', ...
                # Buoy and drifting ('dart', 'tao') are the relevant ocean set.
                if ttype not in {"buoy", "dart", "tao"}:
                    continue
                rows.append({"station_id": sid, "lat": lat, "lon": lon})
        if not rows:
            raise RuntimeError(
                "Failed to parse any station entries from station_table.txt"
            )
        return pd.DataFrame(rows)

    # ── Private: stdmet fetch + parse ─────────────────────────────────────────

    def _fetch_all_stdmet(
        self, sids: list[str], cache: Path
    ) -> list[pd.DataFrame]:
        stdmet_dir = cache / "stdmet"
        stdmet_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(sid, year) for sid in sids for year in self.years]
        frames: list[pd.DataFrame] = []
        completed = 0

        def _work(sid: str, year: int) -> pd.DataFrame | None:
            return self._fetch_one_stdmet(sid, year, stdmet_dir)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_work, sid, year): (sid, year)
                       for sid, year in tasks}
            for fut in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    print(f"[NOAA] stdmet fetched: {completed}/{len(tasks)}")
                try:
                    df = fut.result()
                except Exception:
                    continue
                if df is not None and not df.empty:
                    frames.append(df)

        return frames

    def _fetch_one_stdmet(
        self,
        sid: str,
        year: int,
        stdmet_dir: Path,
    ) -> pd.DataFrame | None:
        local = stdmet_dir / f"{sid}h{year}.txt.gz"
        missing_marker = stdmet_dir / f"{sid}h{year}.missing"
        if missing_marker.exists():
            return None
        if not local.exists():
            url = _NDBC_STDMET_URL_TEMPLATE.format(sid=sid, year=year)
            try:
                tmp = local.with_suffix(local.suffix + ".part")
                urllib.request.urlretrieve(url, str(tmp))
                tmp.rename(local)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    missing_marker.touch()
                return None
            except (urllib.error.URLError, TimeoutError):
                return None
        return self._parse_stdmet(local, sid)

    @staticmethod
    def _parse_stdmet(path: Path, sid: str) -> pd.DataFrame | None:
        try:
            with gzip.open(path, "rt", errors="replace") as f:
                raw = f.read()
        except Exception:
            return None

        lines = raw.splitlines()
        if not lines:
            return None

        # Header: first '#' line names the columns. A second '#' line gives
        # units and is discarded.
        header: list[str] | None = None
        data_lines: list[str] = []
        for ln in lines:
            if ln.startswith("#"):
                if header is None:
                    header = ln.lstrip("#").split()
                continue
            if ln.strip():
                data_lines.append(ln)
        if header is None or not data_lines:
            return None

        # Build rows; tolerate short lines.
        records = [ln.split() for ln in data_lines]
        width = len(header)
        records = [r for r in records if len(r) >= width]
        if not records:
            return None
        records = [r[:width] for r in records]
        df = pd.DataFrame(records, columns=header)

        # Datetime columns: NDBC uses "YY MM DD hh" (older, 2-digit year) or
        # "#YY MM DD hh mm" (newer). After lstrip the leading '#' from the
        # first column we always have YY as the first item; minute is
        # optional.
        try:
            yy = pd.to_numeric(df["YY"], errors="coerce")
            # Two-digit years (e.g. "99") → 1900+; 4-digit are already fine.
            yy = yy.where(yy >= 1900, yy + 1900).astype("Int64")
            ts_parts = {
                "year":  yy,
                "month": pd.to_numeric(df["MM"], errors="coerce"),
                "day":   pd.to_numeric(df["DD"], errors="coerce"),
                "hour":  pd.to_numeric(df["hh"], errors="coerce"),
            }
            if "mm" in df.columns:
                ts_parts["minute"] = pd.to_numeric(df["mm"], errors="coerce")
            ts = pd.to_datetime(pd.DataFrame(ts_parts), errors="coerce")
        except Exception:
            return None

        df["ts"] = ts.dt.floor("1h") if hasattr(ts, "dt") else ts

        # Coerce feature columns and mask NDBC sentinels.
        for col, miss in _NDBC_MISSING.items():
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce")
                df[col] = v.where(v != miss)
            else:
                df[col] = np.nan

        df = df.dropna(subset=["ts"])
        if df.empty:
            return None

        # Collapse sub-hourly observations (10-min cadence on some buoys) to
        # hourly means.
        agg = df.groupby("ts", as_index=False)[_FEATURES].mean(numeric_only=True)
        agg["station_id"] = sid
        return agg[["ts", "station_id", *_FEATURES]]

    # ── Private: IR assembly ──────────────────────────────────────────────────

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
    ) -> pd.DataFrame:
        rows: list[tuple[str, str, float]] = []
        for a, b in combinations(station_ids, 2):
            lat_a, lon_a = coord_map[a]
            lat_b, lon_b = coord_map[b]
            d_km = round(
                haversine_distance(lat_a, lon_a, lat_b, lon_b) / 1_000.0, 1
            )
            rows.append((a, b, d_km))
            rows.append((b, a, d_km))
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])
