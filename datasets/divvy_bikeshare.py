"""Divvy Chicago bikeshare dataset loader."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader

# ── Source: Divvy public S3 bucket of monthly trip CSVs ────────────────
# Each file is at https://divvy-tripdata.s3.amazonaws.com/{YYYYMM}-divvy-tripdata.zip.
# The loader downloads any missing monthly zips to a local cache, then
# combines them into a single parquet and applies cross-era station-ID
# reconciliation. Nothing is hosted by us — Divvy publishes the files.
_DIVVY_S3_BASE = "https://divvy-tripdata.s3.amazonaws.com"

# Default month range. Matches the registry entry; user can override
# via ``start_month`` / ``end_month`` kwargs on the loader.
_DEFAULT_START_MONTH = "2024-03"
_DEFAULT_END_MONTH = "2026-03"

# Cache layout under ``~/.cache/gnn_benchmark/divvy_bikeshare/``:
#   zips/{YYYYMM}-divvy-tripdata.zip  (one file per month)
#   divvy_combined_{start}_{end}.parquet  (built lazily)
_CACHE_DIR = Path.home() / ".cache" / "gnn_benchmark" / "divvy_bikeshare"
_ZIP_CACHE_DIR = _CACHE_DIR / "zips"

# Repo-local path: if a developer already built the combined parquet
# (e.g. via scripts/combine_divvy_tripdata.py), prefer that over the
# auto-download path so no re-download or re-combine is needed.
_REPO_LOCAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "divvy_tripdata_combined.parquet"
)

# Cross-era station-ID reconciliation. Divvy migrated its station-ID
# scheme around mid-2025: pre-migration trips use short numeric / TA-
# prefixed IDs; post-migration trips use ``CHI*`` IDs with no overlap.
# We match same-physical-station pairs across the two eras by median
# coordinates (mutual-nearest-neighbour, 30 m haversine cap) and rewrite
# pre-migration IDs to their post-migration counterparts. Tag each
# station as era A (first month before _ERA_BOUNDARY) or B otherwise.
_ERA_BOUNDARY = pd.Period("2025-06", freq="M")
_RECONCILE_THRESHOLD_M = 30.0

_EARTH_R_M = 6371008.8

# Canonical column schema for Divvy trip CSVs (stable since 2020-04).
_CANONICAL_COLS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]
_FLOAT_COLS_RAW = ["start_lat", "start_lng", "end_lat", "end_lng"]
_DATE_COLS_RAW = ["started_at", "ended_at"]

# Chicago / Divvy service-area bounding box (same as the combine-script
# audit). Trips with start OR end coords outside are almost always bad
# default pins (0.0, 0.0 or 1.0, 1.0) and would corrupt haversine.
_LAT_MIN, _LAT_MAX = 41.5, 42.2
_LNG_MIN, _LNG_MAX = -88.2, -87.4


# ── S3 download / combine / reconcile helpers ──────────────────────────


def _iter_yyyymm(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYYMM`` strings between ``start`` and ``end``.

    ``start`` / ``end`` accept any pandas-Period-parseable form (e.g.
    ``"2024-03"``, ``"202403"``, ``pd.Period("2024-03", freq="M")``).
    """
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    if e < s:
        raise ValueError(f"end_month {end!r} precedes start_month {start!r}")
    return [p.strftime("%Y%m") for p in pd.period_range(s, e, freq="M")]


def _download_month_zip(yyyymm: str, dest_dir: Path) -> Path:
    """Download the Divvy zip for ``YYYYMM`` if not cached. Returns path."""
    target = dest_dir / f"{yyyymm}-divvy-tripdata.zip"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{_DIVVY_S3_BASE}/{yyyymm}-divvy-tripdata.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    try:
        urllib.request.urlretrieve(url, target)
    except urllib.error.HTTPError as e:
        # Clean up any partial file before re-raising
        target.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Could not download {url} (HTTP {e.code}). "
            "Check that this month is published at "
            f"{_DIVVY_S3_BASE}/index.html."
        ) from e
    return target


def _read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Read the CSV inside a Divvy month zip; coerce to canonical schema."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
        ]
        if not csv_names:
            raise ValueError(f"no CSV inside {zip_path}")
        with zf.open(csv_names[0]) as f:
            raw = f.read()

    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    # Stable schema even if a month is missing a column (rare).
    for c in _CANONICAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[_CANONICAL_COLS].copy()
    for col in _FLOAT_COLS_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _DATE_COLS_RAW:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


def _combine_months(zip_paths: list[Path]) -> pd.DataFrame:
    """Concatenate per-month CSVs, deduping ride_ids across months.

    Divvy occasionally republishes some May trips in the June file
    (~hundreds of rows in 2024 data). Dropping cross-month duplicates
    keeps each ride counted exactly once.
    """
    seen_rides: set[str] = set()
    frames: list[pd.DataFrame] = []
    for zp in zip_paths:
        df = _read_csv_from_zip(zp)
        rid = df["ride_id"].astype(str).fillna("")
        keep = ~rid.isin(seen_rides) & rid.ne("")
        seen_rides.update(rid[keep])
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


def _reconcile_station_ids(raw: pd.DataFrame) -> pd.DataFrame:
    """Crosswalk pre-migration → post-migration station IDs by coords.

    Computes per-station median lat/lon and tags each as era A (first
    month < _ERA_BOUNDARY) or era B. Mutual-nearest-neighbour matching
    with a 30 m haversine cap; matched A-IDs are rewritten to their B
    counterparts in both ``start_station_id`` and ``end_station_id``.
    """
    a = raw[["start_station_id", "start_lat", "start_lng", "ended_at"]].rename(
        columns={"start_station_id": "node_id", "start_lat": "lat", "start_lng": "lon"}
    )
    b = raw[["end_station_id", "end_lat", "end_lng", "ended_at"]].rename(
        columns={"end_station_id": "node_id", "end_lat": "lat", "end_lng": "lon"}
    )
    pts = pd.concat([a, b], ignore_index=True).dropna(subset=["node_id"])
    pts["month"] = pts["ended_at"].dt.to_period("M")
    stations = pts.groupby("node_id", as_index=False).agg(
        lat=("lat", "median"), lon=("lon", "median"),
        first_month=("month", "min"),
    )
    stations["era"] = np.where(stations["first_month"] < _ERA_BOUNDARY, "A", "B")
    era_a = stations[stations["era"] == "A"].reset_index(drop=True)
    era_b = stations[stations["era"] == "B"].reset_index(drop=True)
    if era_a.empty or era_b.empty:
        return raw  # only one era present, nothing to reconcile

    # Pairwise haversine (vectorized).
    la1 = np.radians(era_a["lat"].to_numpy())
    lo1 = np.radians(era_a["lon"].to_numpy())
    la2 = np.radians(era_b["lat"].to_numpy())
    lo2 = np.radians(era_b["lon"].to_numpy())
    dphi = la1[:, None] - la2[None, :]
    dlam = lo1[:, None] - lo2[None, :]
    a_h = (
        np.sin(dphi / 2) ** 2
        + np.cos(la1)[:, None] * np.cos(la2)[None, :] * np.sin(dlam / 2) ** 2
    )
    a_h = np.clip(a_h, 0.0, 1.0)
    dist = _EARTH_R_M * 2 * np.arcsin(np.sqrt(a_h))

    nb_for_a = dist.argmin(axis=1)
    na_for_b = dist.argmin(axis=0)

    crosswalk: dict[str, str] = {}
    for i_a in range(len(era_a)):
        i_b = int(nb_for_a[i_a])
        if na_for_b[i_b] != i_a:
            continue  # not mutual nearest
        if dist[i_a, i_b] > _RECONCILE_THRESHOLD_M:
            continue  # too far apart
        crosswalk[era_a.iloc[i_a]["node_id"]] = era_b.iloc[i_b]["node_id"]

    if not crosswalk:
        return raw
    print(
        f"  reconciled {len(crosswalk)} cross-era station IDs "
        f"(threshold {_RECONCILE_THRESHOLD_M:.0f} m)"
    )
    raw["start_station_id"] = (
        raw["start_station_id"].map(crosswalk).fillna(raw["start_station_id"])
    )
    raw["end_station_id"] = (
        raw["end_station_id"].map(crosswalk).fillna(raw["end_station_id"])
    )
    return raw


def _ensure_combined_parquet(
    start_month: str, end_month: str, force_rebuild: bool = False
) -> Path:
    """Ensure a combined+reconciled Divvy parquet exists locally.

    Returns the path to the parquet. On first call:
      1. Downloads any missing monthly zips into ``_ZIP_CACHE_DIR``.
      2. Combines all per-month CSVs (with cross-month ride_id dedup).
      3. Reconciles cross-era station IDs.
      4. Writes the result to ``_CACHE_DIR``.

    Subsequent calls return the cached parquet unless ``force_rebuild``.
    """
    cache_path = _CACHE_DIR / f"divvy_combined_{start_month}_{end_month}.parquet"
    if cache_path.exists() and not force_rebuild:
        return cache_path

    months = _iter_yyyymm(start_month, end_month)
    print(
        f"[Divvy] preparing combined parquet for "
        f"{months[0]}..{months[-1]} ({len(months)} months)"
    )
    zip_paths = [_download_month_zip(m, _ZIP_CACHE_DIR) for m in months]

    print(f"[Divvy] combining {len(zip_paths)} months and deduping ride_ids...")
    raw = _combine_months(zip_paths)
    print(f"  {len(raw):,} rows after dedup")

    print("[Divvy] reconciling cross-era station IDs...")
    raw = _reconcile_station_ids(raw)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Divvy] writing combined parquet to {cache_path}")
    raw.to_parquet(cache_path, compression="zstd", index=False)
    return cache_path

# Trip-duration ceiling (seconds). Anything longer is typically a lost
# bike or a data-quality error, not a legitimate ride.
_MAX_DURATION_S = 24 * 3600

EDGE_MODES = ("static", "dynamic", "both")


@dataclass
class DivvyBikeshareLoader(DatasetLoader):
    """
    Loader for the Divvy Chicago bikeshare trip dataset.

    Reads a single-month CSV or a multi-month combined parquet / CSV
    (built by ``scripts/combine_divvy_tripdata.py``). Nodes are bike
    stations (identified by ``start_station_id`` / ``end_station_id``).
    Three node features are emitted per station per time window:
    ``activity`` (= ``departures + arrivals``, the prediction target —
    placed first because the model adapters' ``_FeatureScaler``
    denormalizes outputs using the leading-channel mean/std),
    ``departures`` (trips starting at the station, by ``started_at``)
    and ``arrivals`` (trips ending at the station, by ``ended_at``).

    Two edge flavours are available via ``edges_mode``:

    - ``"static"``   — fully-connected haversine distance graph (symmetric).
    - ``"dynamic"``  — hourly directed trip-count snapshots. For each
      snapshot window ``[t, t + resolution)``, ``adj[t][A, B]`` counts
      trips that ENDED in that window, originating at A and finishing at
      B. Bucketing by ``ended_at`` (not ``started_at``) avoids leakage:
      every trip included in ``adj[t]`` is fully observed by ``t+1``, so
      no in-flight destination information is retroactively folded in.
    - ``"both"`` (default) — produce both.

    The ``edges_mode`` is reflected in the dataset name (``_s`` / ``_d`` /
    ``_sd`` suffix) so the workspace cache keys the three shapes
    separately.

    Cleaning rules applied before aggregation (drop counts logged):
      1. Missing ``start_station_id`` / ``end_station_id`` — can't place
         the edge in the graph.
      2. Missing any of ``start_lat/lng`` / ``end_lat/lng``.
      3. ``ended_at <= started_at`` — corrupt / zero-duration rows.
      4. Trip duration > 24 h — likely lost bikes / data errors.
      5. Coord outside Chicago bbox (generous, covers Evanston / Oak
         Park) on either end.
      6. Same-station round-trips (self-loops).
      7. Duplicate ``ride_id`` across the source (safety net — the
         combine script already dedupes, but this guards against a raw
         CSV source).

    Args:
        data_path: Path to the source file. Accepts ``.parquet`` or
            ``.csv``. If ``None`` (default), monthly Divvy zips
            (``{YYYYMM}-divvy-tripdata.zip``) are downloaded directly
            from the public S3 bucket
            ``https://divvy-tripdata.s3.amazonaws.com``, combined into
            a single parquet, cross-era station IDs reconciled, and
            cached at ``~/.cache/gnn_benchmark/divvy_bikeshare/``.
        start_month: Inclusive first month to download, ``"YYYY-MM"`` /
            ``"YYYYMM"``. Default ``"2024-03"``. Only used when
            ``data_path`` is None.
        end_month: Inclusive last month to download, ``"YYYY-MM"`` /
            ``"YYYYMM"``. Default ``"2026-03"``. Only used when
            ``data_path`` is None.
        resolution: Pandas offset alias for the aggregation window. One
            of ``"1h"``, ``"6h"``, ``"1D"``. Applies to both node series
            and dynamic-edge snapshots.
        edges_mode: One of ``"static"``, ``"dynamic"``, ``"both"``.
        bbox: Optional ``(lat_min, lat_max, lng_min, lng_max)``. If
            given, only stations whose (median) coordinates fall inside
            are kept, and every trip with either endpoint outside the
            kept set is dropped entirely — so ``departures`` /
            ``arrivals`` for kept stations are not polluted by trips
            crossing in from dropped stations.
        top_k_stations: Optional integer. If given, further trims the
            kept station set to the top K stations by trip activity
            (sum of departures + arrivals across the full period).
            Applied after ``bbox`` when both are provided.
        static_edge_cutoff_m: Optional float, meters. If given, the
            static haversine graph keeps only pairs within this
            distance (symmetric). Dense matmul in this repo's models
            still multiplies by zeros, but the sparsified graph gives
            message-passing meaningful neighborhoods instead of
            global-mean averaging. Dataframe / CSV size shrinks
            proportionally.
        keep_months: Optional tuple of calendar months (1-12) to keep.
            Trips whose ``ended_at`` falls outside these months are
            dropped, and the densified time grid is restricted to the
            kept months. **Produces a non-contiguous time axis** at
            year boundaries (e.g. Nov 30 → Mar 1 are adjacent rows);
            downstream models / sliding-window code must be aware of
            this. ``None`` keeps every month.
        min_active_fraction: Optional float in [0, 1]. Drop stations
            whose fraction of nonzero (departures+arrivals) cells over
            the kept time grid is below this threshold. Trips with
            either endpoint in a dropped station are removed entirely
            before the final aggregation, so ``departures`` /
            ``arrivals`` for kept stations reflect only flows within
            the kept set (no inflation from dropped-station flows).
            Computed on the same time grid that the IR uses, so
            applying ``keep_months`` first changes the denominator.
    """

    data_path: Path | None = None
    resolution: str = "1h"
    edges_mode: str = "both"
    start_month: str = _DEFAULT_START_MONTH
    end_month: str = _DEFAULT_END_MONTH
    bbox: tuple[float, float, float, float] | None = None
    top_k_stations: int | None = None
    static_edge_cutoff_m: float | None = None
    keep_months: tuple[int, ...] | None = None
    min_active_fraction: float | None = None

    # Internal state populated during download_and_convert
    _node_order: list[str] = field(default_factory=list, init=False, repr=False)
    _dynamic_edges_df: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )
    _range_start: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _range_end: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.edges_mode not in EDGE_MODES:
            raise ValueError(
                f"edges_mode must be one of {list(EDGE_MODES)}, "
                f"got {self.edges_mode!r}"
            )
        if self.keep_months is not None:
            self.keep_months = tuple(sorted(set(int(m) for m in self.keep_months)))
            if not self.keep_months or any(
                m < 1 or m > 12 for m in self.keep_months
            ):
                raise ValueError(
                    f"keep_months must be a non-empty subset of 1..12, "
                    f"got {self.keep_months!r}"
                )
            if len(self.keep_months) == 12:
                # All months kept = no-op; normalize to None for cache hygiene.
                self.keep_months = None
        if self.min_active_fraction is not None:
            if not (0.0 <= self.min_active_fraction <= 1.0):
                raise ValueError(
                    f"min_active_fraction must be in [0, 1], got "
                    f"{self.min_active_fraction!r}"
                )
        if self.data_path is not None:
            self.data_path = Path(self.data_path)
        # else: resolved lazily in _resolve_data_path() — monthly zips
        # are downloaded from Divvy's public S3 bucket on first use,
        # combined + reconciled, and cached.

        # Normalize month strings (accept both "2024-03" and "202403").
        self.start_month = pd.Period(self.start_month, freq="M").strftime("%Y-%m")
        self.end_month = pd.Period(self.end_month, freq="M").strftime("%Y-%m")
        if pd.Period(self.start_month, freq="M") > pd.Period(self.end_month, freq="M"):
            raise ValueError(
                f"start_month {self.start_month!r} is after end_month "
                f"{self.end_month!r}"
            )

    # ── DatasetLoader interface ──────────────────────────────────────────

    @property
    def info(self) -> DatasetInfo:
        mode_tag = {"static": "s", "dynamic": "d", "both": "sd"}[self.edges_mode]
        mode_desc = {
            "static": "fully-connected haversine graph only.",
            "dynamic": "hourly directed trip-count graph only, bucketed by trip end-time.",
            "both": "both static haversine and hourly trip-count graphs (end-time bucketed).",
        }[self.edges_mode]
        # Include the source-file stem + any geographic/top-K
        # restrictions in the cache key so switching these rebuilds
        # instead of silently reusing a previous run's IR.
        if self.data_path is not None:
            src_tag = self.data_path.stem
        else:
            src_tag = f"S3_{self.start_month}_{self.end_month}"
        subset_tag = ""
        if self.bbox is not None:
            la0, la1, lo0, lo1 = self.bbox
            subset_tag += f"_bbox{la0:.3f}-{la1:.3f}-{lo0:.3f}-{lo1:.3f}"
        if self.top_k_stations is not None:
            subset_tag += f"_topk{self.top_k_stations}"
        if self.static_edge_cutoff_m is not None:
            subset_tag += f"_cut{int(self.static_edge_cutoff_m)}m"
        if self.keep_months is not None:
            subset_tag += "_kM" + ",".join(str(m) for m in self.keep_months)
        if self.min_active_fraction is not None:
            subset_tag += f"_act{int(self.min_active_fraction * 100)}"
        if self.data_path is not None:
            source_name = self.data_path.name
        else:
            source_name = (
                f"S3:{self.start_month}..{self.end_month} (combined+reconciled)"
            )
        return DatasetInfo(
            name=f"divvy_bikeshare_{src_tag}{subset_tag}_{self.resolution}_{mode_tag}",
            url=f"{_DIVVY_S3_BASE}/index.html",
            frequency=self.resolution,
            node_order=list(self._node_order),
            feature_columns=["activity", "departures", "arrivals"],
            units={
                "activity": "trips",
                "departures": "trips",
                "arrivals": "trips",
            },
            window_config=WindowConfig(target_columns=["activity"]),
            description=(
                f"Divvy Chicago bikeshare (source={source_name}). "
                f"Edge mode = {self.edges_mode!r}: {mode_desc}"
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        raw = self._read_and_clean()
        self._range_start, self._range_end = self._derive_range(raw)
        stations = self._build_stations(raw)
        self._node_order = sorted(stations["node_id"].tolist())

        series_df = self._aggregate_series(raw, self._node_order)

        edges_df: pd.DataFrame | None = None
        if self.edges_mode in ("static", "both"):
            edges_df = self._compute_static_edges(
                stations, cutoff_m=self.static_edge_cutoff_m
            )

        self._dynamic_edges_df = None
        if self.edges_mode in ("dynamic", "both"):
            self._dynamic_edges_df = self._aggregate_dynamic_edges(raw)

        return series_df, edges_df

    def build_dynamic_edges(self) -> pd.DataFrame | None:
        """Return the hourly snapshot edges DataFrame, or None if not built."""
        return self._dynamic_edges_df

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve_data_path(self) -> Path:
        """Return an existing path to the source file.

        Preference order when ``self.data_path`` is None:
          1. Repo-local combined parquet (output of
             ``scripts/combine_divvy_tripdata.py``, optionally with
             cross-era station-ID reconciliation already applied).
          2. Cached combined+reconciled parquet built lazily by
             ``_ensure_combined_parquet`` from monthly Divvy zips
             downloaded directly from S3.
        """
        if self.data_path is not None:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"Divvy source file not found at {self.data_path}. "
                    "Pass data_path=None to auto-build from the public S3 zips."
                )
            return self.data_path

        if _REPO_LOCAL_PATH.exists():
            return _REPO_LOCAL_PATH
        return _ensure_combined_parquet(self.start_month, self.end_month)

    def _read_source(self) -> pd.DataFrame:
        """Read the source parquet or CSV into a DataFrame with canonical dtypes."""
        path = self._resolve_data_path()
        suffix = path.suffix.lower()
        needed_cols = [
            "ride_id",
            "started_at",
            "ended_at",
            "start_station_id",
            "end_station_id",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
        ]

        if suffix == ".parquet":
            raw = pd.read_parquet(path, columns=needed_cols)
            # Parquet preserves dtypes; just ensure str for id columns
            for col in ["ride_id", "start_station_id", "end_station_id"]:
                raw[col] = raw[col].astype("string")
        elif suffix == ".csv":
            raw = pd.read_csv(
                path,
                usecols=needed_cols,
                dtype={
                    "ride_id": str,
                    "start_station_id": str,
                    "end_station_id": str,
                },
                parse_dates=["started_at", "ended_at"],
            )
        else:
            raise ValueError(
                f"Unsupported source extension {suffix!r}; use .parquet or .csv"
            )
        return raw

    def _read_and_clean(self) -> pd.DataFrame:
        """Read the source and apply every cleaning rule, logging drops."""
        raw = self._read_source()
        n0 = len(raw)

        # 1. Missing station IDs (either end)
        mask = raw["start_station_id"].notna() & raw["end_station_id"].notna()
        n_missing_ids = int((~mask).sum())
        raw = raw[mask]

        # 2. Missing coordinates (either end)
        coord_cols = ["start_lat", "start_lng", "end_lat", "end_lng"]
        mask = raw[coord_cols].notna().all(axis=1)
        n_missing_coords = int((~mask).sum())
        raw = raw[mask]

        # 3. Corrupt or zero durations (ended_at <= started_at)
        mask = raw["ended_at"] > raw["started_at"]
        n_bad_duration = int((~mask).sum())
        raw = raw[mask]

        # 4. Absurd durations (> 24h = lost bikes / data errors)
        dur_s = (raw["ended_at"] - raw["started_at"]).dt.total_seconds()
        mask = dur_s <= _MAX_DURATION_S
        n_too_long = int((~mask).sum())
        raw = raw[mask]

        # 5. Coords outside the Chicago bbox — either start or end out is a drop
        in_box_start = (
            (raw["start_lat"] >= _LAT_MIN)
            & (raw["start_lat"] <= _LAT_MAX)
            & (raw["start_lng"] >= _LNG_MIN)
            & (raw["start_lng"] <= _LNG_MAX)
        )
        in_box_end = (
            (raw["end_lat"] >= _LAT_MIN)
            & (raw["end_lat"] <= _LAT_MAX)
            & (raw["end_lng"] >= _LNG_MIN)
            & (raw["end_lng"] <= _LNG_MAX)
        )
        mask = in_box_start & in_box_end
        n_oob = int((~mask).sum())
        raw = raw[mask]

        # 6. Same-station round-trips (self loops). Done BEFORE the
        # bbox / month / activity filters so the activity check reflects
        # the trip set that actually ends up in the IR.
        mask = raw["start_station_id"] != raw["end_station_id"]
        n_self_loop = int((~mask).sum())
        raw = raw[mask]

        # 7. Duplicate ride_ids (safety net — combine script already dedupes).
        before = len(raw)
        raw = raw.drop_duplicates(subset="ride_id", keep="first")
        n_dup_ride = before - len(raw)

        # 8. Geographic / top-K station subset (optional).
        # Apply to the STATION set, then drop every trip with either
        # endpoint outside the kept set. This is critical: a trip with
        # origin=dropped and dest=kept would otherwise still contribute
        # to `arrivals` for the kept station, polluting the features.
        n_bbox = 0
        n_topk = 0
        n_crossing = 0
        if self.bbox is not None or self.top_k_stations is not None:
            stations = self._build_stations(raw)
            n_pre = len(stations)

            if self.bbox is not None:
                la0, la1, lo0, lo1 = self.bbox
                mask_s = (
                    (stations["lat"] >= la0)
                    & (stations["lat"] <= la1)
                    & (stations["lon"] >= lo0)
                    & (stations["lon"] <= lo1)
                )
                n_bbox = int((~mask_s).sum())
                stations = stations[mask_s].reset_index(drop=True)

            if self.top_k_stations is not None and len(stations) > self.top_k_stations:
                # Activity = trips touching the station as origin OR destination.
                activity = pd.concat(
                    [raw["start_station_id"], raw["end_station_id"]],
                    ignore_index=True,
                ).value_counts()
                stations = stations.assign(
                    _n=stations["node_id"].map(activity).fillna(0)
                )
                before_topk = len(stations)
                stations = (
                    stations.nlargest(self.top_k_stations, "_n")
                    .drop(columns="_n")
                    .reset_index(drop=True)
                )
                n_topk = before_topk - len(stations)

            if len(stations) == 0:
                raise ValueError(
                    f"bbox / top_k_stations filter dropped every station "
                    f"(started with {n_pre}). Relax the filter."
                )

            kept = set(stations["node_id"].tolist())
            mask = raw["start_station_id"].isin(kept) & raw["end_station_id"].isin(
                kept
            )
            n_crossing = int((~mask).sum())
            raw = raw[mask]

        # 7. Calendar-month filter (optional). Drop trips whose
        # ``started_at`` OR ``ended_at`` month is not in ``keep_months``.
        # Both endpoints must fall inside the kept grid so that the
        # trip contributes to both ``departures`` (bucket on
        # ``started_at``) and ``arrivals`` (bucket on ``ended_at``)
        # — otherwise a trip spanning the keep/drop month boundary
        # would contribute to only one side, breaking the
        # sum(departures) == sum(arrivals) invariant. The densified
        # grid built later will mirror this — yielding a NON-CONTIGUOUS
        # time axis at year boundaries (e.g. Nov 30 → Mar 1 are
        # adjacent rows). Downstream sliding-window code must be aware.
        n_month_drop = 0
        if self.keep_months is not None:
            mask = (
                raw["started_at"].dt.month.isin(self.keep_months)
                & raw["ended_at"].dt.month.isin(self.keep_months)
            )
            n_month_drop = int((~mask).sum())
            raw = raw[mask]
            if len(raw) == 0:
                raise ValueError(
                    f"keep_months={self.keep_months} dropped every trip. "
                    "Check that the kept months overlap the data range."
                )

        # 8. Activity-rate filter (optional). Drop stations whose
        # nonzero (departures+arrivals) cell rate over the kept time
        # grid is below ``min_active_fraction``.
        #
        # We iterate: each pass drops trips whose either endpoint is
        # not in the current kept set, then recomputes activity on the
        # surviving trips. Without iteration, kept stations that
        # interacted heavily with now-dropped stations would have an
        # inflated active-hour count from the FIRST pass and end up
        # below threshold in the final IR. Iterating until the kept
        # set stabilizes guarantees every kept station meets the
        # threshold on the trips that actually appear in the IR.
        n_activity_drop_stations = 0
        n_activity_drop_trips = 0
        if self.min_active_fraction is not None:
            t_min = raw["ended_at"].min()
            t_max = raw["ended_at"].max()
            range_start = pd.Timestamp(t_min.year, t_min.month, 1)
            range_end = pd.Timestamp(t_max.year, t_max.month, 1) + pd.offsets.MonthBegin(1)
            grid = pd.date_range(
                range_start, range_end, freq=self.resolution, inclusive="left"
            )
            if self.keep_months is not None:
                grid = grid[grid.month.isin(self.keep_months)]
            n_grid_cells = len(grid)
            threshold_count = self.min_active_fraction * n_grid_cells

            for _iteration in range(10):  # ≥99% of cases converge in ≤3
                stacked = pd.concat(
                    [
                        pd.DataFrame({
                            "node_id": raw["start_station_id"],
                            "ts": raw["started_at"].dt.floor(self.resolution),
                        }),
                        pd.DataFrame({
                            "node_id": raw["end_station_id"],
                            "ts": raw["ended_at"].dt.floor(self.resolution),
                        }),
                    ],
                    ignore_index=True,
                )
                active_hours = stacked.groupby("node_id")["ts"].nunique()
                kept_act = set(
                    active_hours[active_hours >= threshold_count].index
                )
                # Stations that fell below threshold this iteration.
                fell_below = int((active_hours < threshold_count).sum())
                if fell_below == 0:
                    break
                n_activity_drop_stations += fell_below
                mask = (
                    raw["start_station_id"].isin(kept_act)
                    & raw["end_station_id"].isin(kept_act)
                )
                n_activity_drop_trips += int((~mask).sum())
                raw = raw[mask]
                if len(raw) == 0:
                    raise ValueError(
                        f"min_active_fraction={self.min_active_fraction} "
                        f"dropped every trip (no station meets {threshold_count:.1f} "
                        f"active hours over a grid of {n_grid_cells} cells). "
                        "Relax the threshold."
                    )

        raw = raw.reset_index(drop=True)
        n_final = len(raw)

        subset_parts: list[str] = []
        if n_bbox:
            subset_parts.append(f"{n_bbox:,} stations outside bbox")
        if n_topk:
            subset_parts.append(f"{n_topk:,} stations below top-K")
        if n_crossing:
            subset_parts.append(f"{n_crossing:,} trips crossing into dropped stations")
        if n_month_drop:
            subset_parts.append(f"{n_month_drop:,} trips outside keep_months")
        if n_activity_drop_stations:
            subset_parts.append(
                f"{n_activity_drop_stations:,} stations below min_active_fraction "
                f"(dropping {n_activity_drop_trips:,} trips)"
            )
        subset_msg = ("; subset filter dropped: " + ", ".join(subset_parts)) if subset_parts else ""

        print(
            f"[Divvy] {n0:,} raw rows → {n_final:,} kept. Dropped: "
            f"{n_missing_ids:,} missing-id, "
            f"{n_missing_coords:,} missing-coord, "
            f"{n_bad_duration:,} bad-duration, "
            f"{n_too_long:,} >24h-duration, "
            f"{n_oob:,} out-of-bbox, "
            f"{n_self_loop:,} same-station, "
            f"{n_dup_ride:,} duplicate-ride-id."
            + subset_msg
        )
        return raw

    @staticmethod
    def _derive_range(
        raw: pd.DataFrame,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Month-aligned [range_start, range_end) bounding the cleaned data.

        Uses ``ended_at`` (the bucketing axis). Floors the min to the
        start of its month and ceils the max to the start of the next
        month, so the densified grid tiles whole months.
        """
        t_min = raw["ended_at"].min()
        t_max = raw["ended_at"].max()
        range_start = pd.Timestamp(t_min.year, t_min.month, 1)
        nxt = t_max + pd.offsets.MonthBegin(1)
        range_end = pd.Timestamp(nxt.year, nxt.month, 1)
        return range_start, range_end

    @staticmethod
    def _build_stations(raw: pd.DataFrame) -> pd.DataFrame:
        """Canonical station table: median lat/lon per station_id."""
        a = raw[["start_station_id", "start_lat", "start_lng"]].rename(
            columns={
                "start_station_id": "node_id",
                "start_lat": "lat",
                "start_lng": "lon",
            }
        )
        b = raw[["end_station_id", "end_lat", "end_lng"]].rename(
            columns={
                "end_station_id": "node_id",
                "end_lat": "lat",
                "end_lng": "lon",
            }
        )
        stations = (
            pd.concat([a, b], ignore_index=True)
            .groupby("node_id", as_index=False)
            .agg(lat=("lat", "median"), lon=("lon", "median"))
        )
        return stations

    def _aggregate_series(
        self, raw: pd.DataFrame, node_order: list[str]
    ) -> pd.DataFrame:
        """Node-feature panel over a full (ts × node) grid."""
        freq = self.resolution

        dep = (
            raw.assign(ts=raw["started_at"].dt.floor(freq))
            .groupby(["ts", "start_station_id"])
            .size()
            .rename("departures")
            .reset_index()
            .rename(columns={"start_station_id": "node_id"})
        )

        arr = (
            raw.assign(ts=raw["ended_at"].dt.floor(freq))
            .groupby(["ts", "end_station_id"])
            .size()
            .rename("arrivals")
            .reset_index()
            .rename(columns={"end_station_id": "node_id"})
        )

        full_ts = pd.date_range(
            self._range_start, self._range_end, freq=freq, inclusive="left"
        )
        if self.keep_months is not None:
            full_ts = full_ts[full_ts.month.isin(self.keep_months)]
        grid = pd.MultiIndex.from_product(
            [full_ts, node_order], names=["ts", "node_id"]
        ).to_frame(index=False)

        series = (
            grid.merge(dep, on=["ts", "node_id"], how="left")
            .merge(arr, on=["ts", "node_id"], how="left")
        )
        series["departures"] = series["departures"].fillna(0).astype("int32")
        series["arrivals"] = series["arrivals"].fillna(0).astype("int32")
        # `activity` is the sum and serves as the prediction target.
        # It MUST be the first feature column so model adapters'
        # _FeatureScaler.inverse_transform (which slices std[:D_out])
        # denormalizes predictions using the right per-feature stats.
        series["activity"] = (series["departures"] + series["arrivals"]).astype("int32")
        series = series[["ts", "node_id", "activity", "departures", "arrivals"]]

        series = series.sort_values(["ts", "node_id"]).reset_index(drop=True)
        return series

    @staticmethod
    def _compute_static_edges(
        stations: pd.DataFrame, cutoff_m: float | None = None
    ) -> pd.DataFrame:
        """All-pairs symmetric haversine edges. If ``cutoff_m`` is given,
        emit only pairs within ``cutoff_m`` meters (still symmetric,
        diagonal excluded)."""
        node_ids = stations["node_id"].to_numpy()
        lat = np.radians(stations["lat"].to_numpy(dtype=np.float64))
        lon = np.radians(stations["lon"].to_numpy(dtype=np.float64))

        dphi = lat[:, None] - lat[None, :]
        dlam = lon[:, None] - lon[None, :]
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlam / 2) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        dist_m = 6371008.8 * c

        N = len(node_ids)
        src_idx, dst_idx = np.meshgrid(
            np.arange(N), np.arange(N), indexing="ij"
        )
        keep = src_idx != dst_idx
        if cutoff_m is not None:
            keep = keep & (dist_m <= cutoff_m)

        return pd.DataFrame(
            {
                "src": node_ids[src_idx[keep]].astype(str),
                "dst": node_ids[dst_idx[keep]].astype(str),
                "cost": dist_m[keep].astype(np.float32),
            }
        )

    def _aggregate_dynamic_edges(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Hourly snapshot [ts, src, dst, cost], bucketed by ended_at."""
        freq = self.resolution
        dyn = (
            raw.assign(ts=raw["ended_at"].dt.floor(freq))
            .groupby(
                ["ts", "start_station_id", "end_station_id"], as_index=False
            )
            .size()
            .rename(
                columns={
                    "start_station_id": "src",
                    "end_station_id": "dst",
                    "size": "cost",
                }
            )
        )
        dyn["src"] = dyn["src"].astype(str)
        dyn["dst"] = dyn["dst"].astype(str)
        dyn["cost"] = dyn["cost"].astype("int32")
        dyn = dyn.sort_values(["ts", "src", "dst"]).reset_index(drop=True)
        return dyn
