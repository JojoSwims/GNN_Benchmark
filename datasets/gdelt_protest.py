"""GDELT Geopolitical Diffusion dataset loader.

Produces daily per-country features from GDELT Events 2.0, with edges drawn
either from UNGA voting similarity (Erik Voeten, Harvard Dataverse) or from
geographic centroid proximity for the same node set. The two topologies share
the same signal, enabling same-signal / different-topology ablation.

GDELT 2.0 publishes a 15-minute "export" file roughly every quarter hour
starting on 2015-02-18. Each file is a TSV with 61 fixed columns and no
header. The loader fetches all 96 files per day in the requested date range,
aggregates to daily per-country records, filters to the top-K most active
countries, then densifies into the canonical (ts, node_id) grid.

The default range excludes 2020 and 2021 to avoid the covid regime shift
in both news coverage volume and protest patterns.
"""

import io
import json
import socket
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.utils.geo import haversine_distance

# Country-code lookup tables are shipped as CSV alongside this module so
# that updating a mapping is a one-line data edit, not a code change.
_CODES_DIR = Path(__file__).resolve().parent / "gdelt_codes"

FIPS_TO_ISO3: dict[str, str] = dict(
    pd.read_csv(_CODES_DIR / "fips_to_iso3.csv", dtype=str)
    .itertuples(index=False, name=None)
)
COW_TO_ISO3: dict[int, str] = {
    int(cow): iso3
    for cow, iso3 in pd.read_csv(
        _CODES_DIR / "cow_to_iso3.csv", dtype={"cow": int, "iso3": str}
    ).itertuples(index=False, name=None)
}
_centroids_df = pd.read_csv(_CODES_DIR / "iso3_centroids.csv")
ISO3_CENTROIDS: dict[str, tuple[float, float]] = {
    row.iso3: (float(row.lat), float(row.lon))
    for row in _centroids_df.itertuples(index=False)
}
del _centroids_df

# CAMEO historical overrides: a handful of legacy codes in older GDELT rows
# don't match current ISO-3 values. Small enough to keep inline.
_CAMEO_OVERRIDES: dict[str, str] = {
    "GER": "DEU",  # Germany (legacy)
    "KOS": "XKX",  # Kosovo
    "PAL": "PSE",  # Palestine
    "VAT": "VAT",  # Holy See
    "SUN": "RUS",  # Soviet Union successor
    "YUG": "SRB",  # Yugoslavia successor
}
_VALID_ISO3 = set(FIPS_TO_ISO3.values()) | set(COW_TO_ISO3.values())


def cameo_to_iso3(code: str | None) -> str | None:
    """Map a CAMEO actor country code to ISO-3, or None for non-state codes."""
    if not code or not isinstance(code, str):
        return None
    code = code.strip().upper()
    if not code:
        return None
    if code in _CAMEO_OVERRIDES:
        return _CAMEO_OVERRIDES[code]
    if code in _VALID_ISO3:
        return code
    return None

GDELT_FILE_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"

# Pre-fetched parquet cache hosted on Google Drive — produced by
# ``fetch_gdelt_data.py``. The loader downloads these files on first use
# instead of hitting GDELT directly, which would take ~12h. Per-file IDs
# are hard-coded (rather than using gdown.download_folder) because Drive's
# large-file confirmation page trips up gdown's folder iterator.
_GDELT_GDRIVE_FILES: dict[str, str] = {
    "gdelt_all.parquet": "1wNxrXPYNpj0r0kRuybxueYqskYcC2XCb",
    "unga_votes.parquet": "1IKDDDv5iUSI16xIV-g4lHLUILi5PTAtk",
    "manifest.json": "1vy7FdqbeA6DvEpJrg-Ht32bARE3OKDwF",
}


def _default_cache_dir() -> Path:
    """Persistent cache shared across workspaces."""
    return Path.home() / ".cache" / "gnn_benchmark" / "gdelt_protest"


UNGA_DOI = "doi:10.7910/DVN/LEJUQZ"
DATAVERSE_METADATA_URL = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest"
    f"?persistentId={UNGA_DOI}"
)
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{fid}"

# Zero-based column positions to keep from the 61-column GDELT export.
# Using SQLDATE (col 1, YYYYMMDD) — the event date — for daily aggregation.
_GDELT_USECOLS = [1, 7, 17, 28, 29, 30, 34, 53, 60]
# pandas.read_csv returns usecols in ascending index order, so names follow
# the same ordering.
_GDELT_COL_NAMES = [
    "day",
    "actor1_country",
    "actor2_country",
    "event_root",
    "quad_class",
    "goldstein",
    "avg_tone",
    "action_country",
    "source_url",
]

# Aggregation frequency. Daily matches the protest-event literature and
# averages out news-cycle diurnal artifacts that contaminate sub-daily bins.
_BIN_FREQ = "1D"

_RELATIONAL_ROOTS = {
    "threats": 13,
    "coercions": 17,
    "assaults": 18,
    "appeals": 4,
    "cooperation": 6,
}

_FEATURE_COLUMNS = [
    "protest_count",
    "threats_issued",
    "coercions_issued",
    "assaults_issued",
    "appeals_issued",
    "cooperation_issued",
    "avg_goldstein",
    "avg_tone",
    "total_event_count",
    "material_conflict_count",
]

_UNITS = {
    "protest_count": "count",
    "threats_issued": "count",
    "coercions_issued": "count",
    "assaults_issued": "count",
    "appeals_issued": "count",
    "cooperation_issued": "count",
    "avg_goldstein": "goldstein_scale",
    "avg_tone": "tone",
    "total_event_count": "count",
    "material_conflict_count": "count",
}

_COUNT_COLUMNS = [c for c in _FEATURE_COLUMNS if _UNITS[c] == "count"]


@dataclass
class GDELTProtestLoader(DatasetLoader):
    """
    Loader for the GDELT Geopolitical Diffusion dataset.

    Reads pre-fetched daily per-country aggregates from a parquet cache
    (downloaded from Google Drive on first use via ``gdown``) and builds
    a diplomatic-similarity or geographic graph over the top-K most
    active countries. The raw-fetch helpers (``_fetch_day``,
    ``_aggregate_bins``, ``_collapse``) remain on the class because
    ``fetch_gdelt_data.py`` reuses them when regenerating the cache.

    Attributes:
        start_date: Inclusive ISO date string (default 2015-02-18, the
            earliest GDELT 2.0 date).
        end_date: Inclusive ISO date string. Default ``"2020-01-31"`` cuts
            off before the widespread COVID-19 impact on news coverage and
            protest patterns. Pass ``None`` for two days before today.
        top_k_countries: Keep the K most-active countries after aggregation.
        diplomatic_top_k: Number of strongest neighbors per node in the
            edge set.
        unga_window_years: How many recent years of UNGA roll-call votes
            feed the similarity computation.
        use_geographic_edges: When True, build edges from centroid distance
            between countries instead of UNGA voting similarity. Same node
            set either way.
        exclude_years: Years to drop from both the download loop and the
            final grid. Defaults to ``()`` — the default date range already
            ends before COVID impact, so no exclusion is needed.
        data_dir: Optional directory of pre-fetched parquet files. When
            ``None`` (default), uses ``~/.cache/gnn_benchmark/gdelt_protest/``
            and auto-downloads the parquet folder from Google Drive on
            first use via ``gdown``. Expected layout::

                <data_dir>/gdelt_YYYY-MM.parquet
                <data_dir>/unga_votes.parquet   (optional)
        max_workers: Concurrency for the 15-minute file downloads.
    """

    start_date: str = "2015-02-18"
    end_date: str | None = "2020-01-31"
    top_k_countries: int = 100
    diplomatic_top_k: int = 10
    unga_window_years: int = 5
    use_geographic_edges: bool = False
    exclude_years: tuple[int, ...] = ()
    data_dir: str | Path | None = None
    max_workers: int = 16

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="gdelt_protest",
            url="https://www.gdeltproject.org/",
            frequency=_BIN_FREQ,
            # Node order is determined at download time from the top-K most
            # active countries.
            node_order=[],
            feature_columns=list(_FEATURE_COLUMNS),
            units=dict(_UNITS),
            description=(
                "GDELT Protest Diffusion — daily geopolitical features "
                f"per country, edges from "
                f"{'geographic adjacency' if self.use_geographic_edges else 'UNGA voting similarity'}"
                + (
                    f", excluding {sorted(self.exclude_years)}"
                    if self.exclude_years else ""
                )
                + f" (~{self.top_k_countries} nodes)"
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download GDELT and UNGA data (or read from cache), convert to IR."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        if self.end_date is None:
            end = date.today() - timedelta(days=2)
        else:
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError(f"end_date ({end}) before start_date ({start})")

        cache = Path(self.data_dir) if self.data_dir else _default_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        self._ensure_cache(cache)
        binned = self._load_cached_aggregates(cache, start, end)

        node_order = self._select_top_countries(binned)
        if len(node_order) == 0:
            raise RuntimeError("Top-K country selection produced no nodes.")

        series_df = self._build_series_panel(binned, node_order, start, end)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            if self.use_geographic_edges:
                edges_df = self._compute_geographic_edges(node_order)
            else:
                edges_df = self._compute_unga_edges(tmp_path, node_order)

        return series_df, edges_df

    # ---- Cached aggregates ----------------------------------------------

    @staticmethod
    def _ensure_cache(cache: Path) -> None:
        """Populate ``cache`` from Google Drive if ``gdelt_all.parquet`` is absent."""
        if (cache / "gdelt_all.parquet").exists():
            return
        print(f"[GDELT] Downloading cached parquet files from Google Drive -> {cache}")
        for name, fid in _GDELT_GDRIVE_FILES.items():
            out = cache / name
            if out.exists():
                continue
            print(f"  fetching {name} (id={fid})")
            # `confirm=t` bypasses Drive's virus-scan warning page that
            # otherwise wraps files >25 MB in HTML instead of the file bytes.
            url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={fid}&export=download&confirm=t"
            )
            tmp = out.with_suffix(out.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            tmp.rename(out)
        if not (cache / "gdelt_all.parquet").exists():
            raise RuntimeError(
                f"Download completed but gdelt_all.parquet not found in {cache}."
            )

    def _load_cached_aggregates(
        self, data_dir: Path, start: date, end: date
    ) -> pd.DataFrame:
        """Read the consolidated pre-fetched parquet from ``data_dir``."""
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
        combined = data_dir / "gdelt_all.parquet"
        if not combined.exists():
            raise FileNotFoundError(
                f"{combined} not found. Run fetch_gdelt_data.py or let the "
                "loader auto-download from Google Drive."
            )
        print(f"Loading cached GDELT data from {combined}")
        binned = pd.read_parquet(combined)

        binned["ts"] = pd.to_datetime(binned["ts"])
        mask = (binned["ts"] >= pd.Timestamp(start)) & (
            binned["ts"] <= pd.Timestamp(end)
        )
        if self.exclude_years:
            mask &= ~binned["ts"].dt.year.isin(self.exclude_years)
        binned = binned.loc[mask].reset_index(drop=True)

        # If the cache includes multiple months of the same range (or an
        # older partial re-write was appended) de-duplicate.
        return self._collapse(binned)

    # ---- GDELT download & parse -----------------------------------------

    def _fetch_day(self, day: date, pool: ThreadPoolExecutor) -> pd.DataFrame:
        """Fetch and concatenate all 96 GDELT 15-min files for one day."""
        stamps = [
            datetime.combine(day, datetime.min.time()) + timedelta(minutes=15 * i)
            for i in range(96)
        ]
        futures = [pool.submit(self._fetch_quarter_hour, ts) for ts in stamps]
        frames = []
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=_GDELT_COL_NAMES)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _fetch_quarter_hour(ts: datetime) -> pd.DataFrame | None:
        """Download and parse a single GDELT 15-min export file."""
        stamp = ts.strftime("%Y%m%d%H%M%S")
        url = GDELT_FILE_URL.format(stamp=stamp)
        # Retry transient network errors with exponential backoff; GDELT's
        # CDN occasionally stalls and a clean retry almost always succeeds.
        raw: bytes | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as e:
                # 404 = missing file (GDELT maintenance); don't retry.
                if e.code == 404:
                    return None
                if attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
                if attempt == 3:
                    return None
            time.sleep(2 ** attempt)
        if raw is None:
            return None

        with ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                try:
                    df = pd.read_csv(
                        fh,
                        sep="\t",
                        header=None,
                        usecols=_GDELT_USECOLS,
                        dtype=str,
                        on_bad_lines="skip",
                        engine="c",
                        low_memory=False,
                    )
                except pd.errors.EmptyDataError:
                    # GDELT occasionally publishes empty exports during
                    # ingestion outages — treat as a missing file.
                    return None
        df.columns = _GDELT_COL_NAMES
        return df

    # ---- Aggregation ----------------------------------------------------

    @staticmethod
    def _aggregate_bins(events: pd.DataFrame) -> pd.DataFrame:
        """Collapse raw GDELT events into (day, iso3) feature rows."""
        events = events.copy()
        events["ts"] = pd.to_datetime(
            events["day"], format="%Y%m%d", errors="coerce"
        )
        events["event_root"] = pd.to_numeric(events["event_root"], errors="coerce")
        events["quad_class"] = pd.to_numeric(events["quad_class"], errors="coerce")
        events["goldstein"] = pd.to_numeric(events["goldstein"], errors="coerce")
        events["avg_tone"] = pd.to_numeric(events["avg_tone"], errors="coerce")
        events = events.dropna(subset=["ts"])

        events["action_iso3"] = events["action_country"].map(FIPS_TO_ISO3)
        events["actor1_iso3"] = events["actor1_country"].map(cameo_to_iso3)
        events["actor2_iso3"] = events["actor2_country"].map(cameo_to_iso3)

        # Locational features — use ActionGeo_CountryCode (FIPS -> ISO3).
        loc = events[events["action_iso3"].notna()].copy()
        loc["_is_protest"] = (loc["event_root"] == 14).astype("int64")
        loc["_is_material"] = (loc["quad_class"] == 4).astype("int64")
        loc["_is_verbal"] = (loc["quad_class"] == 3).astype("int64")
        loc_agg = (
            loc.groupby(["ts", "action_iso3"], sort=False)
            .agg(
                total_event_count=("action_iso3", "size"),
                avg_goldstein=("goldstein", "mean"),
                avg_tone=("avg_tone", "mean"),
                distinct_sources=("source_url", "nunique"),
                protest_count=("_is_protest", "sum"),
                material_conflict_count=("_is_material", "sum"),
                verbal_conflict_count=("_is_verbal", "sum"),
            )
            .reset_index()
            .rename(columns={"action_iso3": "node_id"})
        )

        # Relational features — split by Actor1 (issued) / Actor2 (received).
        issued_cols = [f"{k}_issued" for k in _RELATIONAL_ROOTS]
        received_cols = [f"{k}_received" for k in _RELATIONAL_ROOTS]

        a1 = events[events["actor1_iso3"].notna()].copy()
        for key, root in _RELATIONAL_ROOTS.items():
            a1[f"{key}_issued"] = (a1["event_root"] == root).astype("int64")
        a1_agg = (
            a1.groupby(["ts", "actor1_iso3"], sort=False)[issued_cols]
            .sum()
            .reset_index()
            .rename(columns={"actor1_iso3": "node_id"})
        )

        a2 = events[events["actor2_iso3"].notna()].copy()
        for key, root in _RELATIONAL_ROOTS.items():
            a2[f"{key}_received"] = (a2["event_root"] == root).astype("int64")
        a2_agg = (
            a2.groupby(["ts", "actor2_iso3"], sort=False)[received_cols]
            .sum()
            .reset_index()
            .rename(columns={"actor2_iso3": "node_id"})
        )

        merged = loc_agg.merge(a1_agg, on=["ts", "node_id"], how="outer").merge(
            a2_agg, on=["ts", "node_id"], how="outer"
        )
        return merged

    @staticmethod
    def _collapse(binned: pd.DataFrame) -> pd.DataFrame:
        """Sum count columns and mean the tone/goldstein averages.

        DATEADDED-floored 6h bins can straddle fetch-day boundaries (events
        added at 00:01 UTC from a file timestamped 23:45 on the previous day
        roll into the next bin), producing duplicate (ts, node_id) rows
        across per-day aggregations. Collapse them here — counts add,
        averages use plain mean.
        """
        if binned.duplicated(subset=["ts", "node_id"]).sum() == 0:
            return binned

        grouped = binned.groupby(["ts", "node_id"], sort=False)
        sum_cols = [c for c in _COUNT_COLUMNS if c in binned.columns]
        summed = grouped[sum_cols].sum(min_count=1)

        avg_parts = {}
        for c in ("avg_goldstein", "avg_tone"):
            if c in binned.columns:
                avg_parts[c] = grouped[c].mean()
        out = summed
        for c, s in avg_parts.items():
            out[c] = s
        return out.reset_index()

    def _select_top_countries(self, binned: pd.DataFrame) -> list[str]:
        """Pick the top-K countries by cumulative total_event_count."""
        totals = (
            binned.groupby("node_id")["total_event_count"]
            .sum(min_count=1)
            .dropna()
            .sort_values(ascending=False)
        )
        return totals.head(self.top_k_countries).index.astype(str).tolist()

    def _build_series_panel(
        self,
        binned: pd.DataFrame,
        node_order: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Densify the per-(day, node) aggregations to the full IR grid."""
        full_ts = pd.date_range(start, end, freq=_BIN_FREQ)
        if self.exclude_years:
            full_ts = full_ts[~full_ts.year.isin(self.exclude_years)]
        grid = pd.MultiIndex.from_product(
            [full_ts, pd.Index(node_order, dtype=object)],
            names=["ts", "node_id"],
        ).to_frame(index=False)

        selected = binned[binned["node_id"].isin(node_order)]
        merged = grid.merge(selected, on=["ts", "node_id"], how="left")

        # Ensure every feature column exists even if no events observed.
        for c in _FEATURE_COLUMNS:
            if c not in merged.columns:
                merged[c] = np.nan

        # Count features: cast to float so missing stays NaN (IR wants float).
        for c in _COUNT_COLUMNS:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")
        for c in ("avg_goldstein", "avg_tone"):
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")

        merged = merged.sort_values(["ts", "node_id"]).reset_index(drop=True)
        return merged[["ts", "node_id", *_FEATURE_COLUMNS]]

    # ---- Edge construction ---------------------------------------------

    def _compute_geographic_edges(self, node_order: list[str]) -> pd.DataFrame:
        """Top-k nearest neighbors by centroid haversine distance (meters)."""
        rows: list[tuple[str, str, float]] = []
        coords = {
            n: ISO3_CENTROIDS[n] for n in node_order if n in ISO3_CENTROIDS
        }
        for src, (lat1, lon1) in coords.items():
            candidates: list[tuple[str, float]] = []
            for dst, (lat2, lon2) in coords.items():
                if src == dst:
                    continue
                candidates.append(
                    (dst, haversine_distance(lat1, lon1, lat2, lon2))
                )
            candidates.sort(key=lambda t: t[1])
            for dst, d in candidates[: self.diplomatic_top_k]:
                rows.append((src, dst, float(d)))
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])

    def _compute_unga_edges(
        self, tmp_path: Path, node_order: list[str]
    ) -> pd.DataFrame:
        """UNGA voting similarity: cost = 1 - mean(agree) over recent window.

        Uses the precomputed dyadic agreement scores from the Voeten
        Dataverse (columns: ccode1, ccode2, agree, year).
        """
        cached = (
            Path(self.data_dir) / "unga_votes.parquet"
            if self.data_dir is not None
            else None
        )
        if cached is not None and cached.exists():
            print(f"Loading cached UNGA agreement scores from {cached}")
            scores = pd.read_parquet(cached)
        else:
            csv_path = self._download_unga_votes(tmp_path)
            scores = self._load_unga_votes(csv_path)

        max_year = int(scores["year"].max())
        min_year = max_year - self.unga_window_years + 1
        scores = scores[scores["year"] >= min_year]

        scores["iso3_src"] = scores["ccode1"].map(COW_TO_ISO3)
        scores["iso3_dst"] = scores["ccode2"].map(COW_TO_ISO3)
        node_set = set(node_order)
        scores = scores[
            scores["iso3_src"].isin(node_set) & scores["iso3_dst"].isin(node_set)
        ]

        if scores.empty:
            return pd.DataFrame(columns=["src", "dst", "cost"])

        # Voeten's AgreementScoresAll stores both (a,b) and (b,a) rows, so
        # a plain groupby already covers both directions — don't mirror.
        dyad = (
            scores.groupby(["iso3_src", "iso3_dst"], as_index=False)["agree"]
            .mean()
        )
        dyad["cost"] = 1.0 - dyad["agree"]

        dyad = dyad.sort_values(["iso3_src", "cost"])
        top = dyad.groupby("iso3_src", as_index=False).head(self.diplomatic_top_k)
        out = top.rename(columns={"iso3_src": "src", "iso3_dst": "dst"})
        return out[["src", "dst", "cost"]].reset_index(drop=True)

    @staticmethod
    def _download_unga_votes(tmp_path: Path) -> Path:
        """Resolve and download the UNGA agreement-scores file from Dataverse."""
        print("Fetching UNGA voting data from Harvard Dataverse...")
        # Dataverse blocks urllib's default User-Agent with 403, so send a
        # browser-ish UA. Retry transient failures with exponential backoff.
        headers = {"User-Agent": "Mozilla/5.0 (gnn-benchmark GDELT loader)"}

        def _open(url: str):
            req = urllib.request.Request(url, headers=headers)
            last_exc: Exception | None = None
            for attempt in range(4):
                try:
                    return urllib.request.urlopen(req, timeout=60)
                except (urllib.error.URLError, TimeoutError, socket.timeout,
                        ConnectionError) as e:
                    last_exc = e
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            raise RuntimeError("unreachable") from last_exc

        with _open(DATAVERSE_METADATA_URL) as resp:
            metadata = json.load(resp)

        # Dataverse returns a nested payload; the `files` list describes each
        # file in the latest version of the dataset.
        files = metadata["data"]["files"]

        # Prefer the dated AgreementScoresAll snapshot (e.g.
        # "AgreementScoresAll_Jun2024.csv"); fall back to the generic
        # "AgreementScores.csv" if the dated one is absent.
        def _score(f: dict) -> tuple[int, str]:
            label = (f.get("label") or "").lower()
            if not label.endswith(".csv"):
                return (-1, label)
            if "agreementscoresall" in label:
                return (2, label)
            if "agreementscores" in label:
                return (1, label)
            return (0, label)

        best = max(files, key=_score)
        if _score(best)[0] <= 0:
            raise RuntimeError(
                "Could not locate an AgreementScores CSV on Dataverse."
            )

        fid = best["dataFile"]["id"]
        label = best["label"]
        out = tmp_path / label
        with _open(DATAVERSE_FILE_URL.format(fid=fid)) as resp, open(out, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        return out

    @staticmethod
    def _load_unga_votes(path: Path) -> pd.DataFrame:
        """Load the precomputed UNGA dyadic agreement scores CSV.

        Returns a frame with columns [ccode1, ccode2, agree, year].
        """
        df = pd.read_csv(path, low_memory=False)

        needed = {"ccode1", "ccode2", "agree", "year"}
        missing = needed - set(df.columns)
        if missing:
            raise RuntimeError(
                f"AgreementScores file missing required columns: {sorted(missing)}"
            )

        df["ccode1"] = pd.to_numeric(df["ccode1"], errors="coerce").astype("Int64")
        df["ccode2"] = pd.to_numeric(df["ccode2"], errors="coerce").astype("Int64")
        df["agree"] = pd.to_numeric(df["agree"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["ccode1", "ccode2", "agree", "year"])
        return df[["ccode1", "ccode2", "agree", "year"]]
