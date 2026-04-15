"""LamaH-CE dataset loader.

LArge-SaMple DAta for Hydrology and Environmental Sciences for Central Europe.
859 gauged catchments in the upper Danube basin and Austrian river network.

Reference:
    Klingler, C., Schulz, K., and Herrnegger, M. (2021).
    LamaH-CE: LArge-SaMple DAta for Hydrology and Environmental Sciences
    for Central Europe. Earth Syst. Sci. Data, 13, 4529–4565.
    https://doi.org/10.5194/essd-13-4529-2021

Dataset:
    https://doi.org/10.5281/zenodo.4525244
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

ZENODO_URL = "https://doi.org/10.5281/zenodo.4525244"

# Sentinel value used in LamaH-CE for missing data
_MISSING = -999.0

# ── Dynamic features (time-series from B_basins_intermediate_all) ────────────
# Curated set: forcings + state variables, no redundant min/max variants.
_MET_COLUMNS = [
    "prec",                    # precipitation (mm) — primary runoff driver
    "2m_temp_mean",            # air temperature (K) — snowmelt vs. rain
    "total_et",                # evapotranspiration (mm) — water leaving system
    "swe",                     # snow water equivalent (mm) — snow storage memory
    "surf_net_solar_rad_mean", # net solar radiation (J/m2) — energy driver
    "volsw_123",               # soil moisture 0–100 cm (m3/m3) — antecedent wetness
    "lai_high_veg",            # leaf area index, high vegetation (m2/m2)
    "lai_low_veg",             # leaf area index, low vegetation (m2/m2)
]

_MET_UNITS = {
    "prec": "mm",
    "2m_temp_mean": "K",
    "total_et": "mm",
    "swe": "mm",
    "surf_net_solar_rad_mean": "J/m2",
    "volsw_123": "m3/m3",
    "lai_high_veg": "m2/m2",
    "lai_low_veg": "m2/m2",
}

# ── Static features (time-invariant, from catchment attribute tables) ────────
# Numeric columns read from B_basins Catchment_attributes.csv
_STATIC_NUM_COLUMNS = [
    "elev_mean",   # mean catchment elevation (m a.s.l.)
    "slope_mean",  # mean catchment slope (deg)
    "area_calc",   # catchment area (km2)
    "forest_fra",  # forest fraction (0–1)
    "glac_fra",    # glacier fraction (0–1)
    "urban_fra",   # urban fraction (0–1)
    "soil_tawc",   # total available water capacity (mm)
    "soil_condu",  # saturated hydraulic conductivity (cm/day)
    "geol_perme",  # bedrock permeability (m2, log-scale)
]

_STATIC_NUM_UNITS = {
    "elev_mean": "m",
    "slope_mean": "deg",
    "area_calc": "km2",
    "forest_fra": "fraction",
    "glac_fra": "fraction",
    "urban_fra": "fraction",
    "soil_tawc": "mm",
    "soil_condu": "cm/day",
    "geol_perme": "m2",
}

# Gauge coordinates from D_gauges/Gauge_attributes.csv (EPSG:3035 projected)
_COORD_COLUMNS = ["lon", "lat"]
_COORD_UNITS = {"lon": "m_EPSG3035", "lat": "m_EPSG3035"}

# Categorical column one-hot encoded from B_basins Catchment_attributes.csv
_GC_DOM_CLASSES = ["mt", "pa", "pi", "sc", "sm", "ss", "su", "vb"]
_GC_DOM_OH_COLUMNS = [f"gc_dom_{c}" for c in _GC_DOM_CLASSES]
_GC_DOM_OH_UNITS = {col: "binary" for col in _GC_DOM_OH_COLUMNS}

# Combined static column list (order matters for the feature vector)
_STATIC_COLUMNS = _STATIC_NUM_COLUMNS + _COORD_COLUMNS + _GC_DOM_OH_COLUMNS
_STATIC_UNITS = {**_STATIC_NUM_UNITS, **_COORD_UNITS, **_GC_DOM_OH_UNITS}


@dataclass
class LamaHCELoader(DatasetLoader):
    """
    Loader for LamaH-CE (Central Europe large-sample hydrology dataset).

    Reads daily or hourly streamflow (discharge) time series for up to 859
    gauged catchments in Central Europe, connected by the actual river network
    topology (upstream -> downstream directed edges).

    Features per node per timestep (C = 30):
      - qobs          : observed discharge (m3/s) — the prediction target
      - 8 dynamic met : curated ERA5-Land forcings from B delineation
      - 9 numeric static : topography, land cover, soil, geology
      - 2 coordinates : lon/lat (EPSG:3035 projected)
      - 8 gc_dom OHE  : dominant geology class one-hot encoded
      Static features are repeated at every timestep.

    The dataset must be downloaded manually from Zenodo and extracted locally:
        https://doi.org/10.5281/zenodo.4525244

    Download either:
        - 1_LamaH-CE_daily_hourly.tar.gz  (~15 GB, both resolutions)
        - 2_LamaH-CE_daily.tar.gz         (~1.5 GB, daily only)

    Args:
        data_root: Path to the extracted LamaH-CE root directory (the folder
            that contains A_basins_total_upstrm/, D_gauges/, etc.).
        resolution: "daily" or "hourly".
        max_gap_fraction: Exclude gauges where the fraction of remaining gaps
            (after the dataset's own gap-filling) exceeds this threshold.
            Default 0.05 (5 %).
    """

    data_root: Path
    resolution: str = "daily"
    max_gap_fraction: float = 0.05

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self._node_order: list[str] = []

    # ── DatasetLoader interface ───────────────────────────────────────────────

    @property
    def info(self) -> DatasetInfo:
        freq = "1D" if self.resolution == "daily" else "1h"
        feature_cols = ["qobs"] + _MET_COLUMNS + _STATIC_COLUMNS
        units = {"qobs": "m3/s", **_MET_UNITS, **_STATIC_UNITS}
        return DatasetInfo(
            name=f"lamah_ce_{self.resolution}",
            url=ZENODO_URL,
            frequency=freq,
            node_order=self._node_order,
            feature_columns=feature_cols,
            units=units,
            description=(
                f"LamaH-CE Central Europe streamflow network, "
                f"{self.resolution} resolution, up to 859 gauges, "
                f"{len(feature_cols)} features "
                f"(qobs + {len(_MET_COLUMNS)} dynamic met + "
                f"{len(_STATIC_COLUMNS)} static catchment)"
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Read LamaH-CE from a local extracted directory."""
        self._check_data_root()

        # 1. Load gauge metadata & catchment attributes
        gauge_attrs = self._read_gauge_attrs()
        catchment_attrs = self._read_catchment_attrs()

        # 2. Filter by gap fraction (gaps_post is fraction of gaps remaining)
        if "gaps_post" in gauge_attrs.columns:
            gauge_attrs = gauge_attrs[
                gauge_attrs["gaps_post"] <= self.max_gap_fraction
            ].copy()

        valid_ids: list[int] = sorted(gauge_attrs["ID"].tolist())
        print(f"[LamaHCE] {len(valid_ids)} gauges after gap filter "
              f"(max_gap_fraction={self.max_gap_fraction})")

        # 3. Build per-gauge static feature lookup (ID → dict of static cols)
        static_lookup = self._build_static_lookup(
            catchment_attrs, gauge_attrs, valid_ids
        )

        # 4. Read time series for all valid gauges
        series_df = self._build_series(valid_ids, static_lookup)

        # 5. Determine final node set (gauges that have at least some data)
        present_ids = sorted(series_df["node_id"].unique(), key=int)
        self._node_order = [str(i) for i in present_ids]

        # 6. Build graph edges from river network topology
        present_id_set = {int(n) for n in present_ids}
        edges_df = self._build_edges(present_id_set)

        return series_df, edges_df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_data_root(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"data_root not found: {self.data_root}\n"
                "Download the LamaH-CE dataset from:\n"
                "  https://doi.org/10.5281/zenodo.4525244\n"
                "Extract the tarball and pass the root directory to LamaHCELoader."
            )
        gauge_dir = self.data_root / "D_gauges"
        if not gauge_dir.exists():
            raise FileNotFoundError(
                f"Expected 'D_gauges/' inside data_root={self.data_root}. "
                "Make sure you extracted the archive correctly."
            )

    def _read_gauge_attrs(self) -> pd.DataFrame:
        """Load gauge attributes table (lon, lat, gaps_post, …)."""
        path = self.data_root / "D_gauges" / "1_attributes" / "Gauge_attributes.csv"
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(int)
        return df

    def _read_catchment_attrs(self) -> pd.DataFrame:
        """Load B_basins catchment attributes (topography, soil, geology, …)."""
        path = (
            self.data_root
            / "B_basins_intermediate_all"
            / "1_attributes"
            / "Catchment_attributes.csv"
        )
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(int)
        return df

    def _build_static_lookup(
        self,
        catchment_attrs: pd.DataFrame,
        gauge_attrs: pd.DataFrame,
        valid_ids: list[int],
    ) -> dict[int, dict[str, float]]:
        """
        Build a {gauge_id: {col: value, ...}} lookup for all static features.

        Combines numeric catchment attrs from B_basins, lon/lat from D_gauges,
        and one-hot encoded gc_dom.
        """
        # Index catchment attrs by ID
        ca = catchment_attrs.set_index("ID")
        ga = gauge_attrs.set_index("ID")

        lookup: dict[int, dict[str, float]] = {}
        for gid in valid_ids:
            row: dict[str, float] = {}

            # Numeric catchment attributes from B_basins
            if gid in ca.index:
                ca_row = ca.loc[gid]
                for col in _STATIC_NUM_COLUMNS:
                    val = ca_row.get(col, np.nan)
                    row[col] = float(val) if pd.notna(val) else np.nan

                # One-hot encode gc_dom
                gc_dom = str(ca_row.get("gc_dom", "")).strip().lower()
                for cls in _GC_DOM_CLASSES:
                    row[f"gc_dom_{cls}"] = 1.0 if gc_dom == cls else 0.0
            else:
                # No B_basins entry (non-delineated gauge) — all NaN
                for col in _STATIC_NUM_COLUMNS:
                    row[col] = np.nan
                for cls in _GC_DOM_CLASSES:
                    row[f"gc_dom_{cls}"] = np.nan

            # Coordinates from D_gauges (EPSG:3035)
            if gid in ga.index:
                ga_row = ga.loc[gid]
                for col in _COORD_COLUMNS:
                    val = ga_row.get(col, np.nan)
                    row[col] = float(val) if pd.notna(val) else np.nan
            else:
                for col in _COORD_COLUMNS:
                    row[col] = np.nan

            lookup[gid] = row

        return lookup

    def _qobs_dir(self) -> Path:
        """D_gauges discharge timeseries directory."""
        return self.data_root / "D_gauges" / "2_timeseries" / self.resolution

    def _met_dir(self) -> Path:
        """B_basins_intermediate_all meteorological timeseries directory."""
        return (
            self.data_root
            / "B_basins_intermediate_all"
            / "2_timeseries"
            / self.resolution
        )

    def _parse_datetime(self, df: pd.DataFrame) -> pd.Series:
        """Build a datetime Series from YYYY, MM, DD (and optionally hh) columns."""
        if "hh" in df.columns or "HH" in df.columns:
            hh_col = "hh" if "hh" in df.columns else "HH"
            return pd.to_datetime(
                df["YYYY"].str.strip()
                + "-"
                + df["MM"].str.strip().str.zfill(2)
                + "-"
                + df["DD"].str.strip().str.zfill(2)
                + " "
                + df[hh_col].str.strip().str.zfill(2)
                + ":00:00"
            )
        return pd.to_datetime(
            df["YYYY"].str.strip()
            + "-"
            + df["MM"].str.strip().str.zfill(2)
            + "-"
            + df["DD"].str.strip().str.zfill(2)
        )

    def _parse_qobs_file(self, gauge_id: int) -> pd.DataFrame | None:
        """Parse one D_gauges discharge CSV -> DataFrame[ts, qobs]."""
        path = self._qobs_dir() / f"ID_{gauge_id}.csv"
        if not path.exists():
            return None

        df = pd.read_csv(path, sep=";", dtype=str)
        df.columns = df.columns.str.strip()

        try:
            df["ts"] = self._parse_datetime(df)
        except Exception:
            return None

        qobs_col = next((c for c in df.columns if c.lower() == "qobs"), None)
        if qobs_col is None:
            return None

        df["qobs"] = pd.to_numeric(df[qobs_col], errors="coerce")
        df.loc[df["qobs"] == _MISSING, "qobs"] = float("nan")

        return df[["ts", "qobs"]].copy()

    def _parse_met_file(self, gauge_id: int) -> pd.DataFrame | None:
        """Parse one B_basins met CSV -> DataFrame[ts, met_col1, met_col2, ...]."""
        path = self._met_dir() / f"ID_{gauge_id}.csv"
        if not path.exists():
            return None

        df = pd.read_csv(path, sep=";", dtype=str)
        df.columns = df.columns.str.strip()

        try:
            df["ts"] = self._parse_datetime(df)
        except Exception:
            return None

        # Keep only the met columns that exist in this file
        available = [c for c in _MET_COLUMNS if c in df.columns]
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["ts"] + available].copy()

    def _build_series(
        self,
        gauge_ids: list[int],
        static_lookup: dict[int, dict[str, float]],
    ) -> pd.DataFrame:
        """
        For each gauge, merge D_gauges qobs with B_basins met forcings on date,
        stamp static catchment features, then concatenate into one long-format
        DataFrame.
        """
        frames: list[pd.DataFrame] = []
        missing_qobs = 0
        missing_met = 0

        for gid in gauge_ids:
            qobs = self._parse_qobs_file(gid)
            if qobs is None or qobs.empty:
                missing_qobs += 1
                continue

            met = self._parse_met_file(gid)
            if met is None or met.empty:
                missing_met += 1
                # No B-basin met file -> non-delineated gauge (artificial
                # channel / karst spring).  These would be isolated nodes
                # with all dynamic features as NaN, causing NaN gradients
                # in GNN message-passing layers.  Skip them entirely.
                continue

            # Left merge: keep all qobs dates; B timeseries extend to 2019
            # but D stops at 2017 -- the merge naturally trims to the
            # overlapping range.
            merged = qobs.merge(met, on="ts", how="left")

            # Stamp static features (constant across all timesteps)
            static = static_lookup.get(gid, {})
            for col in _STATIC_COLUMNS:
                merged[col] = static.get(col, np.nan)

            merged["node_id"] = str(gid)
            frames.append(merged)

        if missing_qobs:
            print(f"[LamaHCE] {missing_qobs} qobs files not found -- skipped.")
        if missing_met:
            print(f"[LamaHCE] {missing_met} non-delineated gauges dropped "
                  f"(no B-basin met file -- would have all-NaN features).")

        if not frames:
            raise RuntimeError(
                f"No time series files found under {self._qobs_dir()}. "
                "Check that data_root and resolution are correct."
            )

        series = pd.concat(frames, ignore_index=True)

        # Fill sparse qobs gaps via per-node temporal interpolation.
        # LamaH-CE already gap-fills up to 6 h internally; only a handful of
        # NaN remain after the 5% gap filter and they would otherwise
        # propagate as NaN gradients through GNN message passing.
        series["qobs"] = series.groupby("node_id")["qobs"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

        series = series.sort_values(["ts", "node_id"]).reset_index(drop=True)
        return series

    def _build_edges(self, valid_ids: set[int]) -> pd.DataFrame | None:
        """
        Build directed river-network edges from B_basins_intermediate_all.

        Sources:
          - Gauge_hierarchy.csv  -> NEXTDOWNID (which gauge is directly downstream)
          - Stream_dist.csv      -> dist_hdn   (flow-path distance in km)

        Uses transitive look-ahead: if the immediate downstream gauge was
        filtered out, follows the chain until a valid gauge is found,
        accumulating flow-path distance.  This prevents the gap filter from
        fragmenting the river network (e.g. A -> [removed B] -> C becomes
        A -> C with cost = dist(A->B) + dist(B->C)).

        Only edges where both src and dst are in valid_ids are kept.
        Returns None if the topology files are missing.
        """
        b_attrs = self.data_root / "B_basins_intermediate_all" / "1_attributes"
        hierarchy_path = b_attrs / "Gauge_hierarchy.csv"
        stream_dist_path = b_attrs / "Stream_dist.csv"

        if not hierarchy_path.exists():
            print(f"[LamaHCE] {hierarchy_path} not found -- no edges.")
            return None

        # Load NEXTDOWNID into a lookup dict {id: next_downstream_id}
        hier = pd.read_csv(hierarchy_path, sep=";")
        hier.columns = hier.columns.str.strip()
        hier["ID"] = hier["ID"].astype(int)
        hier["NEXTDOWNID"] = pd.to_numeric(hier["NEXTDOWNID"], errors="coerce")

        id_to_next: dict[int, int] = {}
        for _, row in hier.iterrows():
            nxt = row["NEXTDOWNID"]
            if pd.notna(nxt) and int(nxt) > 0:
                id_to_next[int(row["ID"])] = int(nxt)

        # Load flow distances (km) if available
        id_to_dist: dict[int, float] = {}
        if stream_dist_path.exists():
            dist = pd.read_csv(stream_dist_path, sep=";")
            dist.columns = dist.columns.str.strip()
            dist["ID"] = dist["ID"].astype(int)
            dist["dist_hdn"] = pd.to_numeric(dist["dist_hdn"], errors="coerce").fillna(1.0)
            id_to_dist = dict(zip(dist["ID"], dist["dist_hdn"]))

        rows: list[tuple[str, str, float]] = []
        skipped_no_downstream = 0
        bridged = 0

        for src in valid_ids:
            if src not in id_to_next:
                continue

            # Walk downstream from src, accumulating distance, until we
            # reach a valid gauge or the end of the river.
            cursor = id_to_next[src]
            curr_dist = id_to_dist.get(src, 1.0)
            visited: set[int] = {src}
            hops = 0

            while cursor > 0 and cursor not in valid_ids:
                if cursor in visited:
                    break  # cycle guard
                visited.add(cursor)
                curr_dist += id_to_dist.get(cursor, 1.0)
                nxt = id_to_next.get(cursor)
                if nxt is None:
                    cursor = 0  # end of chain
                    break
                cursor = nxt
                hops += 1

            if cursor > 0 and cursor in valid_ids:
                rows.append((str(src), str(cursor), float(curr_dist)))
                if hops > 0:
                    bridged += 1
            else:
                skipped_no_downstream += 1

        if not rows:
            print("[LamaHCE] No valid river-network edges found in valid gauge set.")
            return None

        print(f"[LamaHCE] {len(rows)} river-network edges built "
              f"({bridged} bridged over filtered-out gauges, "
              f"{skipped_no_downstream} endpoints with no valid downstream).")
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])
