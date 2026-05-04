"""LamaH-CE dataset loader — dynamic features only.

Same river network and gauges as lamah_ce.py, but the feature vector contains
only the prediction target (qobs) and 8 curated ERA5-Land dynamic forcings.
No static catchment attributes are included.

Features per node per timestep (C = 9):
  - qobs              : observed discharge (m3/s) — prediction target
  - prec              : precipitation (mm)
  - 2m_temp_mean      : air temperature (K)
  - total_et          : evapotranspiration (mm)
  - swe               : snow water equivalent (mm)
  - surf_net_solar_rad_mean : net solar radiation (J/m2)
  - volsw_123         : soil moisture 0–100 cm (m3/m3)
  - lai_high_veg      : leaf area index, high vegetation (m2/m2)
  - lai_low_veg       : leaf area index, low vegetation (m2/m2)

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

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.datasets._lamah_ce_download import ensure_lamah_data_root

ZENODO_URL = "https://doi.org/10.5281/zenodo.4525244"

# Sentinel value used in LamaH-CE for missing data
_MISSING = -999.0

# ── Dynamic features (time-series from B_basins_intermediate_all) ────────────
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


@dataclass
class LamaHCELoader(DatasetLoader):
    """
    Loader for LamaH-CE — dynamic features only (no static catchment attrs).

    Same river network topology and gauge filtering as LamaHCELoader, but the
    feature vector contains only qobs + 8 ERA5-Land dynamic forcings (C = 9).

    Args:
        data_root: Path to the extracted LamaH-CE root directory. If None
            (the default), the Zenodo tarball is fetched and extracted
            automatically into ``~/.cache/gnn_benchmark/lamah_ce/``.
        resolution: "daily" or "hourly".
        max_gap_fraction: Exclude gauges where the fraction of remaining gaps
            exceeds this threshold. Default 0.05 (5 %).
        cache_dir: Override the auto-download cache location.
    """

    data_root: Path | None = None
    resolution: str = "daily"
    max_gap_fraction: float = 0.05
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.data_root is not None:
            self.data_root = Path(self.data_root)
        self._node_order: list[str] = []

    # ── DatasetLoader interface ───────────────────────────────────────────────

    @property
    def info(self) -> DatasetInfo:
        freq = "1D" if self.resolution == "daily" else "1h"
        feature_cols = ["qobs"] + _MET_COLUMNS
        units = {"qobs": "m3/s", **_MET_UNITS}
        return DatasetInfo(
            name=f"lamah_ce_{self.resolution}",
            url=ZENODO_URL,
            frequency=freq,
            node_order=self._node_order,
            feature_columns=feature_cols,
            units=units,
            window_config=WindowConfig(target_columns=["qobs"]),
            description=(
                f"LamaH-CE Central Europe streamflow network (dynamic only), "
                f"{self.resolution} resolution, up to 859 gauges, "
                f"{len(feature_cols)} features (qobs + {len(_MET_COLUMNS)} "
                f"ERA5-Land met forcings)"
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Read LamaH-CE from a local extracted directory."""
        self._check_data_root()

        # 1. Load gauge metadata
        gauge_attrs = self._read_gauge_attrs()

        # 2. Filter by gap fraction
        if "gaps_post" in gauge_attrs.columns:
            gauge_attrs = gauge_attrs[
                gauge_attrs["gaps_post"] <= self.max_gap_fraction
            ].copy()

        valid_ids: list[int] = sorted(gauge_attrs["ID"].tolist())
        print(f"[LamaHCE] {len(valid_ids)} gauges after gap filter "
              f"(max_gap_fraction={self.max_gap_fraction})")

        # 3. Read time series for all valid gauges
        series_df = self._build_series(valid_ids)

        # 4. Determine final node set
        present_ids = sorted(series_df["node_id"].unique(), key=int)
        self._node_order = [str(i) for i in present_ids]

        # 5. Build graph edges from river network topology
        present_id_set = {int(n) for n in present_ids}
        edges_df = self._build_edges(present_id_set)

        return series_df, edges_df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_data_root(self) -> None:
        # Auto-download + extract on first use when no explicit path was given.
        if self.data_root is None:
            self.data_root = ensure_lamah_data_root(
                resolution=self.resolution,
                cache_dir=self.cache_dir,
            )
            return
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"data_root not found: {self.data_root}\n"
                "Either pass data_root=None to auto-download from Zenodo, "
                "or extract the archive to the expected path."
            )
        gauge_dir = self.data_root / "D_gauges"
        if not gauge_dir.exists():
            raise FileNotFoundError(
                f"Expected 'D_gauges/' inside data_root={self.data_root}. "
                "Make sure you extracted the archive correctly."
            )

    def _read_gauge_attrs(self) -> pd.DataFrame:
        path = self.data_root / "D_gauges" / "1_attributes" / "Gauge_attributes.csv"
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(int)
        return df

    def _qobs_dir(self) -> Path:
        return self.data_root / "D_gauges" / "2_timeseries" / self.resolution

    def _met_dir(self) -> Path:
        return (
            self.data_root
            / "B_basins_intermediate_all"
            / "2_timeseries"
            / self.resolution
        )

    def _parse_datetime(self, df: pd.DataFrame) -> pd.Series:
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
        path = self._met_dir() / f"ID_{gauge_id}.csv"
        if not path.exists():
            return None

        df = pd.read_csv(path, sep=";", dtype=str)
        df.columns = df.columns.str.strip()

        try:
            df["ts"] = self._parse_datetime(df)
        except Exception:
            return None

        available = [c for c in _MET_COLUMNS if c in df.columns]
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["ts"] + available].copy()

    def _build_series(self, gauge_ids: list[int]) -> pd.DataFrame:
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
                continue

            merged = qobs.merge(met, on="ts", how="left")
            merged["node_id"] = str(gid)
            frames.append(merged)

        if missing_qobs:
            print(f"[LamaHCE] {missing_qobs} qobs files not found -- skipped.")
        if missing_met:
            print(f"[LamaHCE] {missing_met} non-delineated gauges dropped.")

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
        b_attrs = self.data_root / "B_basins_intermediate_all" / "1_attributes"
        hierarchy_path = b_attrs / "Gauge_hierarchy.csv"
        stream_dist_path = b_attrs / "Stream_dist.csv"

        if not hierarchy_path.exists():
            print(f"[LamaHCE] {hierarchy_path} not found -- no edges.")
            return None

        hier = pd.read_csv(hierarchy_path, sep=";")
        hier.columns = hier.columns.str.strip()
        hier["ID"] = hier["ID"].astype(int)
        hier["NEXTDOWNID"] = pd.to_numeric(hier["NEXTDOWNID"], errors="coerce")

        id_to_next: dict[int, int] = {}
        for _, row in hier.iterrows():
            nxt = row["NEXTDOWNID"]
            if pd.notna(nxt) and int(nxt) > 0:
                id_to_next[int(row["ID"])] = int(nxt)

        id_to_dist: dict[int, float] = {}
        if stream_dist_path.exists():
            dist = pd.read_csv(stream_dist_path, sep=";")
            dist.columns = dist.columns.str.strip()
            dist["ID"] = dist["ID"].astype(int)
            dist["dist_hdn"] = pd.to_numeric(
                dist["dist_hdn"], errors="coerce"
            ).fillna(1.0)
            id_to_dist = dict(zip(dist["ID"], dist["dist_hdn"]))

        rows: list[tuple[str, str, float]] = []
        skipped_no_downstream = 0
        bridged = 0

        for src in valid_ids:
            if src not in id_to_next:
                continue

            cursor = id_to_next[src]
            curr_dist = id_to_dist.get(src, 1.0)
            visited: set[int] = {src}
            hops = 0

            while cursor > 0 and cursor not in valid_ids:
                if cursor in visited:
                    break
                visited.add(cursor)
                curr_dist += id_to_dist.get(cursor, 1.0)
                nxt = id_to_next.get(cursor)
                if nxt is None:
                    cursor = 0
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
            print("[LamaHCE] No valid river-network edges found.")
            return None

        print(f"[LamaHCE] {len(rows)} river-network edges built "
              f"({bridged} bridged over filtered-out gauges, "
              f"{skipped_no_downstream} endpoints with no valid downstream).")
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])


class LamaHCEDynamicLoader(LamaHCELoader):
    """Deprecated alias for :class:`LamaHCELoader`."""

