"""ENTSO-E European zonal electricity load dataset loader."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# Google Drive IDs for the source files.
EDGES_GDRIVE_ID = "1Fxz9GWMGnOVXnrAr7A_e7xb2zMoYQxQW"
LOAD_GDRIVE_ID = "1GKd6PCu-Y_Q771IgBsAELp5e8LHwKGR_"
# Hourly directed cross-zone flow snapshots.
DYNAMIC_EDGES_GDRIVE_ID = "1K-ZR6UToCdIMhYUEIQ-a15h4Ebe8mwZE"

FREQ = "1h"

EDGE_MODES = ("static", "dynamic", "both")


@dataclass
class EULoadLoader(DatasetLoader):
    """
    Loader for ENTSO-E European zonal electricity load.

    Downloads files from Google Drive:
    - Hourly TotalLoad (MW) per ENTSO-E bidding zone (always).
    - Static edge list (zone_a, zone_b) of cross-zone interconnections
      (when ``edges_mode`` ∈ {"static", "both"}).
    - Hourly directed cross-zone flow snapshots [ts, src, dst, flow]
      (when ``edges_mode`` ∈ {"dynamic", "both"}).

    Static edges are undirected and unweighted: the adjacency matrix has
    1 wherever a zone pair is connected and 0 otherwise (zero diagonal).

    Dynamic edges are directed: ``adj[t][src, dst]`` is the cross-zone
    flow from ``src`` to ``dst`` in the hour starting at ``t`` (MW).
    Sparse — only rows with ``cost != 0`` are emitted. Self-loops are
    dropped. Snapshots are bucketed by their stamped hour, matching the
    load file's hourly grid.

    Node order is the alphabetically sorted union of zone codes
    appearing in any of the loaded files, so the series tensor and the
    adjacency matrices share the same axis.

    Args:
        edges_mode: One of ``"static"``, ``"dynamic"``, ``"both"``.
            Defaults to ``"static"`` to preserve the original behaviour.
            The chosen mode is reflected in the dataset name (suffix
            ``_s`` / ``_d`` / ``_sd``) so the workspace cache keys the
            three shapes separately.
    """

    edges_mode: str = "static"

    _node_order: list[str] = field(default_factory=list, init=False, repr=False)
    _dynamic_edges_df: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.edges_mode not in EDGE_MODES:
            raise ValueError(
                f"edges_mode must be one of {list(EDGE_MODES)}, "
                f"got {self.edges_mode!r}"
            )

    @property
    def info(self) -> DatasetInfo:
        mode_desc = {
            "static": "undirected interconnection edges only.",
            "dynamic": "hourly directed cross-zone flow snapshots only (MW).",
            "both": (
                "both undirected interconnection edges and hourly "
                "directed cross-zone flow snapshots (MW)."
            ),
        }[self.edges_mode]
        # Keep the legacy name "eu_load" for the static-only default so
        # existing caches and downstream registry references aren't
        # invalidated; tag the new modes for cache isolation.
        name_suffix = {"static": "", "dynamic": "_d", "both": "_sd"}[
            self.edges_mode
        ]
        return DatasetInfo(
            name=f"eu_load{name_suffix}",
            url=f"https://drive.google.com/uc?id={LOAD_GDRIVE_ID}",
            frequency=FREQ,
            node_order=list(self._node_order),
            feature_columns=["load"],
            units={"load": "MW"},
            description=(
                "ENTSO-E EU zonal hourly electricity load (TotalLoad, MW). "
                "Nodes are bidding-zone codes ordered alphabetically; the "
                "series tensor and the adjacency share the same ordering. "
                "Both the load series and any dynamic edges are stamped at "
                "period-end: timestamp t covers the closed interval "
                "(t-1h, t], so load[t] and adj[t] describe the same hour "
                "and a model predicting t+1 from inputs ending at t never "
                "consumes future information. "
                f"Edge mode = {self.edges_mode!r}: {mode_desc}"
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download the requested files from Drive and build the IR tables."""
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required for EULoad dataset. "
                "Install with: pip install gdown"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            load_path = tmpdir / "eu_load_series.csv"

            print("Downloading EU load series from Google Drive...")
            gdown.download(id=LOAD_GDRIVE_ID, output=str(load_path), quiet=False)
            load_raw = self._read_load(load_path)

            edges_raw: pd.DataFrame | None = None
            if self.edges_mode in ("static", "both"):
                edges_path = tmpdir / "eu_load_edges.csv"
                print("Downloading EU load static edges from Google Drive...")
                gdown.download(
                    id=EDGES_GDRIVE_ID, output=str(edges_path), quiet=False
                )
                edges_raw = self._read_edges(edges_path)

            dyn_raw: pd.DataFrame | None = None
            if self.edges_mode in ("dynamic", "both"):
                dyn_path = tmpdir / "dynamic_edges.csv"
                print(
                    "Downloading EU load dynamic edges from Google Drive..."
                )
                gdown.download(
                    id=DYNAMIC_EDGES_GDRIVE_ID,
                    output=str(dyn_path),
                    quiet=False,
                )
                dyn_raw = self._read_dynamic_edges(dyn_path)

            # Union of zones across every file we touched, alphabetically
            # sorted, so the series tensor and both adjacency flavours
            # share the same axis.
            zones: set[str] = set(load_raw["node_id"].unique())
            if edges_raw is not None:
                zones |= set(edges_raw["zone_a"]) | set(edges_raw["zone_b"])
            if dyn_raw is not None:
                zones |= set(dyn_raw["src"]) | set(dyn_raw["dst"])
            self._node_order = sorted(zones)

            series_df = self._build_series(load_raw, self._node_order)

            edges_df: pd.DataFrame | None = None
            if edges_raw is not None:
                edges_df = self._build_static_edges(edges_raw, self._node_order)

            self._dynamic_edges_df = None
            if dyn_raw is not None:
                self._dynamic_edges_df = self._build_dynamic_edges(
                    dyn_raw, self._node_order
                )

            return series_df, edges_df

    def build_dynamic_edges(self) -> pd.DataFrame | None:
        """Return the hourly snapshot edges DataFrame, or None if not built."""
        return self._dynamic_edges_df

    @staticmethod
    def _read_edges(path: Path) -> pd.DataFrame:
        """Parse the (zone_a, zone_b) interconnection list."""
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df["zone_a"] = df["zone_a"].astype(str).str.strip()
        df["zone_b"] = df["zone_b"].astype(str).str.strip()
        return df.dropna(subset=["zone_a", "zone_b"])

    @staticmethod
    def _read_load(path: Path) -> pd.DataFrame:
        """Parse the hourly TotalLoad CSV into [ts, node_id, load]."""
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            "DateTime(UTC)": "ts",
            "AreaMapCode": "node_id",
            "TotalLoad[MW]": "load",
        })
        df["node_id"] = df["node_id"].astype(str).str.strip()
        # Source uses single-digit hours like "2023-01-01 0:00", so let
        # pandas infer the format rather than locking in "%H:%M".
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["load"] = pd.to_numeric(df["load"], errors="coerce")
        return df.dropna(subset=["ts", "node_id"])

    @staticmethod
    def _read_dynamic_edges(path: Path) -> pd.DataFrame:
        """Parse the hourly directed flow CSV into [ts, src, dst, flow]."""
        df = pd.read_csv(
            path,
            dtype={"src": str, "dst": str},
        )
        df.columns = [c.strip() for c in df.columns]
        df["src"] = df["src"].astype(str).str.strip()
        df["dst"] = df["dst"].astype(str).str.strip()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
        return df.dropna(subset=["ts", "src", "dst", "flow"])

    @staticmethod
    def _build_series(
        load_raw: pd.DataFrame, node_order: list[str]
    ) -> pd.DataFrame:
        """Densify zonal load to a full hourly grid; missing entries → NaN."""
        # Collapse accidental (ts, zone) duplicates by mean before densifying.
        out = (
            load_raw[["ts", "node_id", "load"]]
            .groupby(["ts", "node_id"], as_index=False)["load"]
            .mean()
        )

        full_ts = pd.date_range(out["ts"].min(), out["ts"].max(), freq=FREQ)
        node_idx = pd.Index(node_order, dtype=str)
        grid = pd.MultiIndex.from_product(
            [full_ts, node_idx], names=["ts", "node_id"]
        ).to_frame(index=False)

        return (
            grid.merge(out, on=["ts", "node_id"], how="left")
            .sort_values(["ts", "node_id"])
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_static_edges(
        edges_raw: pd.DataFrame, node_order: list[str]
    ) -> pd.DataFrame:
        """Emit symmetric (src, dst, cost=1) rows for each undirected edge.

        ``IntermediateRepresentation.get_adjacency_matrix`` only fills the
        explicit (src, dst) entries it sees, so we mirror each pair to keep
        the adjacency symmetric. Missing pairs stay at the default 0.
        """
        node_set = set(node_order)
        rows: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for a, b in edges_raw[["zone_a", "zone_b"]].itertuples(index=False):
            if a == b or a not in node_set or b not in node_set:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            rows.append((a, b, 1.0))
            rows.append((b, a, 1.0))
        return pd.DataFrame(rows, columns=["src", "dst", "cost"])

    @staticmethod
    def _build_dynamic_edges(
        dyn_raw: pd.DataFrame, node_order: list[str]
    ) -> pd.DataFrame:
        """Hourly directed snapshots [ts, src, dst, cost] (MW).

        Drops self-loops and zero-flow rows for sparsity; collapses any
        accidental duplicate (ts, src, dst) triples by sum.
        """
        node_set = set(node_order)
        df = dyn_raw[
            (dyn_raw["src"] != dyn_raw["dst"])
            & dyn_raw["src"].isin(node_set)
            & dyn_raw["dst"].isin(node_set)
            & (dyn_raw["flow"] != 0.0)
        ].copy()
        df = (
            df.groupby(["ts", "src", "dst"], as_index=False)["flow"]
            .sum()
            .rename(columns={"flow": "cost"})
        )
        df["cost"] = df["cost"].astype("float32")
        return df.sort_values(["ts", "src", "dst"]).reset_index(drop=True)
