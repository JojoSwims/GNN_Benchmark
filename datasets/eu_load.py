"""ENTSO-E European zonal electricity load dataset loader."""

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


# Google Drive IDs for the two source files.
EDGES_GDRIVE_ID = "12QPWAwGr-36cLQ3TB-KbGE34om3TODH1"
LOAD_GDRIVE_ID = "1mLjtYlbu4cTIAJKDjlD2dS-Zdq-n0ED0"

FREQ = "1h"


@dataclass
class EULoadLoader(DatasetLoader):
    """
    Loader for ENTSO-E European zonal electricity load.

    Downloads two TSV files from Google Drive:
    - An edge list (zone_a, zone_b) of cross-zone interconnections.
    - Hourly TotalLoad (MW) per ENTSO-E bidding zone.

    Edges are undirected and unweighted: the adjacency matrix has 1 wherever
    a zone pair is connected and 0 otherwise (zero diagonal). Node order is
    the alphabetically sorted union of zone codes appearing in either file,
    so the series tensor and the adjacency matrix index into the same nodes.
    """

    _node_order: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="eu_load",
            url=f"https://drive.google.com/uc?id={LOAD_GDRIVE_ID}",
            frequency=FREQ,
            node_order=list(self._node_order),
            feature_columns=["load"],
            units={"load": "MW"},
            description=(
                "ENTSO-E EU zonal hourly electricity load (TotalLoad, MW). "
                "Edges are undirected cross-zone interconnections (cost 1 if "
                "connected, 0 otherwise). Nodes are bidding-zone codes ordered "
                "alphabetically; the series tensor and the adjacency share "
                "the same ordering."
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download both files from Drive and build series + edges."""
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required for EULoad dataset. "
                "Install with: pip install gdown"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            edges_path = tmpdir / "eu_load_edges.csv"
            load_path = tmpdir / "eu_load_series.csv"

            print("Downloading EU load edges from Google Drive...")
            gdown.download(id=EDGES_GDRIVE_ID, output=str(edges_path), quiet=False)
            print("Downloading EU load series from Google Drive...")
            gdown.download(id=LOAD_GDRIVE_ID, output=str(load_path), quiet=False)

            edges_raw = self._read_edges(edges_path)
            load_raw = self._read_load(load_path)

            # Union of zones across both files, alphabetically sorted, so that
            # the series tensor and the adjacency matrix share the same axis.
            zones_edges = set(edges_raw["zone_a"]).union(edges_raw["zone_b"])
            zones_load = set(load_raw["node_id"].unique())
            self._node_order = sorted(zones_edges | zones_load)

            series_df = self._build_series(load_raw, self._node_order)
            edges_df = self._build_edges(edges_raw, self._node_order)

            return series_df, edges_df

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
    def _build_edges(
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
