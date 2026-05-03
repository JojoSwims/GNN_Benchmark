"""Intermediate Representation class for GNN Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import IRMetadata

if TYPE_CHECKING:
    from gnn_benchmark.core.workspace import DataWorkspace


class IntermediateRepresentation:
    """
    The central data object for GNN Benchmark.

    Wraps a series DataFrame in long format (ts, node_id, features...) and
    optional edges DataFrame. Linked to a workspace for persistence.

    Attributes:
        series: DataFrame with columns [ts, node_id, feature1, feature2, ...]
        metadata: IRMetadata with dataset information
        edges: Optional DataFrame with columns [src, dst, cost]
    """

    def __init__(
        self,
        series: pd.DataFrame,
        metadata: IRMetadata,
        edges: pd.DataFrame | None = None,
        dynamic_edges: pd.DataFrame | None = None,
        workspace: DataWorkspace | None = None,
        dataset_name: str | None = None,
    ):
        self.series = series
        self.metadata = metadata
        self.edges = edges
        # Snapshot DataFrame with columns [ts, src, dst, cost]. Sparse — only
        # rows with cost >= 1 are emitted. ts is the window start; bucketing
        # convention is dataset-specific and documented in metadata.description.
        self.dynamic_edges = dynamic_edges
        self._workspace = workspace
        self._dataset_name = dataset_name

        # Ensure ts column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.series["ts"]):
            self.series["ts"] = pd.to_datetime(self.series["ts"])

        if (
            self.dynamic_edges is not None
            and "ts" in self.dynamic_edges.columns
            and not pd.api.types.is_datetime64_any_dtype(self.dynamic_edges["ts"])
        ):
            self.dynamic_edges["ts"] = pd.to_datetime(self.dynamic_edges["ts"])

    # --- Properties ---

    @property
    def nodes(self) -> list[str]:
        """Ordered list of node IDs."""
        return self.metadata.node_order

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Ordered unique timestamps."""
        return pd.DatetimeIndex(self.series["ts"].unique()).sort_values()

    @property
    def feature_columns(self) -> list[str]:
        """List of value/feature column names."""
        return self.metadata.feature_columns

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (T, N, C) dimensions."""
        T = len(self.timestamps)
        N = len(self.nodes)
        C = len(self.feature_columns)
        return (T, N, C)

    @property
    def workspace(self) -> DataWorkspace | None:
        """The workspace this IR is linked to, if any."""
        return self._workspace

    @property
    def dataset_name(self) -> str | None:
        """The dataset name in the workspace, if linked."""
        return self._dataset_name

    # --- Conversion ---

    def to_tensor(self, columns: list[str] | None = None) -> np.ndarray:
        """
        Convert series to (T, N, C) numpy array.

        Args:
            columns: Feature columns to include. If None, use all feature columns.

        Returns:
            numpy array of shape (T, N, C)
        """
        cols = columns or self.feature_columns
        timestamps = self.timestamps
        nodes = self.nodes
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        T, N, C = len(timestamps), len(nodes), len(cols)
        tensor = np.full((T, N, C), np.nan, dtype=np.float32)

        # Create timestamp to index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        for _, row in self.series.iterrows():
            t_idx = ts_to_idx.get(row["ts"])
            n_idx = node_to_idx.get(str(row["node_id"]))
            if t_idx is not None and n_idx is not None:
                for c_idx, col in enumerate(cols):
                    if col in row:
                        tensor[t_idx, n_idx, c_idx] = row[col]

        return tensor

    def get_dynamic_adjacency_snapshot(
        self, ts: pd.Timestamp, default_diagonal: float = 0.0
    ) -> np.ndarray:
        """
        Return (N, N) adjacency for the dynamic-edge snapshot at `ts`.

        Uses the same node_order as get_adjacency_matrix() so indexing aligns.
        Zero where no edge exists at that snapshot.

        Args:
            ts: Snapshot timestamp (window start). Must match a value in
                ``self.dynamic_edges["ts"]``.
            default_diagonal: Value for diagonal entries (default 0.0).

        Returns:
            NxN numpy array of edge costs for the requested snapshot.

        Raises:
            ValueError: If no dynamic edges are defined.
        """
        if self.dynamic_edges is None:
            raise ValueError("No dynamic edges defined for this IR")

        nodes = self.nodes
        N = len(nodes)
        node_to_idx = {str(n): i for i, n in enumerate(nodes)}

        adj = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(adj, default_diagonal)

        ts = pd.Timestamp(ts)
        slice_df = self.dynamic_edges[self.dynamic_edges["ts"] == ts]
        for _, row in slice_df.iterrows():
            src_idx = node_to_idx.get(str(row["src"]))
            dst_idx = node_to_idx.get(str(row["dst"]))
            if src_idx is not None and dst_idx is not None:
                adj[src_idx, dst_idx] = float(row["cost"])

        return adj

    def get_dynamic_adjacency_full(
        self, default_diagonal: float = 0.0
    ) -> np.ndarray:
        """Return the full ``(T, N, N)`` dynamic-adjacency tensor.

        Densifies every snapshot in :attr:`dynamic_edges` against the
        sorted-unique series timestamps so the time axis aligns with
        :meth:`to_tensor`'s ``T`` dimension. Snapshots absent from
        ``dynamic_edges`` (e.g. quiet hours with no recorded flow)
        contribute an all-zero matrix.

        Implementation note: the :meth:`get_dynamic_adjacency_snapshot`
        helper is fine for one-off lookups but iterates DataFrame rows
        with ``iterrows`` per snapshot, which is O(T·E) and far too slow
        for the harness — building the full tensor hits ~30k snapshots
        for divvy and ~17k for eu_load.  This method does a single
        vectorised scatter over the whole table.

        Returns:
            float32 numpy array of shape ``(T, N, N)``.

        Raises:
            ValueError: If no dynamic edges are defined.
        """
        if self.dynamic_edges is None:
            raise ValueError("No dynamic edges defined for this IR")

        nodes = self.nodes
        N = len(nodes)
        node_to_idx = {str(n): i for i, n in enumerate(nodes)}

        timestamps = self.timestamps
        T = len(timestamps)
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        adj = np.zeros((T, N, N), dtype=np.float32)
        if default_diagonal != 0.0:
            diag = np.broadcast_to(
                np.eye(N, dtype=np.float32) * default_diagonal,
                (T, N, N),
            )
            adj = adj + diag

        df = self.dynamic_edges
        # Vectorise the scatter — pandas .map gives O(E) vs ~O(E*T)
        # for a per-snapshot loop over the table.
        t_idx = df["ts"].map(ts_to_idx).to_numpy()
        s_idx = df["src"].astype(str).map(node_to_idx).to_numpy()
        d_idx = df["dst"].astype(str).map(node_to_idx).to_numpy()
        cost = df["cost"].to_numpy(dtype=np.float32)

        valid = (
            ~pd.isna(t_idx) & ~pd.isna(s_idx) & ~pd.isna(d_idx)
        )
        if not valid.all():
            t_idx = t_idx[valid]
            s_idx = s_idx[valid]
            d_idx = d_idx[valid]
            cost = cost[valid]

        adj[t_idx.astype(np.int64), s_idx.astype(np.int64), d_idx.astype(np.int64)] = cost
        return adj

    def get_adjacency_matrix(self, default_diagonal: float = 0.0) -> np.ndarray:
        """
        Convert edges DataFrame to NxN numpy array.

        Args:
            default_diagonal: Value for diagonal entries (default 0.0)

        Returns:
            NxN numpy array with edge costs

        Raises:
            ValueError: If no edges are defined
        """
        if self.edges is None:
            raise ValueError("No edges defined for this IR")

        nodes = self.nodes
        N = len(nodes)
        node_to_idx = {str(n): i for i, n in enumerate(nodes)}

        adj = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(adj, default_diagonal)

        for _, row in self.edges.iterrows():
            src_idx = node_to_idx.get(str(row["src"]))
            dst_idx = node_to_idx.get(str(row["dst"]))
            if src_idx is not None and dst_idx is not None:
                adj[src_idx, dst_idx] = row["cost"]

        return adj

    # --- Splitting ---

    def get_split_timestamps(
        self, train_ratio: float = 0.7, val_ratio: float = 0.1
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Compute temporal split points.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation

        Returns:
            (train_end, val_end) timestamps
            - Train: ts <= train_end
            - Val: train_end < ts <= val_end
            - Test: ts > val_end
        """
        timestamps = self.timestamps
        n = len(timestamps)

        train_idx = int(n * train_ratio) - 1
        val_idx = int(n * (train_ratio + val_ratio)) - 1

        train_end = timestamps[max(0, train_idx)]
        val_end = timestamps[max(0, val_idx)]

        return train_end, val_end

    # --- Persistence ---

    def save(self, path: Path | None = None) -> Path:
        """
        Save IR to disk.

        Args:
            path: Directory to save to. If None, saves to workspace working directory.

        Returns:
            Path where data was saved

        Raises:
            ValueError: If path is None and not linked to workspace
        """
        if path is None:
            if not self._workspace or not self._dataset_name:
                raise ValueError("Cannot save: no path provided and not linked to workspace")
            path = self._workspace.working_dir / self._dataset_name

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save series
        self.series.to_csv(path / "series.csv", index=False)

        # Save edges if present
        if self.edges is not None:
            self.edges.to_csv(path / "edges.csv", index=False)

        # Save dynamic edges (snapshot-per-row) if present
        if self.dynamic_edges is not None:
            self.dynamic_edges.to_csv(path / "dynamic_edges.csv", index=False)

        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(
        cls,
        path: Path,
        workspace: DataWorkspace | None = None,
        dataset_name: str | None = None,
    ) -> IntermediateRepresentation:
        """
        Load IR from a directory.

        Args:
            path: Directory containing series.csv, metadata.json, etc.
            workspace: Optional workspace to link to
            dataset_name: Optional dataset name for workspace linkage

        Returns:
            Loaded IntermediateRepresentation
        """
        path = Path(path)

        # Load series
        series = pd.read_csv(path / "series.csv")
        series["ts"] = pd.to_datetime(series["ts"])
        series["node_id"] = series["node_id"].astype(str)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = IRMetadata.from_dict(json.load(f))

        # Load edges if present
        edges = None
        if (path / "edges.csv").exists():
            edges = pd.read_csv(path / "edges.csv")
            edges["src"] = edges["src"].astype(str)
            edges["dst"] = edges["dst"].astype(str)

        # Load dynamic edges if present
        dynamic_edges = None
        if (path / "dynamic_edges.csv").exists():
            dynamic_edges = pd.read_csv(
                path / "dynamic_edges.csv", parse_dates=["ts"]
            )
            dynamic_edges["src"] = dynamic_edges["src"].astype(str)
            dynamic_edges["dst"] = dynamic_edges["dst"].astype(str)

        return cls(
            series=series,
            metadata=metadata,
            edges=edges,
            dynamic_edges=dynamic_edges,
            workspace=workspace,
            dataset_name=dataset_name,
        )

    def __repr__(self) -> str:
        T, N, C = self.shape
        return (
            f"IntermediateRepresentation("
            f"name={self.metadata.name!r}, "
            f"shape=(T={T}, N={N}, C={C}))"
        )
