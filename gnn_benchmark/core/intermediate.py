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
    from gnn_benchmark.transforms.base import Transform


class IntermediateRepresentation:
    """
    The central data object for GNN Benchmark.

    Wraps a series DataFrame in long format (ts, node_id, features...) and
    optional edges DataFrame. Linked to a workspace for persistence.

    Attributes:
        series: DataFrame with columns [ts, node_id, feature1, feature2, ...]
        metadata: IRMetadata with dataset information and transform history
        edges: Optional DataFrame with columns [src, dst, cost]
        mask: Optional DataFrame with observation mask
    """

    def __init__(
        self,
        series: pd.DataFrame,
        metadata: IRMetadata,
        edges: pd.DataFrame | None = None,
        mask: pd.DataFrame | None = None,
        workspace: DataWorkspace | None = None,
        dataset_name: str | None = None,
    ):
        self.series = series
        self.metadata = metadata
        self.edges = edges
        self.mask = mask
        self._workspace = workspace
        self._dataset_name = dataset_name

        # Ensure ts column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.series["ts"]):
            self.series["ts"] = pd.to_datetime(self.series["ts"])

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

    # --- Transforms ---

    def apply(self, transform: Transform) -> IntermediateRepresentation:
        """
        Apply a transform in-place and record in history.

        Args:
            transform: The transform to apply

        Returns:
            self for method chaining
        """
        transform(self)
        self.metadata.transform_history.append(transform.description)
        return self

    def reset_transforms(self) -> IntermediateRepresentation:
        """
        Reset to clean state from workspace.

        Reloads the clean version from workspace and replaces current data.

        Returns:
            self for method chaining

        Raises:
            ValueError: If not linked to a workspace
        """
        if not self._workspace or not self._dataset_name:
            raise ValueError("Cannot reset: IR not linked to workspace")

        clean = self._workspace.load(self._dataset_name, from_clean=True)
        self.series = clean.series.copy()
        self.edges = clean.edges.copy() if clean.edges is not None else None
        self.mask = clean.mask.copy() if clean.mask is not None else None
        self.metadata.transform_history = []
        self.metadata.feature_columns = clean.metadata.feature_columns.copy()

        return self

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

    def get_observation_mask(self) -> np.ndarray | None:
        """
        Get observation mask as (T, N) boolean array.

        Returns:
            Boolean array where True indicates observed values, or None if no mask
        """
        if self.mask is None:
            return None

        timestamps = self.timestamps
        nodes = self.nodes
        node_to_idx = {str(n): i for i, n in enumerate(nodes)}
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        T, N = len(timestamps), len(nodes)
        mask_arr = np.ones((T, N), dtype=bool)

        for _, row in self.mask.iterrows():
            t_idx = ts_to_idx.get(row["ts"])
            n_idx = node_to_idx.get(str(row["node_id"]))
            if t_idx is not None and n_idx is not None:
                mask_arr[t_idx, n_idx] = bool(row["is_observed"])

        return mask_arr

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

        # Save mask if present
        if self.mask is not None:
            self.mask.to_csv(path / "mask.csv", index=False)

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

        # Load mask if present
        mask = None
        if (path / "mask.csv").exists():
            mask = pd.read_csv(path / "mask.csv")
            mask["ts"] = pd.to_datetime(mask["ts"])
            mask["node_id"] = mask["node_id"].astype(str)

        return cls(
            series=series,
            metadata=metadata,
            edges=edges,
            mask=mask,
            workspace=workspace,
            dataset_name=dataset_name,
        )

    def __repr__(self) -> str:
        T, N, C = self.shape
        return (
            f"IntermediateRepresentation("
            f"name={self.metadata.name!r}, "
            f"shape=(T={T}, N={N}, C={C}), "
            f"transforms={len(self.metadata.transform_history)})"
        )
