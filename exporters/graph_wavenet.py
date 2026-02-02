"""GraphWaveNet model exporter."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.exporters.base import (
    ExportResult,
    ModelExporter,
    SplitConfig,
    WindowConfig,
)


@dataclass
class GraphWaveNetExporter(ModelExporter):
    """
    Export to GraphWaveNet format.

    Creates:
        - train.npz, val.npz, test.npz: Pre-windowed data
        - adj_mx.pkl: Adjacency matrix in DCRNN format

    Each NPZ file contains:
        - x: Input windows (S, L, N, C)
        - y: Target windows (S, H, N, C)
        - x_offsets: Input time offsets
        - y_offsets: Target time offsets
    """

    @property
    def name(self) -> str:
        return "graph_wavenet"

    def export(
        self,
        ir: IntermediateRepresentation,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """Export IR to GraphWaveNet format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get columns to use
        input_cols = window_config.input_columns or ir.feature_columns
        target_cols = window_config.target_columns or input_cols

        # Convert to tensor
        data = ir.to_tensor(columns=input_cols)

        # Create sliding windows
        x, y = self._create_sliding_windows(
            data,
            window_config.input_length,
            window_config.horizon,
            window_config.y_start,
        )

        # If target columns differ from input, create separate target tensor
        if target_cols != input_cols:
            target_data = ir.to_tensor(columns=target_cols)
            _, y = self._create_sliding_windows(
                target_data,
                window_config.input_length,
                window_config.horizon,
                window_config.y_start,
            )

        # Split by time
        x_train, x_val, x_test = self._split_by_time(
            x, split_config.train_ratio, split_config.val_ratio
        )
        y_train, y_val, y_test = self._split_by_time(
            y, split_config.train_ratio, split_config.val_ratio
        )

        # Compute offsets
        x_offsets, y_offsets = self._compute_offsets(
            window_config.input_length,
            window_config.horizon,
            window_config.y_start,
        )

        # Save NPZ files
        files = {}

        for split_name, x_data, y_data in [
            ("train", x_train, y_train),
            ("val", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            npz_path = output_dir / f"{split_name}.npz"
            np.savez_compressed(
                npz_path,
                x=x_data.astype(np.float32),
                y=y_data.astype(np.float32),
                x_offsets=x_offsets,
                y_offsets=y_offsets,
            )
            files[split_name] = npz_path

        # Save adjacency matrix in DCRNN format
        if ir.edges is not None:
            adj_path = output_dir / "adj_mx.pkl"
            self._save_adjacency_pickle(ir, adj_path)
            files["adj"] = adj_path

        return ExportResult(
            files=files,
            window_config=window_config,
            split_config=split_config,
            metadata={
                "train_samples": len(x_train),
                "val_samples": len(x_val),
                "test_samples": len(x_test),
                "input_shape": x_train.shape,
                "target_shape": y_train.shape,
            },
        )

    def _save_adjacency_pickle(
        self, ir: IntermediateRepresentation, path: Path
    ) -> None:
        """Save adjacency matrix in DCRNN pickle format."""
        nodes = ir.nodes
        id_to_ind = {node: i for i, node in enumerate(nodes)}
        adj = ir.get_adjacency_matrix()

        # DCRNN format: (node_ids, id_to_idx_dict, adj_matrix)
        with open(path, "wb") as f:
            pickle.dump((nodes, id_to_ind, adj), f)
