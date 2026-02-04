"""D2STGCN model exporter."""

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
class D2STGCNExporter(ModelExporter):
    """
    Export to D2STGCN format.

    Creates:
        - data.npz: Full tensor (T, N, C)
        - train.npz, val.npz, test.npz: Pre-windowed data
    """

    @property
    def name(self) -> str:
        return "d2stgcn"

    def export_to_directory(
        self,
        ir: IntermediateRepresentation,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """Export IR to D2STGCN format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get columns to use
        input_cols = window_config.input_columns or ir.feature_columns

        # Convert to tensor
        data = ir.to_tensor(columns=input_cols)

        # Save full data
        data_path = output_dir / "data.npz"
        np.savez_compressed(
            data_path,
            data=data.astype(np.float32),
            timestamps=ir.timestamps.astype(str),
            node_ids=np.array(ir.nodes),
        )

        # Create sliding windows
        x, y = self._create_sliding_windows(
            data,
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

        # Save windowed data
        files = {"data": data_path}

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
            )
            files[split_name] = npz_path

        # Save adjacency if available
        if ir.edges is not None:
            adj = ir.get_adjacency_matrix()
            adj_path = output_dir / "adj_mx.npy"
            np.save(adj_path, adj)
            files["adj"] = adj_path

        return ExportResult(
            files=files,
            window_config=window_config,
            split_config=split_config,
            metadata={
                "data_shape": data.shape,
                "train_samples": len(x_train),
                "val_samples": len(x_val),
                "test_samples": len(x_test),
            },
        )
