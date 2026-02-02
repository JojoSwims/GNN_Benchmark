"""STAEformer model exporter."""

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
class STAEformerExporter(ModelExporter):
    """
    Export to STAEformer format.

    Creates:
        - data.npz: Contains the full tensor (T, N, C)
        - index.npz: Contains train/val/test index arrays for windowing

    STAEformer performs its own windowing at runtime using the index arrays.
    """

    @property
    def name(self) -> str:
        return "staeformer"

    def export(
        self,
        ir: IntermediateRepresentation,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """Export IR to STAEformer format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get columns to use
        input_cols = window_config.input_columns or ir.feature_columns
        target_cols = window_config.target_columns or input_cols

        # Convert to tensor (T, N, C)
        data = ir.to_tensor(columns=input_cols)
        T, N, C = data.shape

        # Build index arrays for windowing
        L = window_config.input_length
        H = window_config.horizon
        y_start = window_config.y_start

        # Calculate number of valid windows
        num_samples = T - L - y_start - H + 2

        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for windowing. T={T}, L={L}, H={H}, y_start={y_start}"
            )

        # Create index arrays: each row is [window_start, split_point, window_end]
        # For window i: X covers [start:split), Y covers [split:end)
        indices = []
        for i in range(num_samples):
            start = i
            split = i + L  # End of input, start of gap
            end = i + L + y_start - 1 + H  # End of target
            indices.append([start, split, end])

        indices = np.array(indices, dtype=np.int64)

        # Split indices by time
        train_end = int(num_samples * split_config.train_ratio)
        val_end = int(num_samples * (split_config.train_ratio + split_config.val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Save data
        data_path = output_dir / "data.npz"
        np.savez_compressed(
            data_path,
            data=data.astype(np.float32),
            timestamps=ir.timestamps.astype(str),
            node_ids=np.array(ir.nodes),
            feature_names=np.array(input_cols),
        )

        # Save indices
        index_path = output_dir / "index.npz"
        np.savez_compressed(
            index_path,
            train=train_indices,
            val=val_indices,
            test=test_indices,
        )

        # Save adjacency if available
        files = {"data": data_path, "index": index_path}

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
                "shape": data.shape,
                "num_train": len(train_indices),
                "num_val": len(val_indices),
                "num_test": len(test_indices),
            },
        )
