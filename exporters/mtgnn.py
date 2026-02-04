"""MTGNN model exporter."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.exporters.base import (
    ExportResult,
    ModelExporter,
    SplitConfig,
    WindowConfig,
)


@dataclass
class MTGNNExporter(ModelExporter):
    """
    Export to MTGNN format.

    Creates:
        - data.h5: Time series data in HDF5 format
        - adj_mx.pkl: Adjacency matrix

    MTGNN uses a wide-format HDF5 file where each column is a sensor.
    """

    @property
    def name(self) -> str:
        return "mtgnn"

    def export_to_directory(
        self,
        ir: IntermediateRepresentation,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """Export IR to MTGNN format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get columns to use (MTGNN typically uses single feature)
        input_cols = window_config.input_columns or ir.feature_columns[:1]
        col = input_cols[0]

        # Convert to wide format DataFrame
        wide_df = ir.series.pivot(index="ts", columns="node_id", values=col)

        # Reorder columns to match node order
        wide_df = wide_df[ir.nodes]

        # Fill NaN with 0 (MTGNN expects complete data)
        wide_df = wide_df.fillna(0.0)

        # Save HDF5
        h5_path = output_dir / "data.h5"
        wide_df.to_hdf(h5_path, key="df", mode="w")

        files = {"data": h5_path}

        # Save adjacency matrix
        if ir.edges is not None:
            adj_path = output_dir / "adj_mx.pkl"
            nodes = ir.nodes
            id_to_ind = {node: i for i, node in enumerate(nodes)}
            adj = ir.get_adjacency_matrix()

            with open(adj_path, "wb") as f:
                pickle.dump((nodes, id_to_ind, adj), f)
            files["adj"] = adj_path

        # Also save windowed NPZ for convenience
        data = ir.to_tensor(columns=input_cols)
        x, y = self._create_sliding_windows(
            data,
            window_config.input_length,
            window_config.horizon,
            window_config.y_start,
        )

        x_train, x_val, x_test = self._split_by_time(
            x, split_config.train_ratio, split_config.val_ratio
        )
        y_train, y_val, y_test = self._split_by_time(
            y, split_config.train_ratio, split_config.val_ratio
        )

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

        return ExportResult(
            files=files,
            window_config=window_config,
            split_config=split_config,
            metadata={
                "data_shape": wide_df.shape,
                "train_samples": len(x_train),
                "val_samples": len(x_val),
                "test_samples": len(x_test),
            },
        )
