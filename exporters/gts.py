"""GTS model exporter."""

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
class GTSExporter(ModelExporter):
    """
    Export to GTS (Graph for Time Series) format.

    Creates:
        - data.h5: Time series data in HDF5 format
        - adj_mx.pkl: Adjacency matrix
        - sensor_ids.txt: List of sensor IDs
    """

    @property
    def name(self) -> str:
        return "gts"

    def export_to_directory(
        self,
        ir: IntermediateRepresentation,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """Export IR to GTS format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get columns to use
        input_cols = window_config.input_columns or ir.feature_columns[:1]
        col = input_cols[0]

        # Convert to wide format DataFrame
        wide_df = ir.series.pivot(index="ts", columns="node_id", values=col)
        wide_df = wide_df[ir.nodes]
        wide_df = wide_df.fillna(0.0)

        # Save HDF5
        h5_path = output_dir / "data.h5"
        wide_df.to_hdf(h5_path, key="df", mode="w")

        files = {"data": h5_path}

        # Save sensor IDs
        sensor_path = output_dir / "sensor_ids.txt"
        with open(sensor_path, "w") as f:
            f.write(",".join(ir.nodes))
        files["sensor_ids"] = sensor_path

        # Save adjacency matrix
        if ir.edges is not None:
            adj_path = output_dir / "adj_mx.pkl"
            nodes = ir.nodes
            id_to_ind = {node: i for i, node in enumerate(nodes)}
            adj = ir.get_adjacency_matrix()

            with open(adj_path, "wb") as f:
                pickle.dump((nodes, id_to_ind, adj), f)
            files["adj"] = adj_path

        return ExportResult(
            files=files,
            window_config=window_config,
            split_config=split_config,
            metadata={
                "data_shape": wide_df.shape,
                "num_nodes": len(ir.nodes),
            },
        )
