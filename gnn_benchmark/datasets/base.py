"""Base DatasetLoader class for loading external datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation
    from gnn_benchmark.core.workspace import DataWorkspace


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Users can subclass this to add their own datasets.
    Each loader handles downloading external data and converting it
    to the intermediate representation format.

    Subclasses must implement:
        - info: Property returning DatasetInfo
        - download_and_convert: Method to download raw data and convert to IR format
    """

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return static dataset information."""

    @abstractmethod
    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Download raw data and convert to IR format.

        Returns:
            (series_df, edges_df) tuple where:
            - series_df: DataFrame with columns [ts, node_id, feature1, feature2, ...]
            - edges_df: DataFrame with columns [src, dst, cost], or None if no graph
        """

    def prepare(
        self, workspace: "DataWorkspace", force_download: bool = False
    ) -> "IntermediateRepresentation":
        """
        Prepare dataset using a workspace.

        This is a convenience method that delegates to workspace.prepare().

        Args:
            workspace: DataWorkspace to use
            force_download: If True, re-download even if cached

        Returns:
            IntermediateRepresentation linked to the workspace
        """
        return workspace.prepare(self, force_download=force_download)
