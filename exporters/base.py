"""Base ModelExporter class and config dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gnn_benchmark.exporters.dataloader import (
    create_sliding_windows as _create_sliding_windows_fn,
    split_by_time as _split_by_time_fn,
)

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class WindowConfig:
    """
    Configuration for sliding window creation.

    Attributes:
        input_length: Number of past timesteps (L) to use as input.
        horizon: Number of future timesteps (H) to predict.
        y_start: Gap between input end and target start. Default is 1,
            meaning target starts immediately after input.
        input_columns: Feature columns to use for input. None means all columns.
        target_columns: Feature columns to predict. None means same as input_columns.
    """

    input_length: int = 12
    horizon: int = 12
    y_start: int = 1
    input_columns: list[str] | None = None
    target_columns: list[str] | None = None


@dataclass
class SplitConfig:
    """
    Configuration for temporal train/val/test split.

    The test ratio is computed as 1 - train_ratio - val_ratio.

    Attributes:
        train_ratio: Fraction of data for training (e.g., 0.7 for 70%).
        val_ratio: Fraction of data for validation (e.g., 0.1 for 10%).
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.1

    def __post_init__(self):
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")
        if self.train_ratio <= 0 or self.val_ratio < 0:
            raise ValueError("Ratios must be positive")

    @property
    def test_ratio(self) -> float:
        """Compute the test ratio as the remainder."""
        return 1.0 - self.train_ratio - self.val_ratio


@dataclass
class ExportResult:
    """
    Result of an export operation.

    Attributes:
        files: Mapping of file names to their paths.
        window_config: The window configuration used.
        split_config: The split configuration used.
        stats: Optional statistics (e.g., normalization parameters).
        metadata: Additional metadata about the export.
    """

    files: dict[str, Path]
    window_config: WindowConfig
    split_config: SplitConfig
    stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelExporter(ABC):
    """
    Abstract base class for model-specific exporters.

    Exporters convert an IntermediateRepresentation to model-specific formats,
    handling windowing, splitting, and file format conversion.

    The primary entry point is the `export()` method which takes an IR linked
    to a workspace and delegates to `workspace.export()`.

    Subclasses must implement:
        - name: Property returning the exporter/model name (used for directories).
        - export_to_directory: Method to write files to a specific directory.

    Example:
        >>> exporter = STAEformerExporter()
        >>> ir = workspace.load("my_dataset")
        >>> result = exporter.export(
        ...     ir,
        ...     window_config=WindowConfig(input_length=12, horizon=12),
        ... )
        >>> print(result.files)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Exporter/model name.

        Used for naming the export subdirectory under workspace/exports/{dataset}/.
        """

    @abstractmethod
    def export_to_directory(
        self,
        ir: "IntermediateRepresentation",
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """
        Export IR to model-specific format in the given directory.

        This is the method subclasses must implement. It performs the actual
        conversion and file writing.

        Args:
            ir: The intermediate representation to export.
            output_dir: Directory where output files should be written.
            window_config: Sliding window parameters.
            split_config: Train/val/test split ratios.

        Returns:
            ExportResult with paths to created files and metadata.
        """

    def export(
        self,
        ir: "IntermediateRepresentation",
        window_config: WindowConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> ExportResult:
        """
        Export IR to model-specific format.

        Output goes to: workspace/exports/{dataset_name}/{exporter_name}/

        Args:
            ir: IntermediateRepresentation linked to a workspace.
            window_config: Window configuration. Uses defaults if None.
            split_config: Split configuration. Uses defaults if None.

        Returns:
            ExportResult with paths to created files.

        Raises:
            ValueError: If IR is not linked to a workspace.
        """
        if ir.workspace is None or ir.dataset_name is None:
            raise ValueError("IR must be linked to a workspace")

        return ir.workspace.export(
            self,
            ir,
            window_config=window_config,
            split_config=split_config,
        )

    # --- Helper methods for subclasses ---

    def _create_sliding_windows(
        self,
        data: np.ndarray,
        input_length: int,
        horizon: int,
        y_start: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data.

        Args:
            data: Array of shape (T, N, C) or (T, N)
            input_length: Number of past timesteps (L)
            horizon: Number of future timesteps (H)
            y_start: Gap between input end and target start

        Returns:
            (x, y) where:
            - x: Array of shape (S, L, N, C) - input windows
            - y: Array of shape (S, H, N, C) - target windows
            S = number of valid samples
        """
        return _create_sliding_windows_fn(data, input_length, horizon, y_start)

    def _split_by_time(
        self,
        data: np.ndarray,
        train_ratio: float,
        val_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally into train/val/test.

        Args:
            data: Array with first dimension as samples (time)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation

        Returns:
            (train, val, test) arrays
        """
        return _split_by_time_fn(data, train_ratio, val_ratio)

    def _compute_offsets(
        self, input_length: int, horizon: int, y_start: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute offset arrays for windowing.

        Args:
            input_length: Number of past timesteps (L)
            horizon: Number of future timesteps (H)
            y_start: Gap between input end and target start

        Returns:
            (x_offsets, y_offsets) relative to window start
        """
        x_offsets = np.arange(-input_length + 1, 1).reshape(-1, 1)
        y_offsets = np.arange(y_start, y_start + horizon).reshape(-1, 1)
        return x_offsets, y_offsets
