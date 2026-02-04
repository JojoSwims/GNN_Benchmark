"""Base ModelExporter class and config dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation
    from gnn_benchmark.core.workspace import DataWorkspace


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

    This class follows a similar pattern to DatasetLoader: the primary entry
    point is the `export()` method which takes a workspace and delegates to
    `workspace.export()`.

    Subclasses must implement:
        - name: Property returning the exporter/model name (used for directories).
        - export_to_directory: Method to write files to a specific directory.

    Example:
        >>> exporter = STAEformerExporter()
        >>> result = exporter.export(
        ...     workspace,
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
        workspace: "DataWorkspace",
        ir: "IntermediateRepresentation",
        window_config: WindowConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> ExportResult:
        """
        Export IR using a workspace.

        This is the primary entry point for exporting, following the same pattern
        as DatasetLoader.prepare(). It delegates to workspace.export().

        Output goes to: workspace/exports/{dataset_name}/{exporter_name}/

        Args:
            workspace: DataWorkspace to use for export.
            ir: IntermediateRepresentation to export.
            window_config: Window configuration. Uses defaults if None.
            split_config: Split configuration. Uses defaults if None.

        Returns:
            ExportResult with paths to created files.
        """
        return workspace.export(
            self,
            ir,
            window_config=window_config,
            split_config=split_config,
        )

    def export_from_workspace(
        self,
        ir: "IntermediateRepresentation",
        window_config: WindowConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> ExportResult:
        """
        Export using workspace linked to the IR.

        This is a convenience method when the IR is already linked to a workspace.
        For new code, prefer using `export(workspace, ir, ...)` directly.

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
            raise ValueError("IR must be linked to workspace for this method")

        return self.export(
            ir.workspace,
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
        T = data.shape[0]

        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        # Calculate number of valid samples
        # For each sample i, we need:
        # - Input: data[i:i+L]
        # - Target: data[i+L+y_start-1:i+L+y_start-1+H]
        # Last valid i: i+L+y_start-1+H-1 < T => i < T - L - y_start - H + 2
        num_samples = T - input_length - y_start - horizon + 2

        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for windowing. T={T}, L={input_length}, "
                f"H={horizon}, y_start={y_start}"
            )

        x_list = []
        y_list = []

        for i in range(num_samples):
            x_start = i
            x_end = i + input_length
            y_start_idx = x_end + y_start - 1
            y_end_idx = y_start_idx + horizon

            x_list.append(data[x_start:x_end])
            y_list.append(data[y_start_idx:y_end_idx])

        x = np.stack(x_list, axis=0)  # (S, L, N, C)
        y = np.stack(y_list, axis=0)  # (S, H, N, C)

        return x, y

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
        n = data.shape[0]
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return data[:train_end], data[train_end:val_end], data[val_end:]

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
