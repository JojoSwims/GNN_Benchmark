"""Base ModelExporter class and config dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class WindowConfig:
    """Configuration for sliding window creation."""

    input_length: int = 12  # L: number of past timesteps
    horizon: int = 12  # H: number of future timesteps to predict
    y_start: int = 1  # Gap between input end and target start
    input_columns: list[str] | None = None  # None = all feature columns
    target_columns: list[str] | None = None  # None = same as input_columns


@dataclass
class SplitConfig:
    """Configuration for train/val/test split."""

    train_ratio: float = 0.7
    val_ratio: float = 0.1
    # test_ratio is implicit: 1 - train_ratio - val_ratio

    def __post_init__(self):
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")
        if self.train_ratio <= 0 or self.val_ratio < 0:
            raise ValueError("Ratios must be positive")

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


@dataclass
class ExportResult:
    """Result of an export operation."""

    files: dict[str, Path]  # name -> path mapping
    window_config: WindowConfig
    split_config: SplitConfig
    stats: dict[str, Any] | None = None  # Optional: normalization stats, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelExporter(ABC):
    """
    Base class for model-specific exporters.

    Handles windowing, splitting, and format conversion from IR to
    model-specific formats.

    Subclasses must implement:
        - name: Property returning the exporter/model name
        - export: Method to export IR to model-specific format
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Exporter/model name (used for directory naming)."""

    @abstractmethod
    def export(
        self,
        ir: "IntermediateRepresentation",
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        """
        Export IR to model-specific format.

        Args:
            ir: The intermediate representation
            output_dir: Where to write output files
            window_config: Sliding window parameters
            split_config: Train/val/test split ratios

        Returns:
            ExportResult with paths to created files
        """

    def export_from_workspace(
        self,
        ir: "IntermediateRepresentation",
        window_config: WindowConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> ExportResult:
        """
        Export using workspace's export directory structure.

        Output goes to: workspace/exports/{dataset_name}/{exporter_name}/

        Args:
            ir: IntermediateRepresentation linked to a workspace
            window_config: Window configuration (uses defaults if None)
            split_config: Split configuration (uses defaults if None)

        Returns:
            ExportResult with paths to created files

        Raises:
            ValueError: If IR is not linked to a workspace
        """
        if ir.workspace is None or ir.dataset_name is None:
            raise ValueError("IR must be linked to workspace for this method")

        output_dir = ir.workspace.exports_dir / ir.dataset_name / self.name
        return self.export(
            ir,
            output_dir,
            window_config or WindowConfig(),
            split_config or SplitConfig(),
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
