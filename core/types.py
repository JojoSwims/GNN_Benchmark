"""Core dataclasses and type definitions for GNN Benchmark."""

from dataclasses import dataclass, field
from typing import Any


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
class DatasetInfo:
    """Static information about a dataset."""

    name: str
    url: str
    frequency: str  # e.g., "1h", "5min", "15min"
    node_order: list[str]
    feature_columns: list[str]
    units: dict[str, str] = field(default_factory=dict)
    description: str = ""
    window_config: WindowConfig = field(default_factory=WindowConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class IRMetadata:
    """Metadata about an intermediate representation."""

    name: str
    frequency: str
    node_order: list[str]
    feature_columns: list[str]
    units: dict[str, str] = field(default_factory=dict)
    source_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "frequency": self.frequency,
            "node_order": self.node_order,
            "feature_columns": self.feature_columns,
            "units": self.units,
            "source_url": self.source_url,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IRMetadata":
        """Create IRMetadata from dictionary."""
        return cls(
            name=data["name"],
            frequency=data["frequency"],
            node_order=data["node_order"],
            feature_columns=data["feature_columns"],
            units=data.get("units", {}),
            source_url=data.get("source_url", ""),
            extra=data.get("extra", {}),
        )
