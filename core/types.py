"""Core dataclasses and type definitions for GNN Benchmark."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetInfo:
    """Static information about a dataset."""

    name: str
    url: str
    frequency: str  # e.g., "1H", "5T", "15T"
    node_order: list[str]
    feature_columns: list[str]
    units: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class IRMetadata:
    """Metadata about an intermediate representation."""

    name: str
    frequency: str
    node_order: list[str]
    feature_columns: list[str]
    units: dict[str, str] = field(default_factory=dict)
    source_url: str = ""
    transform_history: list[str] = field(default_factory=list)
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
            "transform_history": self.transform_history,
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
            transform_history=data.get("transform_history", []),
            extra=data.get("extra", {}),
        )
