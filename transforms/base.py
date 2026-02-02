"""Base Transform class for intermediate representation manipulation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


class Transform(ABC):
    """
    Base class for IR transformations.

    Transforms modify the IR in-place (series, mask, metadata).
    They do NOT change the fundamental format (ts, node_id, features).

    Subclasses must implement:
        - description: Human-readable description for transform history
        - __call__: Apply the transform in-place
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for transform history."""

    @abstractmethod
    def __call__(self, ir: "IntermediateRepresentation") -> None:
        """
        Apply transform in-place.

        Args:
            ir: IntermediateRepresentation to modify
        """
