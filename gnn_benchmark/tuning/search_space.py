"""Search-space primitives for the hyperparameter tuner.

Each :class:`Sampler` represents one hyperparameter dimension.  The same
object supports both grid search (via :meth:`grid_values`) and random
search (via :meth:`sample`), and is shaped to map 1-to-1 onto
Optuna's ``trial.suggest_*`` API so a future ``OptunaTuner`` can reuse
the same search-space objects without call-site changes.

Usage::

    from gnn_benchmark.tuning import Categorical, LogUniform, IntUniform

    search_space = {
        "lr":      LogUniform(1e-4, 1e-2),
        "dropout": Categorical([0.1, 0.3, 0.5]),
        "nhid":    IntUniform(16, 64, step=16),
    }
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Sampler(Protocol):
    """One hyperparameter dimension.

    Implementations must provide :meth:`sample` (random search) and may
    provide :meth:`grid_values` (grid search) — continuous samplers such as
    :class:`Uniform` raise when asked for grid values.
    """

    def sample(self, rng: random.Random) -> Any: ...

    def grid_values(self) -> list[Any]: ...


@dataclass(frozen=True)
class Categorical:
    """Discrete choice over a fixed list of values.

    Supports both grid and random search.
    """

    choices: Sequence[Any]

    def __post_init__(self) -> None:
        if len(self.choices) == 0:
            raise ValueError("Categorical needs at least one choice.")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.choices))

    def grid_values(self) -> list[Any]:
        return list(self.choices)


@dataclass(frozen=True)
class IntUniform:
    """Integer range ``[low, high]`` inclusive with optional step.

    Supports both grid and random search.
    """

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.high < self.low:
            raise ValueError("high must be >= low")

    def sample(self, rng: random.Random) -> int:
        values = self.grid_values()
        return rng.choice(values)

    def grid_values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))


@dataclass(frozen=True)
class Uniform:
    """Continuous uniform range ``[low, high]``.

    Random search only — :meth:`grid_values` raises.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("high must be >= low")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid_values(self) -> list[float]:
        raise TypeError(
            "Uniform is continuous and has no grid values; use IntUniform or "
            "Categorical for grid search, or switch strategy='random'."
        )


@dataclass(frozen=True)
class LogUniform:
    """Log-uniform range ``[low, high]`` (both strictly positive).

    Random search only.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniform requires strictly positive bounds")
        if self.high < self.low:
            raise ValueError("high must be >= low")

    def sample(self, rng: random.Random) -> float:
        log_lo = math.log(self.low)
        log_hi = math.log(self.high)
        return math.exp(rng.uniform(log_lo, log_hi))

    def grid_values(self) -> list[float]:
        raise TypeError(
            "LogUniform is continuous and has no grid values; use "
            "Categorical for log-scale grid search."
        )
