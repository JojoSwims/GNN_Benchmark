"""Hyperparameter tuning for GNN Benchmark model submissions.

Public entry points:

- :class:`HyperparameterTuner` — grid / random search over a model's
  config dataclass, driven entirely by validation loss.  The test set is
  never touched during tuning.
- :class:`TuningResult`, :class:`TrialResult` — result containers.
- :class:`Categorical`, :class:`IntUniform`, :class:`Uniform`,
  :class:`LogUniform` — search-space primitives, shaped to match Optuna's
  ``trial.suggest_*`` API.
"""

from gnn_benchmark.tuning.search_space import (
    Categorical,
    IntUniform,
    LogUniform,
    Sampler,
    Uniform,
)
from gnn_benchmark.tuning.tuner import (
    HyperparameterTuner,
    TrialResult,
    TuningResult,
)

__all__ = [
    "HyperparameterTuner",
    "TuningResult",
    "TrialResult",
    "Sampler",
    "Categorical",
    "IntUniform",
    "Uniform",
    "LogUniform",
]
