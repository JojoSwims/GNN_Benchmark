"""Hyperparameter tuning for GNN Benchmark model submissions.

Public entry points:

- :class:`HyperparameterTuner` — grid / random search over a model's
  config dataclass, driven entirely by validation loss.  The test set is
  never touched during tuning.
- :class:`TuningResult`, :class:`TrialResult` — result containers.
- :class:`Categorical`, :class:`IntUniform`, :class:`Uniform`,
  :class:`LogUniform` — search-space primitives, shaped to match Optuna's
  ``trial.suggest_*`` API.
- ``gnn_benchmark.tuning.spaces`` — per-model default search-space
  factories (``gwn_search_space``, ``mtgnn_search_space``, …).  Each
  factory returns a fresh dict and accepts ``**overrides`` for
  per-dataset capping.
"""

from gnn_benchmark.tuning.search_space import (
    Categorical,
    IntUniform,
    LogUniform,
    Sampler,
    Uniform,
)
from gnn_benchmark.tuning.spaces import (
    DEFAULT_LR_HIGH,
    DEFAULT_LR_LOW,
    astgcn_search_space,
    dyngwn_search_space,
    gwn_search_space,
    mlp_multivariate_search_space,
    mtgnn_search_space,
    mtgode_search_space,
    staeformer_search_space,
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
    "DEFAULT_LR_LOW",
    "DEFAULT_LR_HIGH",
    "dyngwn_search_space",
    "gwn_search_space",
    "mlp_multivariate_search_space",
    "mtgnn_search_space",
    "astgcn_search_space",
    "staeformer_search_space",
    "mtgode_search_space",
]
