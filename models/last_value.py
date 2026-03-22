"""Last-value (persistence) toy model for end-to-end pipeline testing."""

from typing import Any

import numpy as np

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory


class LastValueModel(BenchmarkModel):
    """
    Last-value persistence model for smoke-testing the benchmark pipeline.

    For every test window it forecasts the final observed value of the input
    window for all horizon steps:

        y_hat(t+1) = y_hat(t+2) = … = y_hat(t+H) = x(t)

    The model requires no training.  Its deterministic predictions make it
    easy to verify metric correctness in end-to-end tests.
    """

    def __init__(self) -> None:
        self._horizon: int | None = None

    @property
    def name(self) -> str:
        return "LastValue"

    def fit(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any,
        y_val: Any,
        adj: np.ndarray | None,
        config: Any,
        norm_stats: dict | None = None,
    ) -> TrainingHistory | None:
        """No-op: store horizon length inferred from training targets."""
        self._horizon = y_train.shape[1]
        return None

    def predict(
        self,
        x_test: Any,
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        """
        Repeat the last input timestep for each forecast horizon step.

        Args:
            x_test: ``[S, L, N, D]`` input windows (torch.Tensor or ndarray).
            adj:    Ignored.
            config: Ignored.

        Returns:
            ``np.ndarray`` of shape ``[S, H, N, D]`` where every horizon step
            equals the final value of the corresponding input window.
        """
        if self._horizon is None:
            raise RuntimeError("Call fit() before predict().")

        # Extract last timestep: [S, N, D]
        last = x_test[:, -1, :, :]
        if hasattr(last, "numpy"):
            last = last.numpy()
        else:
            last = np.asarray(last)

        # Tile along a new horizon axis: [S, 1, N, D] -> [S, H, N, D]
        return np.repeat(last[:, np.newaxis, :, :], self._horizon, axis=1)
