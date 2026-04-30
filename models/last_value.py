"""Last-value (persistence) baseline model."""

from typing import Any

import numpy as np

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory


class LastValueModel(BenchmarkModel):
    """
    Last-value persistence baseline.

    For every test window it forecasts the last *observed* (non-NaN) value of
    the input window for all horizon steps:

        y_hat(t+1) = y_hat(t+2) = … = y_hat(t+H) = x(t_last_valid)

    When an entire (node, feature) sequence is NaN the forecast falls back to
    zero.  The model requires no training; its deterministic predictions make
    it easy to verify metric correctness in end-to-end tests.
    """

    def __init__(self) -> None:
        self._horizon: int | None = None
        self._output_dim: int | None = None

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
    ) -> TrainingHistory | None:
        self._horizon = y_train.shape[1]
        self._output_dim = y_train.shape[3]
        return None

    def predict(
        self,
        x_test: Any,
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        """
        Repeat the last observed input timestep for each forecast horizon step.

        Args:
            x_test: ``[S, L, N, D_in]`` input windows (torch.Tensor or ndarray).
            adj:    Ignored.
            config: Ignored.

        Returns:
            ``np.ndarray`` of shape ``[S, H, N, D_out]`` where every horizon
            step equals the last non-NaN value of the corresponding input
            window's leading ``D_out`` (target) features, or 0 where the
            entire (node, feature) series is missing.
        """
        if self._horizon is None or self._output_dim is None:
            raise RuntimeError("Call fit() before predict().")

        if hasattr(x_test, "numpy"):
            x = x_test.numpy()
        else:
            x = np.asarray(x_test, dtype=np.float32)

        # Datasets place target columns as the leading features, so slice the
        # input down to the first D_out channels before persistence.
        x = x[:, :, :, : self._output_dim]

        # Reverse along the time axis so the most recent timestep is first.
        x_rev = x[:, ::-1, :, :]  # [S, L, N, D_out]

        # For each (s, n, d), find the index of the first finite value in the
        # reversed sequence — that is the last observed value in the original.
        finite = np.isfinite(x_rev)  # [S, L, N, D_out]

        # argmax returns 0 when no True exists (all-NaN), which maps to the
        # last original timestep — still NaN, cleaned up by nan_to_num below.
        first_valid = np.argmax(finite, axis=1)  # [S, N, D_out]

        S, _, N, D = x_rev.shape
        s_idx = np.arange(S)[:, None, None]
        n_idx = np.arange(N)[None, :, None]
        d_idx = np.arange(D)[None, None, :]

        last = x_rev[s_idx, first_valid, n_idx, d_idx]  # [S, N, D_out]
        last = np.nan_to_num(last, nan=0.0)              # all-NaN → 0

        # Tile to [S, H, N, D_out]
        return np.repeat(last[:, np.newaxis, :, :], self._horizon, axis=1)
