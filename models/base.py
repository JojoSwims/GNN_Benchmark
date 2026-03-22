"""Abstract base class for GNN model submissions to the benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    Tensor = torch.Tensor
except ImportError:  # pragma: no cover
    Tensor = Any  # type: ignore[misc,assignment]


@dataclass
class TrainingHistory:
    """
    Training history returned by BenchmarkModel.fit().

    Attributes:
        train_loss: Loss value recorded after each training epoch.
        val_loss: Loss value recorded after each validation epoch.
    """

    train_loss: list[float]
    val_loss: list[float]


class BenchmarkModel(ABC):
    """
    Contract for model submissions to the GNN benchmark.

    A submission must implement ``fit`` and ``predict``.  The harness
    (``BenchmarkRunner``) calls them in sequence for every dataset and
    collects the returned predictions for metric computation.

    Tensor contract
    ---------------
    The harness passes **raw, unnormalised** tensors.  Missing values are
    represented as ``float('nan')`` — do **not** assume zero means missing.

    All tensors are ``torch.Tensor`` of dtype ``float32``:

    .. code-block::

        x_train / x_val / x_test : [S, seq_in_len,  N, D_in]   (may contain NaN)
        y_train / y_val           : [S, seq_out_len, N, D_out]  (may contain NaN)

    ``S`` is the number of sliding-window samples for that split.

    Batching
    --------
    Tensors are passed **unbatched** (no DataLoader is created by the harness).
    The model is free to create a DataLoader with whatever batch size and
    sampling strategy it prefers, e.g.::

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    Normalisation
    -------------
    Normalisation is entirely the model's responsibility.  The harness
    provides ``norm_stats`` (computed from the training split, NaN-excluded)
    so the model can normalise *if* it chooses to::

        x_norm = (x_train - norm_stats["means"]) / norm_stats["stds"]
        # norm_stats["means"] and ["stds"] are shape (D_in,) numpy arrays
        # that broadcast correctly against (S, L, N, D_in) tensors.

    ``y_pred`` returned by ``predict`` **must be in the original
    (unnormalised) space** — the harness computes metrics directly against
    the raw ground-truth values.

    Adjacency matrix
    ----------------
    ``adj`` is a float32 ``(N, N)`` numpy array whose row/column order
    matches the node dimension ``N`` in ``x`` and ``y``.  It is ``None``
    for datasets that carry no edge information.  A model that requires
    a graph should raise ``ValueError("adj is required")`` when called
    with ``adj=None``.

    Config
    ------
    ``config`` is fully submitter-controlled — pass whatever object your
    model needs (a dataclass, a dict, ``None``, …).  Implement
    ``get_config()`` to expose it for result logging.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in result tables."""

    @abstractmethod
    def fit(
        self,
        x_train: "Tensor",
        y_train: "Tensor",
        x_val: "Tensor",
        y_val: "Tensor",
        adj: np.ndarray | None,
        config: Any,
        norm_stats: dict | None = None,
    ) -> "TrainingHistory | None":
        """
        Train the model.

        Args:
            x_train:    Training inputs  ``[S_train, seq_in_len, N, D_in]``,
                        unnormalised, NaN = missing.
            y_train:    Training targets ``[S_train, seq_out_len, N, D_out]``,
                        unnormalised, NaN = missing.
            x_val:      Validation inputs  ``[S_val, seq_in_len, N, D_in]``.
            y_val:      Validation targets ``[S_val, seq_out_len, N, D_out]``.
            adj:        Adjacency matrix [N, N] or None.
            config:     Submitter-defined configuration object.
            norm_stats: Normalisation statistics computed from the training
                        split (NaN excluded).  Dict with keys:

                        * ``"columns"`` — list of feature column names, len D_in
                        * ``"means"``   — ``np.ndarray`` shape ``(D_in,)``
                        * ``"stds"``    — ``np.ndarray`` shape ``(D_in,)``
                        * ``"mins"``    — ``np.ndarray`` shape ``(D_in,)``
                        * ``"maxs"``    — ``np.ndarray`` shape ``(D_in,)``

                        ``None`` if the dataset has no feature columns.

        Returns:
            TrainingHistory if the model tracks per-epoch losses, else None.
        """

    @abstractmethod
    def predict(
        self,
        x_test: "Tensor",
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        """
        Generate predictions for all test windows.

        Args:
            x_test: Test inputs ``[S_test, seq_in_len, N, D_in]``,
                    unnormalised, NaN = missing.
            adj:    Adjacency matrix [N, N] or None.
            config: Submitter-defined configuration object.

        Returns:
            y_pred : np.ndarray of shape ``[S_test, seq_out_len, N, D_out]``
                     in the **original (unnormalised) space**.  The harness
                     computes metrics directly against the raw ground truth.
        """

    def get_config(self) -> dict:
        """
        Return a serialisable representation of the model's configuration.

        Override this to expose hyperparameters in benchmark result logs.
        The default implementation returns an empty dict.
        """
        return {}
