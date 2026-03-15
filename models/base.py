"""Abstract base class for GNN model submissions to the benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    DataLoader = Any  # type: ignore[misc,assignment]


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

    DataLoader contract
    -------------------
    Both loaders yield ``(x_batch, y_batch)`` 2-tuples of float32 tensors:

    .. code-block::

        x_batch : [B, seq_in_len,  N, D_in]
        y_batch : [B, seq_out_len, N, D_out]

    The ``test_loader`` passed to ``predict`` is always created with
    ``shuffle=False`` by ``make_dataloaders``, which guarantees that
    ``y_pred`` rows correspond one-to-one with the ground-truth ``y_test``
    rows in the order they were windowed.

    Adjacency matrix
    ----------------
    ``adj`` is a float32 ``(N, N)`` numpy array whose row/column order
    matches the node dimension ``N`` in ``x`` and ``y``.  It is ``None``
    for datasets that carry no edge information.  A model that requires
    a graph should raise ``ValueError("adj is required")`` when called
    with ``adj=None``.

    Normalisation contract
    ----------------------
    ``x`` and ``y`` arriving in the DataLoaders are already in the
    normalised space produced by the preprocessing pipeline
    (``ZScoreNormalize`` by default).  ``y_pred`` must be returned in
    the **same normalised space**.  The harness handles denormalisation
    before computing human-readable metrics.

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
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        adj: np.ndarray | None,
        config: Any,
    ) -> "TrainingHistory | None":
        """
        Train the model.

        Args:
            train_loader: DataLoader over training windows (shuffle=True).
            val_loader:   DataLoader over validation windows (shuffle=False).
            adj:          Adjacency matrix [N, N] or None.
            config:       Submitter-defined configuration object.

        Returns:
            TrainingHistory if the model tracks per-epoch losses, else None.
        """

    @abstractmethod
    def predict(
        self,
        test_loader: "DataLoader",
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        """
        Generate predictions for all test windows.

        Args:
            test_loader: DataLoader over test windows (shuffle=False, order preserved).
            adj:         Adjacency matrix [N, N] or None.
            config:      Submitter-defined configuration object.

        Returns:
            y_pred : np.ndarray of shape [num_test_samples, seq_out_len, N, D_out]
                     in the same normalised space as the training targets.
        """

    def get_config(self) -> dict:
        """
        Return a serialisable representation of the model's configuration.

        Override this to expose hyperparameters in benchmark result logs.
        The default implementation returns an empty dict.
        """
        return {}
