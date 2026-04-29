"""MLP Multivariate baseline model.

Implements a flat MLP that jointly processes all nodes and features from a
sliding window to forecast all horizon steps — inspired by NixtlaVerse's
MLPMultivariate architecture (nixtlaverse.nixtla.io/neuralforecast/models.mlpmultivariate.html).

The input window ``[S, L, N, D_in]`` is flattened to ``[S, L*N*D_in]``
before being fed to the MLP.  Missing values (NaN) are imputed with the
per-(node, feature) mean computed over training windows; masked_huber_loss
ensures NaN targets never bias the gradient.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.utils.losses import masked_huber_loss
from gnn_benchmark.utils.timing import Stopwatch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MLPMultivariateConfig:
    """Training and architecture configuration for :class:`MLPMultivariateModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = MLPMultivariateConfig(hidden_size=256, num_layers=3)
    """

    # Architecture
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop: int = 20
    clip_grad: float | None = 5.0

    # LR scheduler (unified with other benchmark models)
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [20, 40, 60, 80])
    lr_decay_ratio: float = 0.5

    # Runtime
    seed: int | None = None
    device: str = "auto"  # "auto" | "cuda" | "cpu"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config(config: Any) -> MLPMultivariateConfig:
    if config is None:
        return MLPMultivariateConfig()
    if isinstance(config, MLPMultivariateConfig):
        return config
    if isinstance(config, dict):
        return MLPMultivariateConfig(**config)
    raise TypeError(
        f"config must be MLPMultivariateConfig, dict, or None — got {type(config)}"
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _seed_everything(seed: int) -> None:
    import random, os  # noqa: E401
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _NaNImputer:
    """Per-(node, feature) mean imputation computed over training windows."""

    def __init__(self, fill_values: torch.Tensor) -> None:
        # fill_values: [N, D_in]
        self.fill_values = fill_values

    def to(self, device: torch.device) -> "_NaNImputer":
        self.fill_values = self.fill_values.to(device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN in x ``[S, L, N, D]`` with per-(N, D) fill values."""
        mask = torch.isnan(x)
        if not mask.any():
            return x
        fills = self.fill_values.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        return torch.where(mask, fills.expand_as(x), x)


def _fit_imputer(x_train: torch.Tensor) -> _NaNImputer:
    """Compute per-(node, feature) mean over training windows ``[S, L, N, D]``."""
    S, L, N, D = x_train.shape
    flat = x_train.reshape(S * L, N, D)          # [S*L, N, D]
    fill = torch.nanmean(flat, dim=0)             # [N, D]
    fill = torch.nan_to_num(fill, nan=0.0)        # guard fully-missing (N,D) pairs
    return _NaNImputer(fill)


class _FeatureScaler:
    """Per-feature z-score scaler that lives on a torch device."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean  # [D]
        self.std = std    # [D]

    def to(self, device: torch.device) -> "_FeatureScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` of shape ``(..., D)``."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """De-normalize ``x`` of shape ``(..., D)``."""
        d = x.shape[-1]
        return x * self.std[:d] + self.mean[:d]


def _fit_scaler(
    tensor: torch.Tensor, feature_dim: int, device: torch.device
) -> _FeatureScaler:
    """NaN-aware per-feature z-score scaler fit."""
    flat = tensor.reshape(-1, feature_dim)
    mean = torch.nanmean(flat, dim=0)
    diff_sq = (flat - mean) ** 2
    count = (~torch.isnan(flat)).sum(dim=0).float()
    std = torch.sqrt(torch.nansum(diff_sq, dim=0) / count.clamp(min=1))
    std = torch.clamp(std, min=1e-8)
    return _FeatureScaler(mean, std).to(device)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _MLPMultivariateNet(nn.Module):
    """Flat MLP: ``in_dim → [hidden_size] × num_layers → out_dim``.

    Each hidden layer is Linear → ReLU → Dropout.  The output layer is a
    plain Linear (no activation) so the model can freely span the real line.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_size
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPMultivariateModel(BenchmarkModel):
    """MLP baseline that jointly forecasts all nodes from a flattened window.

    Architecture mirrors NixtlaVerse's MLPMultivariate:
      - Impute NaN with per-(node, feature) training-mean fill values.
      - z-score normalize input and target features.
      - Flatten window ``[S, L, N, D_in] → [S, L*N*D_in]``.
      - Stack of Linear → ReLU → Dropout blocks.
      - Linear output head → reshape to ``[S, H, N, D_out]``.

    NaN handling:
      - Imputation: per-(node, feature) training-mean fill before MLP.
      - Loss: ``masked_huber_loss`` ignores NaN positions in training targets.
      - Metrics: the harness masks NaN ground-truth positions (outside this model).
    """

    def __init__(self) -> None:
        self._model: _MLPMultivariateNet | None = None
        self._imputer: _NaNImputer | None = None
        self._x_scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: MLPMultivariateConfig | None = None
        self._out_steps: int | None = None
        self._num_nodes: int | None = None
        self._output_dim: int | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    @property
    def name(self) -> str:
        return "MLPMultivariate"

    # ------------------------------------------------------------------
    # BenchmarkModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
    ) -> TrainingHistory:
        cfg = _resolve_config(config)
        self._cfg = cfg

        if cfg.seed is not None:
            _seed_everything(cfg.seed)

        device = _resolve_device(cfg.device)
        self._device = device

        # Shapes
        _S, in_steps, num_nodes, input_dim = x_train.shape
        out_steps = y_train.shape[1]
        output_dim = y_train.shape[3]
        self._out_steps = out_steps
        self._num_nodes = num_nodes
        self._output_dim = output_dim

        # Imputer — fit on training inputs only, keep NaN out of MLP
        imputer = _fit_imputer(x_train)
        imputer.to(device)
        self._imputer = imputer

        # Scalers (NaN-aware z-score, separate for x and y)
        x_scaler = _fit_scaler(x_train, input_dim, device)
        y_scaler = _fit_scaler(y_train, output_dim, device)
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler

        def _prep_x(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            x = imputer.transform(x)       # fill NaN → [S, L, N, D_in]
            x = x_scaler.transform(x)      # z-score  → [S, L, N, D_in]
            return x.reshape(x.shape[0], -1)  # flatten → [S, L*N*D_in]

        x_tr = _prep_x(x_train)
        x_vl = _prep_x(x_val)
        y_tr = y_train.to(device)   # keep NaN so masked_huber_loss can mask them
        y_vl = y_val.to(device)

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_vl, y_vl),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        in_dim = in_steps * num_nodes * input_dim
        out_dim = out_steps * num_nodes * output_dim
        model = _MLPMultivariateNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = None
        if cfg.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg.lr_milestones,
                gamma=cfg.lr_decay_ratio,
            )

        train_sw = Stopwatch()
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(cfg.max_epochs):
            # ---- train ----
            model.train()
            batch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                with train_sw:
                    out = model(x_batch)                                          # [B, H*N*D_out]
                    out = out.reshape(-1, out_steps, num_nodes, output_dim)       # [B, H, N, D_out]
                    out = y_scaler.inverse_transform(out)                         # original space
                    loss = masked_huber_loss(out, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.clip_grad is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    batch_losses.append(loss.item())

            with train_sw:
                if scheduler is not None:
                    scheduler.step()
            train_losses.append(float(np.mean(batch_losses)))

            # ---- validate ----
            model.eval()
            batch_losses_val: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    with train_sw:
                        out = model(x_batch)
                        out = out.reshape(-1, out_steps, num_nodes, output_dim)
                        out = y_scaler.inverse_transform(out)
                        loss = masked_huber_loss(out, y_batch)
                        batch_losses_val.append(loss.item())

            val_loss = float(np.mean(batch_losses_val))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= cfg.early_stop:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._train_compute_sec = train_sw.elapsed

        return TrainingHistory(train_loss=train_losses, val_loss=val_losses)

    def predict(
        self,
        x_test: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        if self._model is None or self._imputer is None or self._x_scaler is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        model = self._model
        out_steps = self._out_steps
        num_nodes = self._num_nodes
        output_dim = self._output_dim

        def _prep_x(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            x = self._imputer.transform(x)
            x = self._x_scaler.transform(x)
            return x.reshape(x.shape[0], -1)

        x_te = _prep_x(x_test)
        loader = DataLoader(
            TensorDataset(x_te),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        infer_sw = Stopwatch()
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                with infer_sw:
                    out = model(x_batch)
                    out = out.reshape(-1, out_steps, num_nodes, output_dim)
                    out = self._y_scaler.inverse_transform(out)
                    preds.append(out.cpu().numpy())

        self._infer_compute_sec = infer_sw.elapsed
        return np.concatenate(preds, axis=0)

    def get_config(self) -> dict:
        if self._cfg is None:
            return {}
        return asdict(self._cfg)

    def get_train_compute_sec(self) -> float | None:
        return self._train_compute_sec or None

    def get_inference_compute_sec(self) -> float | None:
        return self._infer_compute_sec or None
