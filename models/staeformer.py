"""STAEFormer benchmark wrapper.

Wraps the standalone STAEFormer (Spatial-Temporal Adaptive Embedding
Transformer) model so it satisfies the ``BenchmarkModel`` contract.

Usage::

    from gnn_benchmark.models import STAEFormerModel, STAEFormerConfig

    cfg = STAEFormerConfig(max_epochs=200, batch_size=16)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(STAEFormerModel(), config=cfg)
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
from gnn_benchmark.models.STAEFormer.model.STAEformer import STAEformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class STAEFormerConfig:
    """Training and architecture configuration for :class:`STAEFormerModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = STAEFormerConfig(lr=0.0005, max_epochs=300)
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0003
    milestones: list[int] = field(default_factory=lambda: [20, 30])
    lr_decay_rate: float = 0.1
    batch_size: int = 16
    max_epochs: int = 200
    early_stop: int = 30
    clip_grad: float | None = None

    # Architecture (paper defaults — override only if experimenting)
    input_embedding_dim: int = 24
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Runtime
    seed: int | None = None
    device: str = "auto"  # "auto" | "cuda" | "cpu"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FeatureScaler:
    """Per-feature z-score scaler that lives on a torch device."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean  # (D_in,)
        self.std = std    # (D_in,)

    def to(self, device: torch.device) -> "_FeatureScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` of shape ``(..., D_in)``."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """De-normalize ``x`` of shape ``(..., D_out)``."""
        d = x.shape[-1]
        return x * self.std[:d] + self.mean[:d]


def _resolve_config(config: Any) -> STAEFormerConfig:
    if config is None:
        return STAEFormerConfig()
    if isinstance(config, STAEFormerConfig):
        return config
    if isinstance(config, dict):
        return STAEFormerConfig(**config)
    raise TypeError(
        f"config must be STAEFormerConfig, dict, or None — got {type(config)}"
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class STAEFormerModel(BenchmarkModel):
    """Benchmark-ready STAEFormer wrapper.

    The underlying ``STAEformer`` ``nn.Module`` is instantiated at
    ``fit()`` time once tensor shapes are known.  TOD/DOW embeddings are
    disabled (the benchmark does not provide temporal features); the
    learnable adaptive embedding still captures positional patterns.
    """

    def __init__(self) -> None:
        self._model: STAEformer | None = None
        self._scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: STAEFormerConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "STAEFormer"

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

        # -- Infer shapes --------------------------------------------------
        num_nodes = x_train.shape[2]
        in_steps = x_train.shape[1]
        out_steps = y_train.shape[1]
        input_dim = x_train.shape[3]
        output_dim = y_train.shape[3]
        self._out_steps = out_steps
        self._output_dim = output_dim

        # -- Fit scaler on x_train (per-feature, ignoring NaN) -------------
        # x_train: (S, T, N, D)
        flat = x_train.reshape(-1, input_dim)  # (S*T*N, D)
        mean = torch.nanmean(flat, dim=0)      # (D,)
        # nanstd: manual since torch has no nanstd
        diff_sq = (flat - mean) ** 2
        count = (~torch.isnan(flat)).sum(dim=0).float()
        std = torch.sqrt(torch.nansum(diff_sq, dim=0) / count)
        std = torch.clamp(std, min=1e-8)
        scaler = _FeatureScaler(mean, std).to(device)
        self._scaler = scaler

        # -- Normalize x and handle NaN ------------------------------------
        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))

        # -- Prepare y (replace NaN with 0 for loss) -----------------------
        y_train_d = torch.nan_to_num(y_train.to(device))
        y_val_d = torch.nan_to_num(y_val.to(device))

        # -- DataLoaders ---------------------------------------------------
        train_loader = DataLoader(
            TensorDataset(x_train_n, y_train_d),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val_n, y_val_d),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        # -- Build model ---------------------------------------------------
        model = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=288,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=cfg.input_embedding_dim,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=cfg.adaptive_embedding_dim,
            feed_forward_dim=cfg.feed_forward_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_mixed_proj=True,
        ).to(device)

        # -- Training setup ------------------------------------------------
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.lr_decay_rate,
        )

        # -- Training loop -------------------------------------------------
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(cfg.max_epochs):
            # ---- train one epoch ----
            model.train()
            batch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                out = model(x_batch)
                out = scaler.inverse_transform(out)
                loss = criterion(out, y_batch)

                optimizer.zero_grad()
                loss.backward()
                if cfg.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                batch_losses.append(loss.item())

            scheduler.step()
            train_losses.append(float(np.mean(batch_losses)))

            # ---- validate ----
            model.eval()
            batch_losses_val: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    out = model(x_batch)
                    out = scaler.inverse_transform(out)
                    loss = criterion(out, y_batch)
                    batch_losses_val.append(loss.item())
            val_loss = float(np.mean(batch_losses_val))
            val_losses.append(val_loss)

            # ---- early stopping ----
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

        return TrainingHistory(train_loss=train_losses, val_loss=val_losses)

    def predict(
        self,
        x_test: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        scaler = self._scaler
        model = self._model

        x_test_n = torch.nan_to_num(scaler.transform(x_test.to(device)))

        loader = DataLoader(
            TensorDataset(x_test_n),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                out = model(x_batch)
                out = scaler.inverse_transform(out)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds, axis=0)

    def get_config(self) -> dict:
        if self._cfg is None:
            return {}
        return asdict(self._cfg)
