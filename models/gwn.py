"""GWN (Graph WaveNet) benchmark wrapper.

Wraps the standalone Graph WaveNet model so it satisfies the
``BenchmarkModel`` contract.

Usage::

    from gnn_benchmark.models import GWNModel, GWNConfig

    cfg = GWNConfig(max_epochs=100, batch_size=64)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(GWNModel(), config=cfg)
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
from gnn_benchmark.models.GWN.model import gwnet
from gnn_benchmark.models.GWN.util import (
    asym_adj,
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    sym_adj,
)
from gnn_benchmark.utils.losses import masked_huber_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GWNConfig:
    """Training and architecture configuration for :class:`GWNModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = GWNConfig(lr=0.0005, max_epochs=200)
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    max_epochs: int = 100
    early_stop: int = 30
    clip_grad: float = 5.0

    # Architecture (paper defaults)
    nhid: int = 32
    dropout: float = 0.3
    blocks: int = 4
    layers: int = 2
    kernel_size: int = 2

    # Graph convolution
    gcn_bool: bool = True
    addaptadj: bool = True
    adjtype: str = "doubletransition"

    # LR scheduler (multi-step decay — unified across all benchmark models)
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [20, 40, 60, 80])
    lr_decay_ratio: float = 0.5

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


def _resolve_config(config: Any) -> GWNConfig:
    if config is None:
        return GWNConfig()
    if isinstance(config, GWNConfig):
        return config
    if isinstance(config, dict):
        return GWNConfig(**config)
    raise TypeError(
        f"config must be GWNConfig, dict, or None — got {type(config)}"
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


def _build_supports(
    adj: np.ndarray | None,
    adjtype: str,
    device: torch.device,
) -> tuple[list[torch.Tensor] | None, torch.Tensor | None]:
    """Process raw adjacency into GWN support tensors.

    Returns ``(supports, aptinit)`` where *supports* is a list of
    normalised adjacency tensors (or ``None`` when no graph is available)
    and *aptinit* is the first support used to initialise the adaptive
    adjacency via SVD.
    """
    if adj is None:
        return None, None

    adj_mx = np.array(adj, dtype=np.float32)

    if adjtype == "scalap":
        supports_np = [np.array(calculate_scaled_laplacian(adj_mx))]
    elif adjtype == "normlap":
        supports_np = [np.array(
            calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()
        )]
    elif adjtype == "symnadj":
        supports_np = [np.array(sym_adj(adj_mx))]
    elif adjtype == "transition":
        supports_np = [np.array(asym_adj(adj_mx))]
    elif adjtype == "doubletransition":
        supports_np = [
            np.array(asym_adj(adj_mx)),
            np.array(asym_adj(np.transpose(adj_mx))),
        ]
    elif adjtype == "identity":
        supports_np = [np.eye(adj_mx.shape[0], dtype=np.float32)]
    else:
        raise ValueError(f"Unsupported adjtype '{adjtype}'")

    supports = [
        torch.tensor(s, dtype=torch.float32).to(device) for s in supports_np
    ]
    aptinit = supports[0]
    return supports, aptinit


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GWNModel(BenchmarkModel):
    """Benchmark-ready Graph WaveNet wrapper.

    The underlying ``gwnet`` ``nn.Module`` is instantiated at ``fit()``
    time once tensor shapes are known.  When an adjacency matrix is
    provided it is normalised (default: double-transition) and used as
    graph support; the model also learns an adaptive adjacency.  When
    no adjacency is given, the model relies on adaptive adjacency alone.
    """

    def __init__(self) -> None:
        self._model: gwnet | None = None
        self._scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: GWNConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "GWN"

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
        flat = x_train.reshape(-1, input_dim)  # (S*T*N, D)
        mean = torch.nanmean(flat, dim=0)      # (D,)
        diff_sq = (flat - mean) ** 2
        count = (~torch.isnan(flat)).sum(dim=0).float()
        std = torch.sqrt(torch.nansum(diff_sq, dim=0) / count)
        std = torch.clamp(std, min=1e-8)
        scaler = _FeatureScaler(mean, std).to(device)
        self._scaler = scaler

        # -- Normalize x and handle NaN ------------------------------------
        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))

        # -- Targets keep NaN so masked_huber_loss can ignore them ---------
        y_train_d = y_train.to(device)
        y_val_d = y_val.to(device)

        # -- Build graph supports ------------------------------------------
        supports, aptinit = _build_supports(adj, cfg.adjtype, device)

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
        model = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=cfg.dropout,
            supports=supports,
            gcn_bool=cfg.gcn_bool,
            addaptadj=cfg.addaptadj,
            aptinit=aptinit,
            in_dim=input_dim,
            out_dim=out_steps * output_dim,
            residual_channels=cfg.nhid,
            dilation_channels=cfg.nhid,
            skip_channels=cfg.nhid * 8,
            end_channels=cfg.nhid * 16,
            kernel_size=cfg.kernel_size,
            blocks=cfg.blocks,
            layers=cfg.layers,
        ).to(device)

        # -- Training setup ------------------------------------------------
        criterion = masked_huber_loss
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler = None
        if cfg.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg.lr_milestones,
                gamma=cfg.lr_decay_ratio,
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
                # x_batch: (B, T, N, D) -> (B, D, N, T) for gwnet
                x_in = x_batch.permute(0, 3, 2, 1)
                out = model(x_in)  # (B, out_steps*output_dim, N, T_out)
                # T_out is 1 only when in_len <= receptive_field; otherwise the
                # dilated convs leave a trailing window. Collapse by taking the
                # last (most recent) step so the reshape preserves the batch dim.
                out = (
                    out[..., -1]
                    .reshape(-1, out_steps, output_dim, num_nodes)
                    .permute(0, 1, 3, 2)
                )  # (B, out_steps, N, output_dim)
                out = scaler.inverse_transform(out)
                loss = criterion(out, y_batch)

                optimizer.zero_grad()
                loss.backward()
                if cfg.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                batch_losses.append(loss.item())

            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(np.mean(batch_losses)))

            # ---- validate ----
            model.eval()
            batch_losses_val: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_in = x_batch.permute(0, 3, 2, 1)
                    out = model(x_in)
                    out = (
                        out[..., -1]
                        .reshape(-1, out_steps, output_dim, num_nodes)
                        .permute(0, 1, 3, 2)
                    )
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
        out_steps = self._out_steps
        output_dim = self._output_dim

        x_test_n = torch.nan_to_num(scaler.transform(x_test.to(device)))
        num_nodes = x_test.shape[2]

        loader = DataLoader(
            TensorDataset(x_test_n),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_in = x_batch.permute(0, 3, 2, 1)
                out = model(x_in)
                out = (
                    out[..., -1]
                    .reshape(-1, out_steps, output_dim, num_nodes)
                    .permute(0, 1, 3, 2)
                )
                out = scaler.inverse_transform(out)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds, axis=0)

    def get_config(self) -> dict:
        if self._cfg is None:
            return {}
        return asdict(self._cfg)
