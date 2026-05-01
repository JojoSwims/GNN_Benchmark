"""ASTGCN benchmark wrapper.

Wraps the standalone ASTGCN (Attention-Based Spatial-Temporal Graph
Convolutional Network) model so it satisfies the ``BenchmarkModel``
contract.

Usage::

    from gnn_benchmark.models import ASTGCNModel, ASTGCNConfig

    cfg = ASTGCNConfig(max_epochs=80, batch_size=32)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(ASTGCNModel(), config=cfg)
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
from gnn_benchmark.models.ASTGCN.model.ASTGCN_r import ASTGCN
from gnn_benchmark.utils.losses import masked_huber_loss
from gnn_benchmark.utils.timing import Stopwatch


def _fit_feature_scaler(
    tensor: torch.Tensor, feature_dim: int, device: torch.device
) -> "_FeatureScaler":
    """NaN-aware per-feature z-score scaler fit (used for both x and y)."""
    flat = tensor.reshape(-1, feature_dim)
    mean = torch.nanmean(flat, dim=0)
    diff_sq = (flat - mean) ** 2
    count = (~torch.isnan(flat)).sum(dim=0).float()
    std = torch.sqrt(torch.nansum(diff_sq, dim=0) / count)
    std = torch.clamp(std, min=1e-8)
    return _FeatureScaler(mean, std).to(device)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ASTGCNConfig:
    """Training and architecture configuration for :class:`ASTGCNModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = ASTGCNConfig(lr=0.0005, max_epochs=200)
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0
    eps: float = 1e-8                        # Adam epsilon
    batch_size: int = 32
    max_epochs: int = 80
    early_stop: int = 30
    clip_grad: float | None = 5.0            # gradient-norm clip; None disables

    # Architecture (paper defaults)
    nb_block: int = 2
    K: int = 3
    nb_chev_filter: int = 64
    nb_time_filter: int = 64
    time_strides: int = 1

    # Ablation: replace the Chebyshev polynomials with identity matrices
    # so the spatial graph convolution stops mixing across nodes (each
    # node gets a per-node linear rescaled by its own spatial-attention
    # score). Spatial/temporal attention parameters and Theta weights are
    # preserved. When True the supplied adjacency is ignored.
    no_graph: bool = False

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
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        return x * self.std[:d] + self.mean[:d]


def _resolve_config(config: Any) -> ASTGCNConfig:
    if config is None:
        return ASTGCNConfig()
    if isinstance(config, ASTGCNConfig):
        return config
    if isinstance(config, dict):
        return ASTGCNConfig(**config)
    raise TypeError(
        f"config must be ASTGCNConfig, dict, or None — got {type(config)}"
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

class ASTGCNModel(BenchmarkModel):
    """Benchmark-ready ASTGCN wrapper.

    The underlying ``ASTGCN`` ``nn.Module`` is instantiated at ``fit()``
    time once tensor shapes are known.  ASTGCN performs Chebyshev graph
    convolutions over the scaled Laplacian of the supplied adjacency
    matrix, so a graph is required.
    """

    def __init__(self) -> None:
        self._model: ASTGCN | None = None
        self._scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: ASTGCNConfig | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "ASTGCN"

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
        if adj is None and not cfg.no_graph:
            raise ValueError("adj is required")
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

        if in_steps % cfg.time_strides != 0:
            raise ValueError(
                f"in_steps ({in_steps}) must be divisible by time_strides "
                f"({cfg.time_strides})."
            )

        # -- Fit scalers (per-feature z-score, NaN-aware) ------------------
        scaler = _fit_feature_scaler(x_train, input_dim, device)
        y_scaler = _fit_feature_scaler(y_train, output_dim, device)
        self._scaler = scaler
        self._y_scaler = y_scaler

        # -- Normalize x and handle NaN ------------------------------------
        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))
        # Targets keep NaN so masked_huber_loss can ignore them.
        y_train_d = y_train.to(device)
        y_val_d = y_val.to(device)

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
        adj_mx = None if cfg.no_graph else np.asarray(adj, dtype=np.float32)
        model = ASTGCN(
            adj_mx=adj_mx,
            nb_block=cfg.nb_block,
            in_channels=input_dim,
            K=cfg.K,
            nb_chev_filter=cfg.nb_chev_filter,
            nb_time_filter=cfg.nb_time_filter,
            time_strides=cfg.time_strides,
            num_for_predict=out_steps,
            len_input=in_steps,
            num_of_vertices=num_nodes,
            output_dim=output_dim,
            no_graph=cfg.no_graph,
        ).to(device)

        # -- Training setup ------------------------------------------------
        criterion = masked_huber_loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eps=cfg.eps,
        )
        scheduler = None
        if cfg.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg.lr_milestones,
                gamma=cfg.lr_decay_ratio,
            )

        # -- Training loop -------------------------------------------------
        train_sw = Stopwatch()
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(cfg.max_epochs):
            model.train()
            batch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                with train_sw:
                    # x_batch: (B, T, N, D) -> (B, N, D, T) for ASTGCN
                    x_in = x_batch.permute(0, 2, 3, 1)
                    out = model(x_in)  # (B, N, T_out, D_out)
                    out = out.permute(0, 2, 1, 3)  # (B, T_out, N, D_out)
                    out = y_scaler.inverse_transform(out)
                    loss = criterion(out, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.clip_grad:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    batch_loss = loss.item()
                batch_losses.append(batch_loss)
            with train_sw:
                if scheduler is not None:
                    scheduler.step()
            train_losses.append(float(np.mean(batch_losses)))

            model.eval()
            batch_losses_val: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    with train_sw:
                        x_in = x_batch.permute(0, 2, 3, 1)
                        out = model(x_in).permute(0, 2, 1, 3)
                        out = y_scaler.inverse_transform(out)
                        loss = criterion(out, y_batch)
                        batch_loss = loss.item()
                    batch_losses_val.append(batch_loss)
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
        if self._model is None or self._scaler is None or self._y_scaler is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        scaler = self._scaler
        y_scaler = self._y_scaler
        model = self._model

        x_test_n = torch.nan_to_num(scaler.transform(x_test.to(device)))

        loader = DataLoader(
            TensorDataset(x_test_n),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        infer_sw = Stopwatch()
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                with infer_sw:
                    x_in = x_batch.permute(0, 2, 3, 1)
                    out = model(x_in).permute(0, 2, 1, 3)
                    out = y_scaler.inverse_transform(out)
                    out_cpu = out.cpu().numpy()
                preds.append(out_cpu)

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
