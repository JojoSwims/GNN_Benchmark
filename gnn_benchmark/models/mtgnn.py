"""MTGNN benchmark wrapper.

Wraps the standalone MTGNN (Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks) model so it satisfies the
``BenchmarkModel`` contract.

Usage::

    from gnn_benchmark.models import MTGNNModel, MTGNNConfig

    cfg = MTGNNConfig(max_epochs=100, batch_size=64)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(MTGNNModel(), config=cfg)
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
from gnn_benchmark.models.MTGNN.net import gtnet
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
class MTGNNConfig:
    """Training and architecture configuration for :class:`MTGNNModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = MTGNNConfig(lr=0.0005, max_epochs=200)
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0001
    eps: float = 1e-8                        # Adam epsilon
    batch_size: int = 64
    max_epochs: int = 100
    early_stop: int = 30
    clip_grad: float | None = 5.0            # gradient-norm clip; None disables

    # Architecture (paper defaults for multi-step forecasting)
    gcn_true: bool = True
    buildA_true: bool = True
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 1
    conv_channels: int = 32
    residual_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    layers: int = 3
    propalpha: float = 0.05
    tanhalpha: float = 3.0
    layer_norm_affline: bool = True

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


def _resolve_config(config: Any) -> MTGNNConfig:
    if config is None:
        return MTGNNConfig()
    if isinstance(config, MTGNNConfig):
        return config
    if isinstance(config, dict):
        return MTGNNConfig(**config)
    raise TypeError(
        f"config must be MTGNNConfig, dict, or None — got {type(config)}"
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

class MTGNNModel(BenchmarkModel):
    """Benchmark-ready MTGNN wrapper.

    The underlying ``gtnet`` ``nn.Module`` is instantiated at ``fit()``
    time once tensor shapes are known.  MTGNN learns an adaptive
    adjacency matrix via its graph constructor module, so it does not
    require a predefined adjacency matrix (though it will still work
    when one is unavailable).
    """

    def __init__(self) -> None:
        self._model: gtnet | None = None
        self._scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: MTGNNConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "MTGNN"

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

        # -- Fit scalers (per-feature z-score, NaN-aware) ------------------
        # Separate y-scaler so inverse_transform uses target stats and
        # never silently relies on "first D_out features are the targets".
        scaler = _fit_feature_scaler(x_train, input_dim, device)
        y_scaler = _fit_feature_scaler(y_train, output_dim, device)
        self._scaler = scaler
        self._y_scaler = y_scaler

        # -- Normalize x and handle NaN ------------------------------------
        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))

        # -- Targets keep NaN so masked_huber_loss can ignore them ---------
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

        # -- Predefined adjacency ------------------------------------------
        # MTGNN's graph mode is exclusive: when ``buildA_true=True`` it
        # learns a graph from scratch and the predefined adjacency is
        # ignored; when ``buildA_true=False`` it reads from
        # ``predefined_A`` instead.  We forward the provided ``adj`` so
        # the second mode is actually usable end-to-end (callers can
        # tune ``buildA_true`` to compare the two).
        predefined_A: torch.Tensor | None = None
        if not cfg.buildA_true and adj is not None:
            predefined_A = torch.tensor(
                np.asarray(adj, dtype=np.float32), device=device,
            )

        # -- Build model ---------------------------------------------------
        model = gtnet(
            gcn_true=cfg.gcn_true,
            buildA_true=cfg.buildA_true,
            gcn_depth=cfg.gcn_depth,
            num_nodes=num_nodes,
            device=device,
            predefined_A=predefined_A,
            static_feat=None,
            dropout=cfg.dropout,
            # `subgraph_size` is the top-k neighbours per node inside MTGNN's
            # graph constructor, so it cannot exceed `num_nodes - 1`
            # (a node's top-k must exclude itself). Tune this hyperparameter
            # per dataset — 20 is the paper default for ~200-node graphs.
            subgraph_size=min(cfg.subgraph_size, max(num_nodes - 1, 1)),
            node_dim=cfg.node_dim,
            dilation_exponential=cfg.dilation_exponential,
            conv_channels=cfg.conv_channels,
            residual_channels=cfg.residual_channels,
            skip_channels=cfg.skip_channels,
            end_channels=cfg.end_channels,
            seq_length=in_steps,
            in_dim=input_dim,
            out_dim=out_steps * output_dim,
            layers=cfg.layers,
            propalpha=cfg.propalpha,
            tanhalpha=cfg.tanhalpha,
            layer_norm_affline=cfg.layer_norm_affline,
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
            # ---- train one epoch ----
            model.train()
            batch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                with train_sw:
                    # x_batch: (B, T, N, D) -> (B, D, N, T) for gtnet
                    x_in = x_batch.permute(0, 3, 2, 1)
                    out = model(x_in)  # (B, out_steps*output_dim, N, 1)
                    out = (
                        out.squeeze(-1)
                        .reshape(-1, out_steps, output_dim, num_nodes)
                        .permute(0, 1, 3, 2)
                    )  # (B, out_steps, N, output_dim)
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

            # ---- validate ----
            model.eval()
            batch_losses_val: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    with train_sw:
                        x_in = x_batch.permute(0, 3, 2, 1)
                        out = model(x_in)
                        out = (
                            out.squeeze(-1)
                            .reshape(-1, out_steps, output_dim, num_nodes)
                            .permute(0, 1, 3, 2)
                        )
                        out = y_scaler.inverse_transform(out)
                        loss = criterion(out, y_batch)
                        batch_loss = loss.item()
                    batch_losses_val.append(batch_loss)
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
        out_steps = self._out_steps
        output_dim = self._output_dim

        x_test_n = torch.nan_to_num(scaler.transform(x_test.to(device)))
        num_nodes = x_test.shape[2]

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
                    x_in = x_batch.permute(0, 3, 2, 1)
                    out = model(x_in)
                    out = (
                        out.squeeze(-1)
                        .reshape(-1, out_steps, output_dim, num_nodes)
                        .permute(0, 1, 3, 2)
                    )
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
