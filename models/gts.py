"""GTS (Graph for Time Series) benchmark wrapper.

Wraps the standalone GTS model (Shang et al., "Discrete Graph Structure
Learning for Forecasting Multiple Time Series", ICLR 2021) so it
satisfies the ``BenchmarkModel`` contract.

Usage::

    from gnn_benchmark.models import GTSModel, GTSConfig

    cfg = GTSConfig(max_epochs=100, batch_size=64)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(GTSModel(), config=cfg)
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
from gnn_benchmark.models.GTS.model import GTSModel as _GTSNet
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
class GTSConfig:
    """Training and architecture configuration for :class:`GTSModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = GTSConfig(lr=0.005, max_epochs=100)
    """

    # Training hyper-parameters
    lr: float = 0.005
    weight_decay: float = 0.0
    eps: float = 1e-3                        # GTS-specific (paper default)
    batch_size: int = 64
    max_epochs: int = 100
    early_stop: int = 30
    clip_grad: float | None = 5.0            # gradient-norm clip; None disables

    # Architecture (paper defaults)
    rnn_units: int = 64
    num_rnn_layers: int = 1
    max_diffusion_step: int = 3
    cl_decay_steps: int = 2000
    use_curriculum_learning: bool = True
    embedding_dim: int = 100
    temperature: float = 0.5

    # Latent graph feature pool
    # Node-feature series length used to build the latent graph.  If None,
    # the full training time series (reconstructed from x_train) is used.
    feas_len: int | None = None

    # Optional BCE regularization of the learned graph against a reference.
    # When > 0 and an adjacency is provided to fit(), adds
    #   loss_total = loss_pred + bce_weight * BCE(learned_adj, adj)
    # for the first ``epoch_use_regularization`` epochs.
    bce_weight: float = 1.0
    epoch_use_regularization: int = 0

    # LR scheduler (multi-step)
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [20, 30, 40])
    lr_decay_ratio: float = 0.1

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


def _resolve_config(config: Any) -> GTSConfig:
    if config is None:
        return GTSConfig()
    if isinstance(config, GTSConfig):
        return config
    if isinstance(config, dict):
        return GTSConfig(**config)
    raise TypeError(
        f"config must be GTSConfig, dict, or None — got {type(config)}"
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _seed_everything(seed: int) -> None:
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_node_feas(x_train: torch.Tensor, feas_len: int | None) -> torch.Tensor:
    """Reconstruct a ``(T_feas, N)`` series from the windowed training set.

    Uses feature 0 and the first timestep of each window, which approximates
    the original continuous series when windows are stride-1 contiguous.
    NaNs are filled with 0 after z-scoring to keep the CNN well-behaved.
    """
    # x_train: (S, T, N, D). Use first feature, first timestep per window.
    series = x_train[:, 0, :, 0]  # (S, N)
    if feas_len is not None and series.shape[0] > feas_len:
        series = series[:feas_len]
    return series


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GTSModel(BenchmarkModel):
    """Benchmark-ready GTS wrapper.

    The underlying network learns a latent graph from a small "feature
    pool" of node-level time series via 1D convolutions plus a
    Gumbel-softmax edge sampler, then runs an encoder-decoder with
    diffusion-convolutional GRU cells on top.
    """

    def __init__(self) -> None:
        self._model: _GTSNet | None = None
        self._scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: GTSConfig | None = None
        self._node_feas: torch.Tensor | None = None
        self._num_nodes: int | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "GTS"

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
        self._num_nodes = num_nodes
        self._out_steps = out_steps
        self._output_dim = output_dim

        # -- Fit scalers (per-feature z-score, NaN-aware) ------------------
        # Separate y-scaler so inverse_transform uses target stats and
        # never silently relies on "first D_out features are the targets".
        scaler = _fit_feature_scaler(x_train, input_dim, device)
        y_scaler = _fit_feature_scaler(y_train, output_dim, device)
        self._scaler = scaler
        self._y_scaler = y_scaler

        # -- Build node-feature pool for latent graph ---------------------
        node_feas = _build_node_feas(x_train, cfg.feas_len).to(device)
        # Z-score only feature 0 since that is what node_feas contains.
        node_feas = (node_feas - scaler.mean[0]) / scaler.std[0]
        node_feas = torch.nan_to_num(node_feas)
        t_feas = node_feas.shape[0]
        # dim_fc = 16 * (T_feas after two Conv1d(kernel=10, stride=1))
        dim_fc = 16 * (t_feas - 18)
        if dim_fc <= 0:
            raise ValueError(
                "Node-feature series too short for GTS graph learner: "
                f"got T_feas={t_feas}, need > 18. Increase training window "
                "count or reduce cfg.feas_len."
            )
        self._node_feas = node_feas

        # -- Normalize inputs and move to device --------------------------
        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))
        # Targets keep NaN so masked_huber_loss can ignore missing positions.
        y_train_d = y_train.to(device)
        y_val_d = y_val.to(device)

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

        # -- Reference adjacency for the optional BCE term ----------------
        adj_ref: torch.Tensor | None = None
        if adj is not None and cfg.bce_weight > 0 and cfg.epoch_use_regularization > 0:
            adj_bin = (np.asarray(adj, dtype=np.float32) != 0).astype(np.float32)
            adj_ref = torch.tensor(adj_bin, device=device)

        # -- Build model ---------------------------------------------------
        model = _GTSNet(
            temperature=cfg.temperature,
            cl_decay_steps=cfg.cl_decay_steps,
            filter_type="dual_random_walk",
            horizon=out_steps,
            input_dim=input_dim,
            max_diffusion_step=cfg.max_diffusion_step,
            num_nodes=num_nodes,
            num_rnn_layers=cfg.num_rnn_layers,
            output_dim=output_dim,
            rnn_units=cfg.rnn_units,
            seq_len=in_steps,
            use_curriculum_learning=cfg.use_curriculum_learning,
            dim_fc=dim_fc,
            embedding_dim=cfg.embedding_dim,
        ).to(device)

        # -- Training setup -----------------------------------------------
        criterion = masked_huber_loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
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
        batches_seen = 0

        for epoch in range(cfg.max_epochs):
            use_reg = (
                epoch < cfg.epoch_use_regularization and adj_ref is not None
            )

            # ---- train one epoch ----
            model.train()
            batch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                with train_sw:
                    # (B, T, N, D) -> (T, B, N*D)
                    seq = x_batch.permute(1, 0, 2, 3).reshape(
                        in_steps, -1, num_nodes * input_dim
                    )
                    # Teacher-forcing labels for curriculum learning. Decoder
                    # feedback cannot tolerate NaN, so the labels fed back
                    # into the model must be filled — but the loss below is
                    # still computed against the NaN-bearing ``y_batch`` so
                    # missing positions do not contribute to gradients.
                    # (B, T_out, N, D_out) -> (T_out, B, N*D_out)
                    lbl = torch.nan_to_num(
                        y_batch[..., :output_dim]
                    ).permute(1, 0, 2, 3).reshape(
                        out_steps, -1, num_nodes * output_dim
                    )

                    output, edge_probs = model(
                        seq,
                        node_feas,
                        temp=cfg.temperature,
                        gumbel_hard=True,
                        labels=lbl,
                        batches_seen=batches_seen,
                    )
                    # output: (T_out, B, N*D_out) -> (B, T_out, N, D_out)
                    output = output.reshape(
                        out_steps, -1, num_nodes, output_dim
                    ).permute(1, 0, 2, 3)
                    output = y_scaler.inverse_transform(output)
                    loss_pred = criterion(output, y_batch)

                    if use_reg:
                        probs = edge_probs.reshape(-1)
                        true_edges = adj_ref.reshape(-1)
                        loss_g = nn.functional.binary_cross_entropy(
                            probs.clamp(1e-7, 1 - 1e-7), true_edges
                        )
                        loss = loss_pred + cfg.bce_weight * loss_g
                    else:
                        loss = loss_pred

                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.clip_grad:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    batch_loss = loss_pred.item()
                batch_losses.append(batch_loss)
                batches_seen += 1

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
                        seq = x_batch.permute(1, 0, 2, 3).reshape(
                            in_steps, -1, num_nodes * input_dim
                        )
                        output, _ = model(
                            seq,
                            node_feas,
                            temp=cfg.temperature,
                            gumbel_hard=True,
                        )
                        output = output.reshape(
                            out_steps, -1, num_nodes, output_dim
                        ).permute(1, 0, 2, 3)
                        output = y_scaler.inverse_transform(output)
                        batch_loss = criterion(output, y_batch).item()
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

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return TrainingHistory(train_loss=train_losses, val_loss=val_losses)

    def predict(
        self,
        x_test: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
    ) -> np.ndarray:
        if (
            self._model is None
            or self._scaler is None
            or self._y_scaler is None
            or self._node_feas is None
        ):
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        scaler = self._scaler
        y_scaler = self._y_scaler
        model = self._model
        num_nodes = self._num_nodes
        out_steps = self._out_steps
        output_dim = self._output_dim
        in_steps = x_test.shape[1]
        input_dim = x_test.shape[3]

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
                    seq = x_batch.permute(1, 0, 2, 3).reshape(
                        in_steps, -1, num_nodes * input_dim
                    )
                    output, _ = model(
                        seq,
                        self._node_feas,
                        temp=cfg.temperature,
                        gumbel_hard=True,
                    )
                    output = output.reshape(
                        out_steps, -1, num_nodes, output_dim
                    ).permute(1, 0, 2, 3)
                    output = y_scaler.inverse_transform(output)
                    out_cpu = output.cpu().numpy()
                preds.append(out_cpu)

        self._infer_compute_sec = infer_sw.elapsed
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.concatenate(preds, axis=0)

    def get_config(self) -> dict:
        if self._cfg is None:
            return {}
        return asdict(self._cfg)

    def get_train_compute_sec(self) -> float | None:
        return self._train_compute_sec or None

    def get_inference_compute_sec(self) -> float | None:
        return self._infer_compute_sec or None
