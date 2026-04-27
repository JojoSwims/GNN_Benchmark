"""D2STGNN benchmark wrapper.

Wraps the standalone D2STGNN (Decoupled Dynamic Spatial-Temporal Graph
Neural Network) model so it satisfies the ``BenchmarkModel`` contract.

Usage::

    from gnn_benchmark.models import D2STGNNModel, D2STGNNConfig

    cfg = D2STGNNConfig(max_epochs=80, batch_size=32)
    runner = BenchmarkRunner(workspace_dir="./ws", datasets=["pems08"])
    result = runner.run(D2STGNNModel(), config=cfg)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.models.D2STGNN.models.model import D2STGNN
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
class D2STGNNConfig:
    """Training and architecture configuration for :class:`D2STGNNModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = D2STGNNConfig(lr=0.001, max_epochs=100)
    """

    # Training hyper-parameters
    lr: float = 0.002
    weight_decay: float = 1e-5
    eps: float = 1e-8                        # Adam epsilon
    batch_size: int = 32
    max_epochs: int = 80
    early_stop: int = 30
    clip_grad: float | None = 5.0            # gradient-norm clip; None disables

    # Architecture (paper defaults)
    num_hidden: int = 32
    node_hidden: int = 10
    time_emb_dim: int = 10
    dropout: float = 0.1
    k_t: int = 3
    k_s: int = 2
    gap: int = 3
    num_modalities: int = 2

    # LR scheduler (multi-step)
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(
        default_factory=lambda: [30, 38, 46, 54, 62, 70],
    )
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


def _resolve_config(config: Any) -> D2STGNNConfig:
    if config is None:
        return D2STGNNConfig()
    if isinstance(config, D2STGNNConfig):
        return config
    if isinstance(config, dict):
        return D2STGNNConfig(**config)
    raise TypeError(
        f"config must be D2STGNNConfig, dict, or None — got {type(config)}"
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


def _transition_matrix(adj: np.ndarray) -> np.ndarray:
    """Row-normalize: P = D^{-1} A."""
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return np.asarray(d_mat.dot(adj_sp).astype(np.float32).todense())


def _build_d2stgnn_adjs(
    adj: np.ndarray | None,
    num_nodes: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build D2STGNN-style doubletransition adjacency list.

    When ``adj`` is available the result is ``[P^T, P'^T]`` where
    ``P = D^{-1}A`` and ``P' = D'^{-1}A^T`` (matching the convention
    used in the original D2STGNN code).

    When ``adj`` is ``None``, returns two all-ones matrices so that the
    ``Mask`` module does not constrain the dynamic graphs.
    """
    if adj is None:
        return [
            torch.ones(num_nodes, num_nodes, device=device),
            torch.ones(num_nodes, num_nodes, device=device),
        ]

    adj_mx = np.array(adj, dtype=np.float32)
    P = _transition_matrix(adj_mx)
    Pt = _transition_matrix(adj_mx.T)
    return [
        torch.tensor(P.T, dtype=torch.float32, device=device),
        torch.tensor(Pt.T, dtype=torch.float32, device=device),
    ]


def _choose_gap(seq_out_len: int, preferred_gap: int) -> int:
    """Return a gap that evenly divides *seq_out_len*.

    Falls back to 1 when the preferred gap does not divide evenly.
    """
    if preferred_gap > 0 and seq_out_len % preferred_gap == 0:
        return preferred_gap
    # Try common divisors in descending order
    for g in range(preferred_gap, 0, -1):
        if seq_out_len % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class D2STGNNModel(BenchmarkModel):
    """Benchmark-ready D2STGNN wrapper.

    The underlying ``D2STGNN`` ``nn.Module`` is instantiated at ``fit()``
    time once tensor shapes are known.

    Since the benchmark does not provide time-of-day / day-of-week
    features, the wrapper pads the input with two zero channels.  The
    model's time embedding parameters (``T_i_D_emb[0]``,
    ``D_i_W_emb[0]``) then act as constant learnable biases, preserving
    the architecture without time awareness.
    """

    def __init__(self) -> None:
        self._model: D2STGNN | None = None
        self._scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: D2STGNNConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    # ---- BenchmarkModel interface ----------------------------------------

    @property
    def name(self) -> str:
        return "D2STGNN"

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

        # -- Build adjacency supports -------------------------------------
        adjs = _build_d2stgnn_adjs(adj, num_nodes, device)

        # -- DataLoaders (splits stay on CPU; batches stream to device) ---
        # Keeping the raw tensors off GPU is the main memory saving: only
        # one batch at a time is resident. Normalisation, NaN-handling and
        # the two zero time-feature channels are applied per-batch below.
        pin = device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=pin,
        )

        # -- Build model ---------------------------------------------------
        gap = _choose_gap(out_steps, cfg.gap)
        model_args = {
            "num_feat": input_dim,
            "num_hidden": cfg.num_hidden,
            "node_hidden": cfg.node_hidden,
            "time_emb_dim": cfg.time_emb_dim,
            "seq_length": out_steps,
            "k_t": cfg.k_t,
            "k_s": cfg.k_s,
            "gap": gap,
            "num_nodes": num_nodes,
            "adjs": adjs,
            "dropout": cfg.dropout,
            "device": device,
            "num_modalities": cfg.num_modalities,
            "output_dim": output_dim,
        }

        model = D2STGNN(**model_args).to(device)

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
                # Host→device copy + scaling are NOT timed — they are data
                # movement, not compute. Only the model evaluation, loss,
                # backward and optimizer step contribute to the budget.
                x_batch, y_batch = self._prepare_batch(
                    x_batch, y_batch, scaler, device,
                )
                with train_sw:
                    out = model(x_batch)  # (B, out_steps, N, output_dim)
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
                    x_batch, y_batch = self._prepare_batch(
                        x_batch, y_batch, scaler, device,
                    )
                    with train_sw:
                        out = model(x_batch)
                        out = y_scaler.inverse_transform(out)
                        loss = criterion(out, y_batch)
                        batch_loss = loss.item()
                    batch_losses_val.append(batch_loss)
            val_loss = float(np.mean(batch_losses_val))
            val_losses.append(val_loss)

            # ---- early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Keep the best snapshot on CPU so it does not compete for
                # VRAM with the live model + optimizer state.
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
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
        if self._model is None or self._scaler is None or self._y_scaler is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        scaler = self._scaler
        y_scaler = self._y_scaler
        model = self._model

        pin = device.type == "cuda"
        loader = DataLoader(
            TensorDataset(x_test),
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=pin,
        )

        infer_sw = Stopwatch()
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                # Host→device transfer + scaling are excluded from the
                # compute budget — only the model.forward call is timed.
                x_batch = x_batch.to(device, non_blocking=pin)
                x_batch = torch.nan_to_num(scaler.transform(x_batch))
                x_batch = self._pad_time_features(x_batch)
                with infer_sw:
                    out = model(x_batch)  # (B, out_steps, N, output_dim)
                    out = y_scaler.inverse_transform(out)
                    out_cpu = out.cpu().numpy()
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

    # ---- Private helpers -------------------------------------------------

    @classmethod
    def _prepare_batch(
        cls,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        scaler: _FeatureScaler,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move one batch to ``device`` and apply scaling/NaN/padding.

        Doing this per batch (instead of once for the whole split) keeps
        only a single mini-batch resident on GPU at a time.
        """
        non_blocking = device.type == "cuda"
        x_batch = x_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)
        x_batch = torch.nan_to_num(scaler.transform(x_batch))
        # Targets keep NaN so masked_huber_loss can ignore missing positions.
        x_batch = cls._pad_time_features(x_batch)
        return x_batch, y_batch

    @staticmethod
    def _pad_time_features(x: torch.Tensor) -> torch.Tensor:
        """Append 2 zero channels for time-in-day and day-in-week."""
        pad = torch.zeros(
            *x.shape[:-1], 2, device=x.device, dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=-1)
