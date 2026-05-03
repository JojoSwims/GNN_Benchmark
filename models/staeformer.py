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

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.models.STAEFormer.model.STAEformer import STAEformer
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
class STAEFormerConfig:
    """Training and architecture configuration for :class:`STAEFormerModel`.

    All fields have sensible defaults.  Override only what you need::

        cfg = STAEFormerConfig(lr=0.0005, max_epochs=300)
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0003
    eps: float = 1e-8                        # Adam epsilon
    batch_size: int = 16
    max_epochs: int = 200
    early_stop: int = 30
    clip_grad: float | None = 5.0            # gradient-norm clip; None disables

    # LR scheduler (multi-step decay — unified across all benchmark models)
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [20, 30])
    lr_decay_ratio: float = 0.1

    # Architecture (paper defaults — override only if experimenting)
    input_embedding_dim: int = 24
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Time-of-day embedding. ``steps_per_day`` is the number of input
    # timesteps per 24h period (288 for 5-min cadence, 24 for hourly,
    # 96 for 15-min, 1 for daily). It only matters when
    # ``tod_embedding_dim > 0``; the default keeps the embedding off
    # because the harness does not provide a TOD feature column. Set
    # both fields explicitly when wiring up TOD inputs by hand.
    steps_per_day: int = 288
    tod_embedding_dim: int = 0

    # Ablation: drop the spatial self-attention stack. Each node is then
    # processed independently by the temporal attention stack (and the
    # per-node output projection). The adaptive embedding stays intact
    # because it is a per-node positional feature, not an adjacency.
    no_graph: bool = False

    # Memory-saving knobs (no architectural impact).
    # ``use_amp`` runs the forward/loss under ``torch.autocast`` in the
    # given dtype — bf16 halves activation memory for attention/FFN on
    # Ampere+ GPUs without needing a GradScaler. ``use_checkpoint``
    # enables activation checkpointing on the temporal/spatial attention
    # stacks (recompute during backward). ``eval_batch_size`` lets the
    # validation/inference DataLoader use a smaller batch than training
    # so peak VRAM is not pushed higher under no_grad.
    use_amp: bool = False
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16"
    use_checkpoint: bool = False
    eval_batch_size: int | None = None

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
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: STAEFormerConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

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

        # -- Fit scalers (per-feature z-score, NaN-aware) ------------------
        # Separate y-scaler so inverse_transform uses target stats and
        # never silently relies on "first D_out features are the targets".
        scaler = _fit_feature_scaler(x_train, input_dim, device)
        y_scaler = _fit_feature_scaler(y_train, output_dim, device)
        self._scaler = scaler
        self._y_scaler = y_scaler

        # -- DataLoaders (splits stay on CPU; batches stream to device) ---
        # Keeping the raw tensors off GPU is the main memory saving: only
        # one batch at a time is resident. Normalisation and NaN-handling
        # are applied per-batch below.
        pin = device.type == "cuda"
        eval_bs = cfg.eval_batch_size or cfg.batch_size
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=eval_bs,
            shuffle=False,
            pin_memory=pin,
        )

        # -- Build model ---------------------------------------------------
        # ``steps_per_day`` only feeds the time-of-day embedding when
        # ``tod_embedding_dim > 0``; with the default 0 it is harmless,
        # but the field is still configurable for users who wire TOD
        # features into ``input_dim`` themselves.
        model = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=cfg.steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=cfg.input_embedding_dim,
            tod_embedding_dim=cfg.tod_embedding_dim,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=cfg.adaptive_embedding_dim,
            feed_forward_dim=cfg.feed_forward_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_mixed_proj=True,
            use_spatial_attn=not cfg.no_graph,
            use_checkpoint=cfg.use_checkpoint,
        ).to(device)

        # Mixed-precision context. bf16 needs no GradScaler (its dynamic
        # range matches fp32); fp16 would, so we restrict AMP to CUDA and
        # bf16 for now — this is a memory pass, not a numerics pass.
        amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
        amp_dtype = (
            torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        )

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
                # Host→device transfer + scaling are excluded from compute.
                x_batch, y_batch = self._prepare_batch(
                    x_batch, y_batch, scaler, device,
                )
                with train_sw:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        out = model(x_batch)
                        out = y_scaler.inverse_transform(out)
                        loss = criterion(out, y_batch)

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
                        with torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
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
                # VRAM with the live model + optimizer state. Drop the old
                # snapshot first so we don't briefly hold two copies on
                # host while a third (the live state_dict view) is on GPU,
                # then empty the CUDA cache to release the transient
                # staging buffers used by the device->host copies.
                best_state = None
                best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
                if device.type == "cuda":
                    torch.cuda.empty_cache()
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
        eval_bs = cfg.eval_batch_size or cfg.batch_size
        loader = DataLoader(
            TensorDataset(x_test),
            batch_size=eval_bs,
            shuffle=False,
            pin_memory=pin,
        )

        amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
        amp_dtype = (
            torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        )

        infer_sw = Stopwatch()
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(device, non_blocking=pin)
                x_batch = torch.nan_to_num(scaler.transform(x_batch))
                with infer_sw:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        out = model(x_batch)
                        out = y_scaler.inverse_transform(out)
                    # Cast back to fp32 before leaving GPU so downstream
                    # numpy consumers see the same dtype as before AMP.
                    out_cpu = out.float().cpu().numpy()
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

    @staticmethod
    def _prepare_batch(
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        scaler: _FeatureScaler,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move one batch to ``device`` and apply scaling / NaN handling.

        Doing this per batch (instead of once for the whole split) keeps
        only a single mini-batch resident on GPU at a time.
        """
        non_blocking = device.type == "cuda"
        x_batch = x_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)
        x_batch = torch.nan_to_num(scaler.transform(x_batch))
        # Targets keep NaN so masked_huber_loss can ignore missing positions.
        return x_batch, y_batch
