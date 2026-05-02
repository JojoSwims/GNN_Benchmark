"""DynGWN (Dyn-GWN) benchmark wrapper.

Wraps the cleaned-up Dyn-GWN model so it satisfies the
``BenchmarkModel`` contract.  The harness extension is documented in
:meth:`fit`/`predict`: the wrapper accepts an optional
``dynamic_adj_full`` keyword argument carrying the dataset's
time-varying graph snapshots, plus ``dynamic_window_starts`` aligning
each sliding-window sample to its first index in the snapshot timeline.

Flow → adjacency preprocessing
------------------------------
The dynamic-edge tables produced by the divvy and eu_load loaders carry
**flow** (trip counts, MW), not graph "cost".  Flows can be:

  * Wildly heavy-tailed (most station pairs see zero trips per hour;
    busy pairs see hundreds).
  * Negative — eu_load reports signed cross-zone power flows.

Feeding raw flow magnitudes into a softmax-over-destinations attention
makes a single hot edge dominate the message-passing entirely.  The
wrapper therefore applies, per snapshot, ``log1p(|flow|)`` followed by a
per-node row-mean normalisation, so the model sees a well-conditioned
[0, ~5] dynamic adjacency that softmax can route over.  This matches
Dyn-GWN's own ``transformation='absolute'`` plus the inner
softmax-on-destinations the model applies.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.models.DynGWN.model import dyngwn
from gnn_benchmark.models.GWN.util import (
    asym_adj,
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    sym_adj,
)
from gnn_benchmark.utils.losses import masked_huber_loss
from gnn_benchmark.utils.timing import Stopwatch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DynGWNConfig:
    """Training and architecture configuration for :class:`DynGWNModel`.

    Mirrors :class:`GWNConfig` plus knobs specific to the time-varying
    graph branch.
    """

    # Training hyper-parameters
    lr: float = 0.001
    weight_decay: float = 0.0001
    eps: float = 1e-8
    batch_size: int = 64
    max_epochs: int = 100
    early_stop: int = 30
    clip_grad: float | None = 5.0

    # Architecture (paper defaults)
    nhid: int = 32
    dropout: float = 0.3
    blocks: int = 4
    layers: int = 2
    kernel_size: int = 2

    # Static-graph branch
    gcn_bool: bool = True
    addaptadj: bool = True
    adjtype: str = "doubletransition"

    # Dynamic-graph branch
    dynamic_gcn_bool: bool = True
    # ``"log1p_abs_rownorm"`` applies log1p(|flow|) per snapshot then
    # divides each row by its mean (zero rows pass through unchanged).
    # ``"log1p_abs"`` skips the row-norm.  ``"abs"`` only takes the
    # absolute value.  ``"identity"`` passes raw flow through (matches
    # the upstream demo path; can blow up the inner softmax).
    flow_transform: str = "log1p_abs_rownorm"

    # LR scheduler
    use_lr_scheduler: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [20, 40, 60, 80])
    lr_decay_ratio: float = 0.5

    # Runtime
    seed: int | None = None
    device: str = "auto"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FeatureScaler:
    """Per-feature z-score scaler that lives on a torch device."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def to(self, device: torch.device) -> "_FeatureScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        return x * self.std[:d] + self.mean[:d]


def _fit_feature_scaler(
    tensor: torch.Tensor, feature_dim: int, device: torch.device
) -> _FeatureScaler:
    flat = tensor.reshape(-1, feature_dim)
    mean = torch.nanmean(flat, dim=0)
    diff_sq = (flat - mean) ** 2
    count = (~torch.isnan(flat)).sum(dim=0).float()
    std = torch.sqrt(torch.nansum(diff_sq, dim=0) / count)
    std = torch.clamp(std, min=1e-8)
    return _FeatureScaler(mean, std).to(device)


def _resolve_config(config: Any) -> DynGWNConfig:
    if config is None:
        return DynGWNConfig()
    if isinstance(config, DynGWNConfig):
        return config
    if isinstance(config, dict):
        return DynGWNConfig(**config)
    raise TypeError(
        f"config must be DynGWNConfig, dict, or None — got {type(config)}"
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _seed_everything(seed: int) -> None:
    import os, random  # noqa: E401
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_supports(
    adj: np.ndarray | None, adjtype: str, device: torch.device
) -> tuple[list[torch.Tensor] | None, torch.Tensor | None]:
    """Same support construction as the GWN wrapper."""
    if adj is None:
        return None, None

    adj_mx = np.array(adj, dtype=np.float32)

    if adjtype == "scalap":
        supports_np = [np.array(calculate_scaled_laplacian(adj_mx))]
    elif adjtype == "normlap":
        supports_np = [
            np.array(
                calculate_normalized_laplacian(adj_mx)
                .astype(np.float32)
                .todense()
            )
        ]
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
    return supports, supports[0]


def _preprocess_flow_snapshots(
    snapshots: np.ndarray, mode: str
) -> np.ndarray:
    """Apply the configured per-snapshot flow → adjacency transform.

    ``snapshots`` is float32 of shape ``(T, N, N)``.  Returns a tensor of
    the same shape with non-negative entries suitable for the inner
    softmax over destination nodes.
    """
    if mode == "identity":
        return snapshots.astype(np.float32, copy=False)

    out = np.abs(snapshots, dtype=np.float32)

    if mode == "abs":
        return out

    if mode in ("log1p_abs", "log1p_abs_rownorm"):
        np.log1p(out, out=out)
        if mode == "log1p_abs":
            return out
        # Row-mean normalise per snapshot. Empty rows stay zero so
        # downstream softmax falls back to the uniform distribution.
        row_means = out.mean(axis=2, keepdims=True)
        nonzero = row_means > 0
        np.divide(out, row_means, out=out, where=nonzero)
        return out

    raise ValueError(f"Unknown flow_transform={mode!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DynGWNModel(BenchmarkModel):
    """Benchmark-ready Dyn-GWN wrapper.

    Contract extension
    ------------------
    On top of the standard ``fit/predict(x_train, y_train, x_val, y_val,
    adj, config)`` signature, this wrapper accepts two extra keyword
    arguments populated by the runner:

    ``dynamic_adj_full``
        ``np.ndarray`` of shape ``(T_total, N, N)``, float32, carrying
        the dataset's snapshot adjacency aligned with the series's
        sorted-unique timestamps.  ``None`` raises ``ValueError`` —
        Dyn-GWN's whole point is the dynamic-graph branch, so a dataset
        without snapshots cannot use it.

    ``dynamic_window_starts``
        ``np.ndarray`` of shape ``(S,)`` int64, mapping every sliding
        window in ``x_*`` to its first timestep index along ``T_total``
        so the wrapper can slice ``dynamic_adj_full`` per batch.

    The harness already populates these for any IR with
    ``dynamic_edges`` (currently ``divvy-bikeshare-dynamic``,
    ``eu-load-dynamic``, ``eu-load-both``).
    """

    def __init__(self) -> None:
        self._model: dyngwn | None = None
        self._scaler: _FeatureScaler | None = None
        self._y_scaler: _FeatureScaler | None = None
        self._device: torch.device | None = None
        self._cfg: DynGWNConfig | None = None
        self._out_steps: int | None = None
        self._output_dim: int | None = None
        self._num_nodes: int | None = None
        self._dynamic_adj_full: torch.Tensor | None = None
        self._train_compute_sec: float = 0.0
        self._infer_compute_sec: float = 0.0

    @property
    def name(self) -> str:
        return "DynGWN"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
        *,
        dynamic_adj_full: np.ndarray | None = None,
        dynamic_window_starts: np.ndarray | None = None,
        dynamic_window_starts_val: np.ndarray | None = None,
    ) -> TrainingHistory:
        cfg = _resolve_config(config)
        self._cfg = cfg

        if cfg.seed is not None:
            _seed_everything(cfg.seed)

        device = _resolve_device(cfg.device)
        self._device = device

        # -- Validate dynamic graph inputs ---------------------------------
        if cfg.dynamic_gcn_bool:
            if dynamic_adj_full is None or dynamic_window_starts is None:
                raise ValueError(
                    "DynGWN requires a dataset with dynamic_edges. "
                    "Either pick a registry entry whose IR carries "
                    "snapshots (e.g. 'divvy-bikeshare-dynamic', "
                    "'eu-load-dynamic') or set dynamic_gcn_bool=False."
                )
            if dynamic_window_starts_val is None:
                raise ValueError(
                    "dynamic_window_starts_val must accompany "
                    "dynamic_window_starts when fitting DynGWN."
                )

        # -- Infer shapes --------------------------------------------------
        num_nodes = x_train.shape[2]
        in_steps = x_train.shape[1]
        out_steps = y_train.shape[1]
        input_dim = x_train.shape[3]
        output_dim = y_train.shape[3]
        self._num_nodes = num_nodes
        self._out_steps = out_steps
        self._output_dim = output_dim

        # -- Fit feature scalers -------------------------------------------
        scaler = _fit_feature_scaler(x_train, input_dim, device)
        y_scaler = _fit_feature_scaler(y_train, output_dim, device)
        self._scaler = scaler
        self._y_scaler = y_scaler

        x_train_n = torch.nan_to_num(scaler.transform(x_train.to(device)))
        x_val_n = torch.nan_to_num(scaler.transform(x_val.to(device)))
        y_train_d = y_train.to(device)
        y_val_d = y_val.to(device)

        # -- Static supports (for the static-graph branch) -----------------
        if not cfg.gcn_bool:
            supports, aptinit = None, None
        else:
            supports, aptinit = _build_supports(adj, cfg.adjtype, device)

        # -- Pre-process & cache the full dynamic adjacency ----------------
        # Cached on CPU (often big) and indexed per-batch on GPU. For
        # eu_load (~30 zones) this is tiny; for divvy (~200 stations)
        # it is on the order of (T_total, N, N) ≈ a few GB float32 —
        # still fits in host RAM and dwarfs any per-batch slice.
        if cfg.dynamic_gcn_bool:
            processed = _preprocess_flow_snapshots(
                np.asarray(dynamic_adj_full, dtype=np.float32),
                cfg.flow_transform,
            )
            self._dynamic_adj_full = torch.from_numpy(processed)
        else:
            self._dynamic_adj_full = None

        # -- DataLoaders ---------------------------------------------------
        # Window-start indices ride alongside (x, y) so each batch knows
        # which slice of dynamic_adj_full to pull.
        if cfg.dynamic_gcn_bool:
            ws_train = torch.from_numpy(
                np.asarray(dynamic_window_starts, dtype=np.int64)
            )
            ws_val = torch.from_numpy(
                np.asarray(dynamic_window_starts_val, dtype=np.int64)
            )
            train_set = TensorDataset(x_train_n, y_train_d, ws_train)
            val_set = TensorDataset(x_val_n, y_val_d, ws_val)
        else:
            train_set = TensorDataset(x_train_n, y_train_d)
            val_set = TensorDataset(x_val_n, y_val_d)

        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, shuffle=False
        )

        # -- Build model ---------------------------------------------------
        model = dyngwn(
            num_nodes=num_nodes,
            dropout=cfg.dropout,
            supports=supports,
            gcn_bool=cfg.gcn_bool,
            addaptadj=cfg.addaptadj if cfg.gcn_bool else False,
            aptinit=aptinit,
            dynamic_gcn_bool=cfg.dynamic_gcn_bool,
            dynamic_supports_len=1,
            in_dim=input_dim,
            input_sequence_dim=in_steps,
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

        train_sw = Stopwatch()
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(cfg.max_epochs):
            model.train()
            batch_losses: list[float] = []
            for batch in train_loader:
                with train_sw:
                    out = self._forward_batch(
                        model, batch, in_steps, out_steps,
                        output_dim, num_nodes, device, cfg.dynamic_gcn_bool,
                    )
                    y_batch = batch[1]
                    out = y_scaler.inverse_transform(out)
                    loss = criterion(out, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.clip_grad:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.clip_grad
                        )
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
                for batch in val_loader:
                    with train_sw:
                        out = self._forward_batch(
                            model, batch, in_steps, out_steps,
                            output_dim, num_nodes, device,
                            cfg.dynamic_gcn_bool,
                        )
                        y_batch = batch[1]
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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: torch.Tensor,
        adj: np.ndarray | None,
        config: Any,
        *,
        dynamic_adj_full: np.ndarray | None = None,
        dynamic_window_starts: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            self._model is None
            or self._scaler is None
            or self._y_scaler is None
        ):
            raise RuntimeError("Call fit() before predict().")

        cfg = _resolve_config(config)
        device = self._device
        scaler = self._scaler
        y_scaler = self._y_scaler
        model = self._model
        out_steps = self._out_steps
        output_dim = self._output_dim
        num_nodes = self._num_nodes
        in_steps = x_test.shape[1]

        if cfg.dynamic_gcn_bool and dynamic_window_starts is None:
            raise ValueError(
                "DynGWN.predict requires dynamic_window_starts when "
                "dynamic_gcn_bool=True."
            )

        x_test_n = torch.nan_to_num(scaler.transform(x_test.to(device)))

        if cfg.dynamic_gcn_bool:
            ws_test = torch.from_numpy(
                np.asarray(dynamic_window_starts, dtype=np.int64)
            )
            test_set = TensorDataset(x_test_n, ws_test)
        else:
            test_set = TensorDataset(x_test_n)

        loader = DataLoader(
            test_set, batch_size=cfg.batch_size, shuffle=False
        )

        infer_sw = Stopwatch()
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                with infer_sw:
                    out = self._forward_batch(
                        model, batch, in_steps, out_steps,
                        output_dim, num_nodes, device,
                        cfg.dynamic_gcn_bool,
                        predict_only=True,
                    )
                    out = y_scaler.inverse_transform(out)
                    out_cpu = out.cpu().numpy()
                preds.append(out_cpu)

        self._infer_compute_sec = infer_sw.elapsed
        return np.concatenate(preds, axis=0)

    # ------------------------------------------------------------------
    # Shared per-batch forward
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        model: dyngwn,
        batch: tuple,
        in_steps: int,
        out_steps: int,
        output_dim: int,
        num_nodes: int,
        device: torch.device,
        dynamic_gcn_bool: bool,
        predict_only: bool = False,
    ) -> torch.Tensor:
        """Run a single batch through Dyn-GWN.

        Centralised so fit (which has y in the batch) and predict
        (which doesn't) share the same conventions.
        """
        x_batch = batch[0]
        # (B, T, N, D) -> (B, D, N, T)
        x_in = x_batch.permute(0, 3, 2, 1)

        graph_input = None
        if dynamic_gcn_bool:
            ws = batch[-1]  # window-start indices
            graph_input = self._slice_dynamic_graph(
                ws, in_steps, num_nodes, device
            )

        out = model(x_in, graph_input=graph_input)
        # Trim the trailing time dim and reshape to the contract's
        # (B, out_steps, N, output_dim) layout.  ``out`` is
        # (B, out_steps*output_dim, N, T_out); take the last time step.
        out = (
            out[..., -1]
            .reshape(-1, out_steps, output_dim, num_nodes)
            .permute(0, 1, 3, 2)
        )
        return out

    def _slice_dynamic_graph(
        self,
        window_starts: torch.Tensor,
        in_steps: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the per-batch ``(B, 1, N*N, T)`` dynamic-graph tensor."""
        if self._dynamic_adj_full is None:
            raise RuntimeError("dynamic_adj_full was not cached at fit time.")
        full = self._dynamic_adj_full  # (T_total, N, N) on CPU
        starts = window_starts.cpu().numpy()
        # Gather windows.  np.stack of T_total slices is fine for
        # batches in the dozens; faster than per-step Python indexing.
        windows = np.stack(
            [full[s : s + in_steps].numpy() for s in starts], axis=0
        )  # (B, T, N, N)
        # (B, T, N*N) -> (B, 1, N*N, T) for Dyn-GWN's conv stack.
        B, T, N, _ = windows.shape
        windows = windows.reshape(B, T, N * N)
        graph_input = (
            torch.from_numpy(windows).to(device).permute(0, 2, 1).unsqueeze(1)
        )
        return graph_input

    # ------------------------------------------------------------------
    # Config / compute accessors
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        if self._cfg is None:
            return {}
        return asdict(self._cfg)

    def get_train_compute_sec(self) -> float | None:
        return self._train_compute_sec or None

    def get_inference_compute_sec(self) -> float | None:
        return self._infer_compute_sec or None
