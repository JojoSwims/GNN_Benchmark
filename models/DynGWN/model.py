"""Dyn-GWN backbone (Ibrahim et al., ICAIF 2023).

This is a cleaned-up rewrite of the upstream reference implementation:

  * The on-the-fly partial-correlation graph builder
    (``graph_input_transform``) was dropped.  In this benchmark the
    dynamic graph is provided by the dataset (e.g. divvy trip-count
    snapshots, ENTSO-E cross-zone flow snapshots) and pre-processed by
    the wrapper, so the model receives ``graph_input`` already in the
    shape Dyn-GWN's dilated stack expects.
  * Bug fix: ``g_x`` was referenced in the receptive-field padding
    branch even when ``dynamic_gcn_bool=False``.  Padding is now gated
    on the flag.
  * Stripped the per-iteration debug prints and the ``tqdm.notebook``
    import (which crashes outside Jupyter).
  * Replaced the ``x.get_device() / torch.device(...)`` workaround with
    plain ``x.device``.
  * The ``end_conv_1`` time kernel is clamped to ``>= 1`` so the model
    is constructible when the input is shorter than the receptive
    field; the forward pass left-pads up to the receptive field as
    before.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    """Static graph convolution: ``y = A @ x`` over the node axis."""

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T), A: (N, N) -> (B, C, N, T)
        return torch.einsum("ncvl,vw->ncwl", x, A).contiguous()


class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class gcn(nn.Module):
    """K-hop diffusion convolution over a list of static supports."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 3,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in support:
            a = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class dynamic_nconv(nn.Module):
    """Time-varying graph convolution.

    ``A`` carries a separate adjacency for every timestep:

    .. code-block::

        x: (B, C, N, T)
        A: (B, C, N, N, T)
        out[b, c, k, l] = sum_j x[b, c, j, l] * A[b, c, j, k, l]
    """

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bijl,bijkl->bikl", x, A).contiguous()


class dynamic_gcn(nn.Module):
    """K-hop diffusion convolution over a list of time-varying supports."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        dynamic_supports_len: int = 1,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.dynamic_nconv = dynamic_nconv()
        self.mlp = linear(order * dynamic_supports_len * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(
        self, x: torch.Tensor, support: list[torch.Tensor]
    ) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for a in support:
            x1 = self.dynamic_nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.dynamic_nconv(x1, a)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class dyngwn(nn.Module):
    """Dyn-GWN model.

    Parameters mirror Graph WaveNet's ``gwnet`` plus three flags for the
    time-varying branch:

    Args:
        dynamic_gcn_bool: Enable the time-varying graph branch.
        dynamic_supports_len: Number of dynamic-support channels per
            timestep (1 for a single flow magnitude; >1 if the wrapper
            stacks multiple precomputed graphs).
        graph_input shape (forward): ``(B, dynamic_supports_len, N*N, T)``.
            The wrapper is responsible for any flow-to-adjacency
            preprocessing — the model treats ``graph_input`` as opaque
            features, applies a 1x1 lift, dilated gated convs in
            lockstep with the time-series branch, then a per-timestep
            softmax over destination nodes before message-passing.
    """

    def __init__(
        self,
        num_nodes: int,
        dropout: float = 0.3,
        supports: list[torch.Tensor] | None = None,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit: torch.Tensor | None = None,
        dynamic_gcn_bool: bool = True,
        dynamic_supports_len: int = 1,
        in_dim: int = 2,
        input_sequence_dim: int = 12,
        out_dim: int = 12,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.dynamic_gcn_bool = dynamic_gcn_bool
        self.num_nodes = num_nodes
        self.input_sequence_dim = input_sequence_dim
        self.dynamic_supports_len = dynamic_supports_len

        self.start_conv = nn.Conv2d(in_dim, residual_channels, (1, 1))
        self.supports = list(supports) if supports is not None else []
        self.supports_len = len(self.supports)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))
            else:
                m, p, n = torch.svd(aptinit)
                self.nodevec1 = nn.Parameter(
                    torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                )
                self.nodevec2 = nn.Parameter(
                    torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                )
            self.supports_len += 1

        if dynamic_gcn_bool:
            self.start_dynamic_graph_conv = nn.Conv2d(
                dynamic_supports_len, residual_channels, (1, 1)
            )
            self.dynamic_graph_filter_convs = nn.ModuleList()
            self.dynamic_graph_gate_convs = nn.ModuleList()
            self.dynamic_gconv = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, (1, 1))
                )
                if (i + 1) * (b + 1) - 1 < (blocks * layers - 1):
                    self.bn.append(nn.BatchNorm2d(residual_channels))
                    if gcn_bool:
                        self.gconv.append(
                            gcn(
                                dilation_channels,
                                residual_channels,
                                dropout,
                                support_len=self.supports_len,
                            )
                        )
                    else:
                        self.residual_convs.append(
                            nn.Conv2d(
                                dilation_channels,
                                residual_channels,
                                (1, 1),
                            )
                        )
                    if dynamic_gcn_bool:
                        self.dynamic_graph_filter_convs.append(
                            nn.Conv2d(
                                residual_channels,
                                dilation_channels,
                                (1, kernel_size),
                                dilation=new_dilation,
                            )
                        )
                        self.dynamic_graph_gate_convs.append(
                            nn.Conv2d(
                                residual_channels,
                                dilation_channels,
                                (1, kernel_size),
                                dilation=new_dilation,
                            )
                        )
                        self.dynamic_gconv.append(
                            dynamic_gcn(
                                dilation_channels,
                                residual_channels,
                                dropout,
                                dynamic_supports_len=dynamic_supports_len,
                            )
                        )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field
        end1_kernel = max(1, input_sequence_dim + 2 - receptive_field)
        self.end_conv_1 = nn.Conv2d(
            skip_channels, end_channels, (1, end1_kernel), bias=True
        )
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(
        self,
        x: torch.Tensor,
        graph_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x:           (B, in_dim, N, T)
        # graph_input: (B, dynamic_supports_len, N*N, T) when
        #              dynamic_gcn_bool=True; None otherwise.
        x = F.pad(x, (1, 0, 0, 0))
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        g_x: torch.Tensor | None = None
        if self.dynamic_gcn_bool:
            if graph_input is None:
                raise ValueError(
                    "graph_input must be provided when dynamic_gcn_bool=True"
                )
            g_x = F.pad(graph_input, (1, 0, 0, 0))
            if g_x.size(3) < self.receptive_field:
                g_x = F.pad(
                    g_x, (self.receptive_field - g_x.size(3), 0, 0, 0)
                )
            g_x = self.start_dynamic_graph_conv(g_x)

        x = self.start_conv(x)
        skip: torch.Tensor | int = 0

        new_supports: list[torch.Tensor] | None = None
        adp: torch.Tensor | None = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1
            )
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            f = torch.tanh(self.filter_convs[i](residual))
            g = torch.sigmoid(self.gate_convs[i](residual))
            x = f * g

            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = skip[:, :, :, -s.size(3):]
            else:
                skip = 0
            skip = s + skip

            if i < (self.blocks * self.layers - 1):
                if self.dynamic_gcn_bool:
                    g_filter = torch.tanh(
                        self.dynamic_graph_filter_convs[i](g_x)
                    )
                    g_gate = torch.sigmoid(
                        self.dynamic_graph_gate_convs[i](g_x)
                    )
                    g_x = (g_filter + g_x[:, :, :, -g_filter.size(3):]) * g_gate
                    # Per-timestep softmax over destination nodes.
                    A_t = F.softmax(
                        g_x.view(
                            g_x.size(0),
                            g_x.size(1),
                            self.num_nodes,
                            self.num_nodes,
                            g_x.size(3),
                        ),
                        dim=2,
                    )
                    d_g_x = self.dynamic_gconv[i](x, [A_t])

                if self.gcn_bool:
                    if new_supports is not None and len(new_supports) > 0:
                        x = self.gconv[i](x, new_supports)
                    elif len(self.supports) > 0:
                        x = self.gconv[i](x, self.supports)
                    else:
                        # gcn_bool but no supports and no adaptive adj —
                        # behaves as a 1x1 mixing conv (gcn with 0 supports).
                        x = self.gconv[i](x, [])
                else:
                    x = self.residual_convs[i](x)

                if self.dynamic_gcn_bool:
                    x = x + d_g_x

                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
