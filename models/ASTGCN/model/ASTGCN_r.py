"""ASTGCN (recent-component) model.

Attention-Based Spatial-Temporal Graph Convolutional Network for traffic
forecasting (Guo et al., AAAI 2019).  Only the recent (``_r``) component
is kept here — the multi-component variant is unused by the benchmark.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def scaled_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute the scaled Laplacian ``2L/lambda_max - I`` for adjacency W."""
    assert W.shape[0] == W.shape[1]
    # Symmetrize defensively — floating-point asymmetry would make L non-Hermitian.
    W_sym = np.maximum(W, W.T)
    D = np.diag(np.sum(W_sym, axis=1))
    L = D - W_sym
    # L is real symmetric, so eigvalsh is exact and numerically robust.
    # ARPACK's eigs(..., which="LR") fails with "Starting vector is zero" on
    # small or ill-conditioned Laplacians (e.g. Beijing Air, N=36).
    lambda_max = float(np.linalg.eigvalsh(L)[-1])
    if lambda_max <= 0:
        lambda_max = 2.0
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomials(L_tilde: np.ndarray, K: int) -> list[np.ndarray]:
    """Chebyshev polynomials ``T_0 ... T_{K-1}`` evaluated at ``L_tilde``."""
    N = L_tilde.shape[0]
    polys = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        polys.append(2 * L_tilde * polys[i - 1] - polys[i - 2])
    return polys


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Spatial attention scores, shape ``(B, N, N)``."""

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_of_timesteps))
        self.W2 = nn.Parameter(torch.empty(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (B, N, T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)        # (B, T, N)
        product = torch.matmul(lhs, rhs)                         # (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(S, dim=1)


class TemporalAttention(nn.Module):
    """Temporal attention scores, shape ``(B, T, T)``."""

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(num_of_vertices))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in, T) -> (B, T, F_in, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)                  # (B, N, T)
        product = torch.matmul(lhs, rhs)                # (B, T, T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=1)


# ---------------------------------------------------------------------------
# Spatial graph convolution with attention
# ---------------------------------------------------------------------------

class ChebConvWithSAt(nn.Module):
    """K-order Chebyshev graph convolution gated by spatial attention."""

    def __init__(self, K: int, cheb_polys: list[torch.Tensor], in_channels: int, out_channels: int):
        super().__init__()
        self.K = K
        # Register polynomials as buffers so they move with .to(device).
        for k, T_k in enumerate(cheb_polys):
            self.register_buffer(f"T_{k}", T_k, persistent=False)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)]
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in, T)
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []
        for t in range(num_of_timesteps):
            graph_signal = x[:, :, :, t]  # (B, N, F_in)
            output = torch.zeros(
                batch_size, num_of_vertices, self.out_channels,
                device=x.device, dtype=x.dtype,
            )
            for k in range(self.K):
                T_k = getattr(self, f"T_{k}")
                T_k_with_at = T_k.mul(spatial_attention)            # (B, N, N)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, F_in)
                output = output + rhs.matmul(self.Theta[k])         # (B, N, F_out)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))                   # (B, N, F_out, T)


# ---------------------------------------------------------------------------
# ASTGCN block + full submodule
# ---------------------------------------------------------------------------

class ASTGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        cheb_polys: list[torch.Tensor],
        num_of_vertices: int,
        num_of_timesteps: int,
    ):
        super().__init__()
        self.TAt = TemporalAttention(in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = SpatialAttention(in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = ChebConvWithSAt(K, cheb_polys, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter,
            kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter,
            kernel_size=(1, 1), stride=(1, time_strides),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in, T)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        temporal_at = self.TAt(x)
        x_TAt = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps), temporal_at,
        ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        spatial_at = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)             # (B, N, F, T)

        time_conv_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B, F, N, T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))           # (B, F, N, T)

        # Apply LayerNorm over the channel dimension.
        x_residual = self.ln(
            F.relu(x_residual + time_conv_out).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)
        return x_residual


class ASTGCN(nn.Module):
    """ASTGCN recent-component network.

    Input shape:  ``(B, N, F_in, T_in)``
    Output shape: ``(B, N, T_out, F_out)``
    """

    def __init__(
        self,
        adj_mx: np.ndarray | None,
        nb_block: int,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_for_predict: int,
        len_input: int,
        num_of_vertices: int,
        output_dim: int = 1,
        no_graph: bool = False,
    ):
        super().__init__()
        if no_graph:
            # No-graph ablation: replace every Chebyshev polynomial with the
            # identity matrix. After `T_k.mul(spatial_attention)` the result is
            # a per-batch diagonal, so the cheb conv becomes a per-node linear
            # rescaled by each node's own attention score — no cross-node
            # mixing. SpatialAttention parameters and Theta_k weights stay live.
            cheb_polys = [
                torch.eye(num_of_vertices, dtype=torch.float32) for _ in range(K)
            ]
        else:
            if adj_mx is None:
                raise ValueError("adj_mx is required when no_graph=False")
            L_tilde = scaled_laplacian(adj_mx)
            cheb_polys = [
                torch.from_numpy(p).to(torch.float32)
                for p in cheb_polynomials(L_tilde, K)
            ]

        self.output_dim = output_dim
        self.num_for_predict = num_for_predict

        blocks: list[nn.Module] = [
            ASTGCNBlock(
                in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                cheb_polys, num_of_vertices, len_input,
            )
        ]
        for _ in range(nb_block - 1):
            blocks.append(
                ASTGCNBlock(
                    nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                    cheb_polys, num_of_vertices, len_input // time_strides,
                )
            )
        self.BlockList = nn.ModuleList(blocks)

        self.final_conv = nn.Conv2d(
            len_input // time_strides,
            num_for_predict * output_dim,
            kernel_size=(1, nb_time_filter),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in, T)
        for block in self.BlockList:
            x = block(x)

        # x: (B, N, F, T) -> (B, T, N, F) -> conv -> (B, T_out*D_out, N, 1)
        out = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        # (B, T_out*D_out, N) -> (B, T_out, D_out, N) -> (B, N, T_out, D_out)
        b, _, n = out.shape
        out = out.reshape(b, self.num_for_predict, self.output_dim, n)
        return out.permute(0, 3, 1, 2)
