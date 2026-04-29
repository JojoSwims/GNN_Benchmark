"""Diffusion Convolutional GRU cell used by the GTS encoder/decoder."""

from __future__ import annotations

import torch


class LayerParams:
    """Lazy container for per-shape weights/biases registered on the cell."""

    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict: dict[tuple, torch.nn.Parameter] = {}
        self._biases_dict: dict[int, torch.nn.Parameter] = {}
        self._type = layer_type

    def get_weights(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Parameter:
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(
                torch.empty(*shape, device=device, dtype=dtype)
            )
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                f"{self._type}_weight_{shape}", nn_param
            )
        return self._params_dict[shape]

    def get_biases(
        self,
        length: int,
        bias_start: float = 0.0,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Parameter:
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(
                torch.empty(length, device=device, dtype=dtype)
            )
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                f"{self._type}_biases_{length}", biases
            )
        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    """Diffusion Convolutional GRU cell.

    Operates on a learned dense adjacency matrix ``adj`` of shape
    ``(num_nodes, num_nodes)`` passed at forward time.
    """

    def __init__(
        self,
        num_units: int,
        max_diffusion_step: int,
        num_nodes: int,
        nonlinearity: str = "tanh",
        use_gc_for_ru: bool = True,
    ) -> None:
        super().__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        self._fc_params = LayerParams(self, "fc")
        self._gconv_params = LayerParams(self, "gconv")

    def _calculate_random_walk_matrix(self, adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        adj = adj + eye
        d = torch.sum(adj, dim=1)
        d_inv = 1.0 / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros_like(d_inv), d_inv)
        d_mat_inv = torch.diag(d_inv)
        return torch.mm(d_mat_inv, adj)

    def forward(
        self, inputs: torch.Tensor, hx: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """GRU step with graph convolution.

        Args:
            inputs: ``(B, num_nodes * input_dim)``.
            hx:     ``(B, num_nodes * rnn_units)``.
            adj:    ``(num_nodes, num_nodes)`` learned adjacency.

        Returns:
            New hidden state ``(B, num_nodes * rnn_units)``.
        """
        adj_mx = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self._num_units
        fn = self._gconv if self._use_gc_for_ru else self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        return u * hx + (1.0 - u) * c

    @staticmethod
    def _concat(x: torch.Tensor, x_: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x_.unsqueeze(0)], dim=0)

    def _fc(
        self,
        inputs: torch.Tensor,
        adj_mx: torch.Tensor,
        state: torch.Tensor,
        output_size: int,
        bias_start: float = 0.0,
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights(
            (input_size, output_size),
            device=inputs_and_state.device,
            dtype=inputs_and_state.dtype,
        )
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(
            output_size,
            bias_start,
            device=inputs_and_state.device,
            dtype=inputs_and_state.dtype,
        )
        return value + biases

    def _gconv(
        self,
        inputs: torch.Tensor,
        adj_mx: torch.Tensor,
        state: torch.Tensor,
        output_size: int,
        bias_start: float = 0.0,
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x0 = inputs_and_state.permute(1, 2, 0)  # (N, input_size, B)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step > 0:
            x1 = torch.mm(adj_mx, x0)
            x = self._concat(x, x1)
            for _ in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        num_matrices = self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)  # (B, N, input_size, order)
        x = torch.reshape(
            x,
            shape=[batch_size * self._num_nodes, input_size * num_matrices],
        )

        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.matmul(x, weights)
        biases = self._gconv_params.get_biases(
            output_size,
            bias_start,
            device=x.device,
            dtype=x.dtype,
        )
        x = x + biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
