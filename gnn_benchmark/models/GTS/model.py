"""GTS (Graph for Time Series) model.

Encoder/decoder built on :class:`DCGRUCell` with a learnable graph
structure sampled from node-level features via Gumbel-softmax.

The forward pass takes:

* ``inputs``    - ``(seq_len, batch_size, num_nodes * input_dim)``
* ``node_feas`` - ``(T_feas, num_nodes)`` feature series used to build
  the latent graph
* ``temp`` / ``gumbel_soft`` - Gumbel-softmax controls
* ``labels``       - optional teacher forcing targets (decoder)
* ``batches_seen`` - optional global step used for curriculum learning
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from gnn_benchmark.models.GTS.cell import DCGRUCell


def _sample_gumbel(shape: torch.Size, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def _gumbel_softmax_sample(
    logits: torch.Tensor, temperature: float, eps: float = 1e-10
) -> torch.Tensor:
    y = logits + _sample_gumbel(logits.size(), logits.device, eps=eps)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float,
    hard: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution, optionally discretize."""
    y_soft = _gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if not hard:
        return y_soft
    shape = logits.size()
    _, k = y_soft.data.max(-1)
    y_hard = torch.zeros(*shape, device=logits.device)
    y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    return (y_hard - y_soft).detach() + y_soft


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs) -> None:
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs["rnn_units"])
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs) -> None:
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs["seq_len"])
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        adj: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size),
                device=inputs.device,
            )
        hidden_states: list[torch.Tensor] = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs) -> None:
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        adj: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states: list[torch.Tensor] = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class GTSModel(nn.Module, Seq2SeqAttrs):
    """Graph-for-Time-Series encoder/decoder with a learned latent graph."""

    def __init__(self, temperature: float, **model_kwargs) -> None:
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False)
        )
        self.temperature = temperature
        self.dim_fc = int(model_kwargs["dim_fc"])
        self.embedding_dim = int(model_kwargs.get("embedding_dim", 100))
        self.conv1 = nn.Conv1d(1, 8, 10, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 10, stride=1)
        self.hidden_drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)

        # Off-diagonal pair indices for every (sender, receiver) node pair.
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(
            self._encode_onehot(np.where(off_diag)[0]), dtype=np.float32
        )
        rel_send = np.array(
            self._encode_onehot(np.where(off_diag)[1]), dtype=np.float32
        )
        self.register_buffer("rel_rec", torch.FloatTensor(rel_rec))
        self.register_buffer("rel_send", torch.FloatTensor(rel_send))

    @staticmethod
    def _encode_onehot(labels: np.ndarray) -> np.ndarray:
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    def _compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def _encoder(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        encoder_hidden_state: torch.Tensor | None = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], adj, encoder_hidden_state
            )
        return encoder_hidden_state

    def _decoder(
        self,
        encoder_hidden_state: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor | None = None,
        batches_seen: int | None = None,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim),
            device=encoder_hidden_state.device,
        )
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs: list[torch.Tensor] = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, adj, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if (
                self.training
                and self.use_curriculum_learning
                and labels is not None
                and batches_seen is not None
            ):
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        return torch.stack(outputs)

    def _build_adj(
        self, node_feas: torch.Tensor, temp: float, hard: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a (N, N) latent adjacency from node features.

        Returns:
            adj:    ``(N, N)`` sampled adjacency (diagonal zeroed).
            logits: ``(N*N, 2)`` Gumbel-softmax logits, used for the
                    optional BCE regularization term.
        """
        x = node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(x, temperature=temp, hard=hard)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes, device=adj.device).bool()
        adj = adj.masked_fill(mask, 0.0)
        return adj, x

    def forward(
        self,
        inputs: torch.Tensor,
        node_feas: torch.Tensor,
        temp: float,
        gumbel_hard: bool = True,
        labels: torch.Tensor | None = None,
        batches_seen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder-decoder forward pass.

        Args:
            inputs:       ``(seq_len, batch_size, num_nodes * input_dim)``.
            node_feas:    ``(T_feas, num_nodes)`` series used to build the graph.
            temp:         Gumbel-softmax temperature.
            gumbel_hard:  Whether to draw a hard (one-hot) sample.
            labels:       Optional teacher-forcing labels
                          ``(horizon, batch_size, num_nodes * output_dim)``.
            batches_seen: Optional global step for curriculum learning.

        Returns:
            outputs:    ``(horizon, batch_size, num_nodes * output_dim)``.
            adj_logits: ``(N*N, 2)`` -- softmax probabilities of edge presence
                        (column 0) can be used for BCE regularization against
                        a reference adjacency.
        """
        adj, logits = self._build_adj(node_feas, temp=temp, hard=gumbel_hard)
        encoder_hidden_state = self._encoder(inputs, adj)
        outputs = self._decoder(
            encoder_hidden_state, adj, labels, batches_seen=batches_seen
        )
        return outputs, logits.softmax(-1)[:, 0].clone().reshape(
            self.num_nodes, -1
        )
