"""Per-model default hyperparameter search spaces.

Each model has knobs that are *hugely* influential and that a generic
"lr + one reg knob + one capacity knob" search would silently leave at
a poor fixed value (e.g. ASTGCN's Chebyshev order ``K``, GWN's
``addaptadj`` toggle, MTGODE's ODE step, transformer depth for
STAEFormer).  This module provides one factory per model that returns
the model's default search space — covering the shared training axis
(``lr``), one regularization axis, one capacity axis, plus 1-2
model-specific high-impact knobs.

Each factory returns a fresh ``dict[str, Sampler]`` and accepts
``**overrides`` so dataset-specific examples can replace a single
dimension (typically to cap a capacity axis for memory) without having
to rewrite the rest of the space::

    from gnn_benchmark.tuning.spaces import gwn_search_space
    from gnn_benchmark.tuning import Categorical

    # Default GWN search space.
    space = gwn_search_space()

    # Same space, but cap nhid for a large graph.
    space = gwn_search_space(nhid=Categorical([16, 32]))

    # Drop a dimension entirely.
    space = gwn_search_space()
    del space["blocks"]

The tuner validates every key against the relevant ``*Config``
dataclass (see ``tuning/tuner.py``) so typos and unknown fields fail
fast.
"""

from __future__ import annotations

from typing import Any

from gnn_benchmark.tuning.search_space import (
    Categorical,
    LogUniform,
    Sampler,
)

__all__ = [
    "DEFAULT_LR_LOW",
    "DEFAULT_LR_HIGH",
    "gwn_search_space",
    "mlp_multivariate_search_space",
    "mtgnn_search_space",
    "astgcn_search_space",
    "staeformer_search_space",
    "mtgode_search_space",
]


# Common LR range, log-uniform.  Spans roughly the published defaults
# across the wrapped models (1e-3 for GWN/MTGNN/STAEFormer/ASTGCN/MTGODE,
# side.  Override per call via ``lr_low``/``lr_high``.
DEFAULT_LR_LOW = 1e-4
DEFAULT_LR_HIGH = 5e-3


def _build(
    space: dict[str, Sampler], overrides: dict[str, Any],
) -> dict[str, Sampler]:
    space.update(overrides)
    return space


def mlp_multivariate_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """MLPMultivariate: lr + dropout + hidden_size + num_layers.

    ``hidden_size`` is the single most impactful capacity knob — larger
    values improve expressiveness but grow the parameter count quadratically
    with the node count.  ``num_layers`` controls depth and rarely needs to
    exceed 3 for this architecture.
    """
    return _build(
        {
            "lr":          LogUniform(lr_low, lr_high),
            "dropout":     Categorical([0.0, 0.1, 0.2, 0.3]),
            "hidden_size": Categorical([256, 512, 1024]),
            "num_layers":  Categorical([1, 2, 3]),
        },
        overrides,
    )


def gwn_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """GWN: lr + dropout + nhid + addaptadj + blocks.

    ``addaptadj`` toggles GWN's adaptive adjacency on top of the
    provided supports — flipping it changes whether the model uses
    the provided graph alone or both the provided graph and a learned
    one.  ``blocks`` controls the depth of the residual TCN/GCN stack.
    """
    return _build(
        {
            "lr":        LogUniform(lr_low, lr_high),
            "dropout":   Categorical([0.0, 0.1, 0.3, 0.5]),
            "nhid":      Categorical([16, 32, 64]),
            "addaptadj": Categorical([True, False]),
            "blocks":    Categorical([2, 3, 4]),
        },
        overrides,
    )


def mtgnn_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """MTGNN: lr + dropout + conv_channels + buildA_true + propalpha.

    ``buildA_true`` picks between a learned graph constructor and the
    provided adjacency (architecturally exclusive in MTGNN).
    ``propalpha`` controls graph-propagation strength and is well-known
    to dominate val loss when wrong.
    """
    return _build(
        {
            "lr":            LogUniform(lr_low, lr_high),
            "dropout":       Categorical([0.0, 0.1, 0.3, 0.5]),
            "conv_channels": Categorical([16, 32, 64]),
            "buildA_true":   Categorical([True, False]),
            "propalpha":     Categorical([0.05, 0.1, 0.2]),
        },
        overrides,
    )




def astgcn_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """ASTGCN: lr + weight_decay + nb_chev_filter + K + nb_block.

    ``K`` is ASTGCN's Chebyshev order — the most influential ASTGCN
    knob.  ``nb_block`` controls depth.
    """
    return _build(
        {
            "lr":             LogUniform(lr_low, lr_high),
            "weight_decay":   Categorical([0.0, 1e-5, 1e-4, 1e-3]),
            "nb_chev_filter": Categorical([32, 64, 128]),
            "K":              Categorical([2, 3, 4]),
            "nb_block":       Categorical([1, 2, 3]),
        },
        overrides,
    )


def staeformer_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """STAEFormer: lr + dropout + adaptive_embedding_dim + num_layers + feed_forward_dim.

    Transformer val loss is highly sensitive to depth (``num_layers``)
    and FFN width (``feed_forward_dim``).  ``adaptive_embedding_dim``
    drives the model_dim alongside the fixed ``input_embedding_dim``;
    callers needing a particular ``num_heads`` divisibility should
    override either of them.
    """
    return _build(
        {
            "lr":                     LogUniform(lr_low, lr_high),
            "dropout":                Categorical([0.0, 0.1, 0.2, 0.3]),
            "adaptive_embedding_dim": Categorical([16, 40, 80]),
            "num_layers":             Categorical([2, 3, 4]),
            "feed_forward_dim":       Categorical([128, 256, 512]),
        },
        overrides,
    )


def mtgode_search_space(
    lr_low: float = DEFAULT_LR_LOW,
    lr_high: float = DEFAULT_LR_HIGH,
    **overrides: Sampler,
) -> dict[str, Sampler]:
    """MTGODE: lr + dropout + conv_channels + buildA_true + step_1.

    Like MTGNN, ``buildA_true`` selects between learned and provided
    graphs.  ``step_1`` is the Continuous Temporal Aggregation ODE
    step size — coarser steps train faster but lose accuracy, finer
    steps converge to better val loss but cost more compute.
    """
    return _build(
        {
            "lr":            LogUniform(lr_low, lr_high),
            "dropout":       Categorical([0.0, 0.1, 0.3, 0.5]),
            "conv_channels": Categorical([16, 32, 64]),
            "buildA_true":   Categorical([True, False]),
            "step_1":        Categorical([0.125, 0.25, 0.5]),
        },
        overrides,
    )
