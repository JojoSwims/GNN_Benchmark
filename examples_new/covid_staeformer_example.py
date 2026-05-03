#!/usr/bin/env python3
"""STAEFormer on NYC COVID — fair-protocol hyperparameter tuning.

See ``examples_new/_shared.py`` for the protocol; the per-model search
space is defined in :mod:`gnn_benchmark.tuning.spaces`.  Spatial
self-attention scales as N²·heads·layers, so on the large county graph we
pin ``num_layers=1`` and ``feed_forward_dim=128`` in the base config (and
drop both from the search), and cap ``adaptive_embedding_dim`` at 16
(``model_dim`` ∈ {32, 40}).  ``num_heads=4`` divides both.

Memory: spatial attention over N=3212 nodes is the OOM driver, so we
also enable bf16 autocast, activation checkpointing on the attention
stacks, and a smaller eval batch size — all wrapper-level knobs that
don't change the architecture. ``PYTORCH_CUDA_ALLOC_CONF`` is set
before importing torch (via gnn_benchmark) to reduce fragmentation
across the 18 trials.
"""

import os

# Must be set before any torch import (the CUDA caching allocator reads
# this env var when first initialized). expandable_segments cuts the
# fragmentation that otherwise turns tight runs into OOMs late in a
# tuning sweep.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

from gnn_benchmark.models import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.tuning import Categorical, staeformer_search_space

from _shared import LR_HIGH, LR_LOW, apply_schedule, run_example

DATASET = "nyc-covid"
base_config = apply_schedule(
    STAEFormerConfig(),
    DATASET,
    num_heads=4,
    num_layers=1,
    feed_forward_dim=128,
    use_amp=True,
    amp_dtype="bfloat16",
    use_checkpoint=True,
    eval_batch_size=1,
)

search_space = staeformer_search_space(
    lr_low=LR_LOW,
    lr_high=LR_HIGH,
    adaptive_embedding_dim=Categorical([8, 16]),
)
# Pinned in base_config for the N=3212 graph.
del search_space["num_layers"]
del search_space["feed_forward_dim"]

run_example(
    model_factory=lambda: STAEFormerModel(),
    base_config=base_config,
    dataset_key=DATASET,
    search_space=search_space,
)
