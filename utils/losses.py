"""Training loss helpers shared across model wrappers.

Every benchmark model wrapper trains with Huber loss, but targets may
contain NaN to mark missing values (see :class:`BenchmarkModel`'s tensor
contract).  Replacing those NaNs with 0 — as a naive ``nan_to_num`` does
— silently treats missingness as a true-zero target, which biases the
model toward predicting zero everywhere a sensor is offline.  Evaluation
metrics already mask NaN positions, so the asymmetry distorts both the
training signal and the train↔val↔test loss comparison.

:func:`masked_huber_loss` instead computes Huber loss only over positions
where the target is finite, matching how metrics are scored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Mean Huber loss over positions where ``target`` is finite.

    NaN entries in ``target`` (the benchmark's missing-value sentinel)
    contribute neither to the sum nor to the denominator, so the model
    is never penalised for them.  When every position is missing the
    function returns a zero tensor that still carries gradient through
    ``pred`` so an autograd graph stays connected.
    """
    mask = ~torch.isnan(target)
    target_safe = torch.where(mask, target, torch.zeros_like(target))
    elementwise = F.huber_loss(pred, target_safe, reduction="none", delta=delta)
    elementwise = elementwise * mask.to(elementwise.dtype)
    n_valid = mask.sum()
    if n_valid == 0:
        return pred.sum() * 0.0
    return elementwise.sum() / n_valid
