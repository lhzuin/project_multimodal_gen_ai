"""Optimizer diagnostics shared without changing either algorithm's objective."""

from collections.abc import Iterable

import torch


def clip_gradients(
    parameters: Iterable[torch.Tensor], max_norm: float
) -> dict[str, float]:
    """Clip once after accumulation and expose the actual rescaling coefficient.

    PyTorch returns the PRE-clip norm. Preserve ``grad_norm`` for older analysis
    scripts, but label its meaning explicitly in the new metrics. The post-clip
    norm is estimated from the common coefficient, avoiding another device sync.
    """
    norm = float(
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            error_if_nonfinite=True,
        ).item()
    )
    scale = min(1.0, max_norm / (norm + 1e-6))
    return {
        "grad_norm": norm,
        "grad_norm_pre_clip": norm,
        "grad_norm_post_clip_estimate": norm * scale,
        "grad_clip_scale": scale,
        "grad_clipped": float(scale < 1.0),
    }
