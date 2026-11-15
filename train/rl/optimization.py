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


def step_with_update_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Measure actual parameter displacement separately from clipped gradients.

    Snapshot only parameters with gradients; AdamW skips the others. This adds
    temporary copies of active trainable weights, never the frozen visual path.
    No loss scaling, optimizer settings, or optimizer state is changed.
    """
    parameters = [
        p for group in optimizer.param_groups for p in group["params"]
        if p.grad is not None
    ]
    before = [p.detach().clone() for p in parameters]
    optimizer.step()
    if not before:
        return {"parameter_update_norm": 0.0, "relative_parameter_update_norm": 0.0}
    with torch.no_grad():
        update_norm = torch.stack([
            torch.linalg.vector_norm((p.detach() - old).float())
            for p, old in zip(parameters, before, strict=True)
        ]).norm().item()
        weight_norm = torch.stack([
            torch.linalg.vector_norm(old.float()) for old in before
        ]).norm().item()
    return {
        "parameter_update_norm": update_norm,
        "relative_parameter_update_norm": update_norm / max(weight_norm, 1e-12),
    }
