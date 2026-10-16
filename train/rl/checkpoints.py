"""Atomic RL checkpoints without Hydra, W&B, or launcher dependencies."""

import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def atomic_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
