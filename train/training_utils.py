"""Shared runtime utilities for robust, portable training jobs.

The checkpoint format deliberately separates the always-overwritten, resumable
``latest`` checkpoint from small, rotating weight-only snapshots.  Writes are
atomic, so an interrupted write cannot destroy the previous valid checkpoint.
"""

from __future__ import annotations

import os
import random
import re
import signal
import tempfile
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from omegaconf import OmegaConf


CHECKPOINT_FORMAT_VERSION = 2


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU, or validate an explicit choice."""
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    if requested != "cpu" and requested != "mps" and not requested.startswith("cuda"):
        raise ValueError("device must be 'auto', 'cpu', 'mps', or a CUDA device such as 'cuda:0'")
    return torch.device(requested)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNGs needed to continue a stochastic training run."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    """Restore RNG state while tolerating checkpoints made on another device."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        saved = list(state["cuda"])
        current_count = torch.cuda.device_count()
        if len(saved) == current_count:
            torch.cuda.set_rng_state_all(saved)
        else:
            warnings.warn(
                "CUDA device count differs from the checkpoint; CUDA RNG state was not restored.",
                RuntimeWarning,
            )


def torch_load(path: os.PathLike[str] | str, map_location: Any = "cpu") -> Any:
    """Load a trusted local checkpoint across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        return torch.load(path, map_location=map_location)


def model_weights(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract weights from either this project's full format or a legacy file."""
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        return checkpoint["model"]
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint is neither a state_dict nor a resumable training checkpoint")
    return checkpoint


class CheckpointManager:
    """Atomically save a full latest checkpoint plus rotating weight snapshots."""

    def __init__(self, latest_path: os.PathLike[str] | str, keep_last: int = 3):
        self.latest_path = Path(latest_path).expanduser()
        self.keep_last = max(0, int(keep_last))

    @property
    def artifact_stem(self) -> str:
        stem = self.latest_path.stem
        return stem[: -len(".latest")] if stem.endswith(".latest") else stem

    @property
    def best_path(self) -> Path:
        return self.latest_path.with_name(f"{self.artifact_stem}.best{self.latest_path.suffix}")

    def resolve_resume_path(self, configured: Optional[str]) -> Optional[Path]:
        if configured is None or str(configured).strip().lower() in {"", "none", "null"}:
            return None
        path = (
            self.latest_path
            if str(configured).strip().lower() == "auto"
            else Path(configured).expanduser()
        )
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path

    def load(self, path: os.PathLike[str] | str) -> Any:
        return torch_load(path, map_location="cpu")

    def save_latest(
        self,
        state: Mapping[str, Any],
        weights: Mapping[str, torch.Tensor],
        *,
        stage_name: str,
        epoch: int,
        save_snapshot: bool = True,
    ) -> Path:
        self._atomic_save(dict(state), self.latest_path)
        if save_snapshot and self.keep_last > 0:
            try:
                self._atomic_save(dict(weights), self._snapshot_path(stage_name, epoch))
                self._prune_snapshots()
            except Exception as exc:  # A snapshot is optional once latest is durable.
                warnings.warn(
                    f"The full checkpoint was saved, but the weight snapshot failed: {exc}",
                    RuntimeWarning,
                )
        return self.latest_path

    def save_best(self, weights: Mapping[str, torch.Tensor]) -> Path:
        self._atomic_save(dict(weights), self.best_path)
        return self.best_path

    def load_best_weights(self) -> Mapping[str, torch.Tensor]:
        if not self.best_path.is_file():
            raise FileNotFoundError(f"Best-model checkpoint does not exist: {self.best_path}")
        return model_weights(self.load(self.best_path))

    def _snapshot_path(self, stage_name: str, epoch: int) -> Path:
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage_name).strip("-") or "stage"
        return self.latest_path.with_name(
            f"{self.artifact_stem}.{safe_stage}.epoch-{epoch:04d}{self.latest_path.suffix}"
        )

    def _snapshot_paths(self) -> list[Path]:
        pattern = f"{self.artifact_stem}.*.epoch-[0-9][0-9][0-9][0-9]{self.latest_path.suffix}"
        return sorted(
            self.latest_path.parent.glob(pattern),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )

    def _prune_snapshots(self) -> None:
        snapshots = self._snapshot_paths()
        for old_path in snapshots[: max(0, len(snapshots) - self.keep_last)]:
            old_path.unlink()

    @staticmethod
    def _atomic_save(value: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


class GracefulStop:
    """Turn termination signals into a flag checked at a safe batch boundary."""

    def __init__(self):
        self.requested = False
        self.signal_name: Optional[str] = None
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        signal_names = ["SIGINT", "SIGTERM"]
        if hasattr(signal, "SIGUSR1"):
            signal_names.append("SIGUSR1")
        for name in signal_names:
            signum = getattr(signal, name)
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        print(f"\n{self.signal_name} received; checkpointing at the next optimizer boundary...")


class TrainingInterrupted(Exception):
    """Internal control flow used after a graceful stop request."""


class SafeWandbRun:
    """Buffer and retry failed W&B writes without terminating local training."""

    def __init__(self, run: Any, max_pending_logs: int = 1000):
        self.run = run
        self._warned = False
        self._pending: deque[dict[str, Any]] = deque(maxlen=max_pending_logs)

    @property
    def id(self) -> Optional[str]:
        return getattr(self.run, "id", None)

    def log(self, values: Mapping[str, Any]) -> None:
        self._pending.append(dict(values))
        self._flush_pending()

    def _flush_pending(self) -> None:
        """Retry buffered entries in order; the W&B SDK also retries transport failures."""
        try:
            while self._pending:
                self.run.log(self._pending[0])
                self._pending.popleft()
        except Exception as exc:  # Logging must never invalidate model progress.
            if not self._warned:
                warnings.warn(
                    f"W&B logging failed; metrics are buffered and training will continue: {exc}",
                    RuntimeWarning,
                )
                self._warned = True

    def finish(self) -> None:
        self._flush_pending()
        try:
            self.run.finish()
        except Exception as exc:
            warnings.warn(
                f"W&B shutdown failed; local checkpoints are unaffected: {exc}",
                RuntimeWarning,
            )


def init_wandb(cfg: Any, *, resume_id: Optional[str] = None) -> Optional[SafeWandbRun]:
    """Initialize W&B with bounded retries and an optional offline fallback."""
    if not bool(getattr(cfg, "log", False)):
        return None

    import wandb

    trainer_cfg = getattr(cfg, "trainer", {})
    retries = max(1, int(getattr(trainer_cfg, "wandb_init_retries", 2)))
    retry_delay = max(0.0, float(getattr(trainer_cfg, "wandb_retry_delay_seconds", 5.0)))
    init_kwargs = {
        "project": cfg.wandb_project,
        "name": cfg.experiment_name,
        "config": OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
    }
    if resume_id:
        init_kwargs.update(id=resume_id, resume="allow")

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return SafeWandbRun(wandb.init(**init_kwargs))
        except Exception as exc:
            last_error = exc
            warnings.warn(
                f"W&B initialization attempt {attempt + 1}/{retries} failed: {exc}",
                RuntimeWarning,
            )
            if attempt + 1 < retries:
                time.sleep(retry_delay * (2**attempt))

    if bool(getattr(trainer_cfg, "wandb_offline_fallback", True)):
        warnings.warn(
            "Falling back to offline W&B logging. Upload later with `wandb sync <run-directory>`.",
            RuntimeWarning,
        )
        offline_kwargs = dict(init_kwargs)
        offline_kwargs.pop("resume", None)
        offline_kwargs["mode"] = "offline"
        try:
            return SafeWandbRun(wandb.init(**offline_kwargs))
        except Exception as exc:
            last_error = exc

    warnings.warn(f"W&B disabled because initialization failed: {last_error}", RuntimeWarning)
    return None
