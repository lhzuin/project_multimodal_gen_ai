"""Launcher-owned metric sinks; training classes remain independent of W&B.

JSONL is append-only and flushed after each record. Run metadata lives beside
it in ``<metrics path>.sessions.jsonl``. W&B is imported only when enabled;
remote logging failures never disable the local journal or stop optimization.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import uuid
import warnings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricsLogger:
    """Context-managed stdout/JSONL/optional W&B callback for ``trainer.fit``.

    ``run_id`` comes from the resume checkpoint. A new process gets a distinct
    session ID, while all resumed sessions retain the logical run ID. JSONL
    records can therefore be reconciled by (run_id, completed_batches).
    """

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        run_config: Mapping[str, Any],
        algorithm: str,
        run_id: str | None = None,
        resumed: bool = False,
    ) -> None:
        unknown = set(settings) - {"stdout", "jsonl_path", "fsync", "wandb"}
        if unknown:
            raise ValueError(f"unknown logging settings: {sorted(unknown)}")
        self.settings = dict(settings)
        self.wandb_settings = dict(settings.get("wandb", {}))
        unknown = set(self.wandb_settings) - {
            "enabled",
            "project",
            "entity",
            "name",
            "id",
            "mode",
            "tags",
            "group",
            "dir",
            "offline_fallback",
            "init_timeout_s",
        }
        if unknown:
            raise ValueError(f"unknown W&B settings: {sorted(unknown)}")
        for values, keys in (
            (self.settings, ("stdout", "fsync")),
            (self.wandb_settings, ("enabled", "offline_fallback")),
        ):
            for key in keys:
                if key in values and not isinstance(values[key], bool):
                    raise ValueError(f"logging {key} must be a boolean")
        if self.wandb_settings.get("mode", "online") not in (
            "online",
            "offline",
            "disabled",
        ):
            raise ValueError("W&B mode must be online, offline, or disabled")
        timeout = self.wandb_settings.get("init_timeout_s", 30)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("W&B init_timeout_s must be finite and positive")
        requested_id = self.wandb_settings.get("id")
        if run_id and requested_id and requested_id != run_id:
            raise ValueError("W&B ID differs from the resumed checkpoint's logging ID")
        self.run_id = run_id or requested_id or uuid.uuid4().hex
        self.session_id = uuid.uuid4().hex
        self.run_config = dict(run_config)
        self.algorithm = algorithm
        self.resumed = resumed
        self._handle = None
        self._run = None
        self._started = time.monotonic()

    def _write(self, handle: Any, record: Mapping[str, Any]) -> None:
        handle.write(json.dumps(dict(record), allow_nan=False) + "\n")
        handle.flush()
        if self.settings.get("fsync", False):
            os.fsync(handle.fileno())

    def __enter__(self) -> MetricsLogger:
        path = self.settings.get("jsonl_path")
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with Path(str(path) + ".sessions.jsonl").open(
                "a", encoding="utf-8"
            ) as meta:
                self._write(
                    meta,
                    {
                        "run_id": self.run_id,
                        "session_id": self.session_id,
                        "timestamp_utc": utc_now(),
                        "resumed": self.resumed,
                        "algorithm": self.algorithm,
                        "config": self.run_config,
                    },
                )
            self._handle = path.open("a", encoding="utf-8")
        try:
            self._start_wandb()
        except BaseException:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            raise
        return self

    def _start_wandb(self) -> None:
        cfg = self.wandb_settings
        if not cfg.get("enabled", False):
            return
        try:
            wandb = importlib.import_module("wandb")
            kwargs = {
                key: cfg[key]
                for key in ("project", "entity", "name", "mode", "tags", "group", "dir")
                if key in cfg
            }
            if "dir" in kwargs:
                Path(kwargs["dir"]).mkdir(parents=True, exist_ok=True)
            kwargs.update(
                id=self.run_id,
                config=self.run_config,
                settings=wandb.Settings(init_timeout=cfg.get("init_timeout_s", 30)),
            )
            # W&B cannot resume offline runs. The local journal still shares
            # the run ID; each offline process has its own syncable directory.
            effective_mode = cfg.get("mode", os.environ.get("WANDB_MODE", "online"))
            if self.resumed and effective_mode == "online":
                kwargs["resume"] = "allow"
            try:
                self._run = wandb.init(**kwargs)
            except Exception as exc:
                if effective_mode != "online" or not cfg.get("offline_fallback", True):
                    raise
                warnings.warn(f"W&B initialization failed; trying offline: {exc}")
                kwargs.pop("resume", None)
                kwargs["mode"] = "offline"
                self._run = wandb.init(**kwargs)
            self._run.define_metric("completed_batches")
            self._run.define_metric("*", step_metric="completed_batches")
        except Exception as exc:
            warnings.warn(f"W&B unavailable; stdout/JSONL logging continues: {exc}")
            self._finish_wandb(1)

    def __call__(self, metrics: Mapping[str, float]) -> None:
        record = dict(metrics)
        record.update(
            timestamp_utc=utc_now(),
            run_id=self.run_id,
            session_id=self.session_id,
            session_elapsed_seconds=time.monotonic() - self._started,
        )
        if self._handle is not None:
            self._write(self._handle, record)
        if self.settings.get("stdout", True):
            print(json.dumps(record, allow_nan=False), flush=True)
        if self._run is not None:
            try:
                # Custom completed_batches axis supports checkpoint recovery
                # without relying on W&B's internal history step counter.
                self._run.log(dict(metrics))
            except Exception as exc:
                warnings.warn(f"W&B logging failed; local logging continues: {exc}")
                self._finish_wandb(1)

    def _finish_wandb(self, exit_code: int) -> None:
        run, self._run = self._run, None
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except Exception as exc:
                warnings.warn(f"W&B finalization failed: {exc}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._finish_wandb(0 if exc_type is None else 1)
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
