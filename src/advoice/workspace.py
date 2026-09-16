"""Shared writer exclusion and output-path checks for supported CLI commands."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


LOCK_TOKEN_ENV = "ADVOICE_WORKSPACE_LOCK_TOKEN"


def validate_output_paths(workspace: Path) -> None:
    workspace = workspace.resolve()
    for name in ("artifacts", "runs", "reports", "data/interim", "data/processed",
                 "executions", "experiment_latest.json"):
        target = workspace / name
        if not target.resolve().is_relative_to(workspace) or target.is_symlink():
            raise ValueError(f"Output path escapes the workspace or is a symlink: {target}")
        if target.is_dir():
            for directory, dirs, files in os.walk(target, followlinks=False):
                for child in dirs + files:
                    path = Path(directory) / child
                    if path.is_symlink():
                        raise ValueError(f"Symlinks are not allowed in managed outputs: {path}")


@contextmanager
def workspace_lock(workspace: Path):
    workspace = workspace.resolve()
    validate_output_paths(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    lock = workspace / ".experiment.lock"
    inherited = os.environ.get(LOCK_TOKEN_ENV)
    if inherited and lock.is_file() and not lock.is_symlink():
        owner = json.loads(lock.read_text())
        if owner.get("token") == inherited:
            os.kill(int(owner["pid"]), 0)
            yield inherited
            return
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Workspace is locked: {lock}. Check its owner before removing a stale lock.") from error
    token = uuid.uuid4().hex
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "token": token,
                       "created_at": datetime.now(timezone.utc).isoformat()}, stream)
        yield token
    finally:
        lock.unlink()
