from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.io_utils import write_table


def check_path(path: str | Path, kind: str) -> dict[str, object]:
    target = Path(path)
    exists = target.exists()
    is_dir = target.is_dir()
    is_file = target.is_file()
    readable = os.access(target, os.R_OK) if exists else False
    writable = os.access(target if is_dir else target.parent, os.W_OK) if exists or target.parent.exists() else False
    return {
        "path": str(target),
        "kind": kind,
        "exists": exists,
        "is_dir": is_dir,
        "is_file": is_file,
        "readable": readable,
        "writable_parent": writable,
    }


def require_existing_readable(path: str | Path, kind: str) -> Path:
    info = check_path(path, kind)
    if not info["exists"]:
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    if not info["readable"]:
        raise PermissionError(f"{kind} is not readable: {path}")
    return Path(path)


def require_writable_dir(path: str | Path, kind: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    info = check_path(target, kind)
    if not info["writable_parent"]:
        raise PermissionError(f"{kind} is not writable: {path}")
    return target


def audit_paths(rows: list[dict[str, object]], out_path: str | Path) -> Path:
    return write_table(pd.DataFrame(rows), out_path)
