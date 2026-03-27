from __future__ import annotations

import logging
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def get_logger(name: str, log_path: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def append_run_manifest(
    manifest_path: str | Path,
    script_name: str,
    config_path: str | Path,
    seed: int,
    status: str,
    outputs: Iterable[str],
) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "hostname": socket.gethostname(),
                "script_name": script_name,
                "config_path": str(config_path),
                "seed": seed,
                "status": status,
                "outputs": ";".join(outputs),
                "python_version": sys.version.replace("\n", " "),
            }
        ]
    )
    if path.exists():
        row.to_csv(path, sep="\t", index=False, mode="a", header=False)
    else:
        row.to_csv(path, sep="\t", index=False)
