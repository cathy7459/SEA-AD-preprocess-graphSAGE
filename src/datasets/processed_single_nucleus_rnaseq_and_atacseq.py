from __future__ import annotations

from pathlib import Path


def infer_region_from_path(path: str | Path) -> str:
    text = str(path).upper()
    for region in ["MTG", "A9", "STG", "HIP", "EC", "ITG", "V1C"]:
        if region in text:
            return region
    return "unknown"
