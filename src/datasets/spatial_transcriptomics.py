from __future__ import annotations

from pathlib import Path


def infer_spatial_platform(path: str | Path) -> str:
    text = str(path).lower()
    if "xenium" in text:
        return "xenium"
    if "merfish" in text or "merscope" in text:
        return "merfish"
    return "spatial"
