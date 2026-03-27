from __future__ import annotations

from pathlib import Path


def is_microglia_resource(path: str | Path) -> bool:
    text = str(path).lower()
    return "microgl" in text
