from __future__ import annotations

from pathlib import Path


def sanitize_token(value: str | None, fallback: str = "na") -> str:
    if not value:
        return fallback
    return str(value).strip().replace(" ", "-").replace("/", "-")


def artifact_name(
    project: str,
    modality: str,
    region: str,
    unit: str,
    stage: str,
    version: str,
    ext: str,
) -> str:
    return (
        f"{sanitize_token(project)}_{sanitize_token(modality)}_{sanitize_token(region)}_"
        f"{sanitize_token(unit)}_{sanitize_token(stage)}_{sanitize_token(version)}.{ext.lstrip('.')}"
    )


def artifact_path(base_dir: str | Path, *parts: str) -> Path:
    return Path(base_dir).joinpath(*parts)
