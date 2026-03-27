from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(
    project_config_path: str | Path,
    qc_config_path: str | Path = "configs/qc.yaml",
    paths_config_path: str | Path = "configs/paths.yaml",
    datasets_config_path: str | Path = "configs/datasets.yaml",
) -> dict[str, Any]:
    config = load_yaml(project_config_path)
    config["qc"] = load_yaml(qc_config_path)
    config["artifact_paths"] = load_yaml(paths_config_path)["artifacts"]
    config["datasets"] = load_yaml(datasets_config_path)["datasets"]
    return config


def ensure_analysis_regions(config: dict[str, Any], regions: list[str] | None = None) -> list[str]:
    chosen = regions or config["analysis"]["default_regions"]
    max_regions = int(config["analysis"]["max_regions"])
    if len(chosen) > max_regions:
        raise ValueError(f"Configured {len(chosen)} regions, above max_regions={max_regions}")
    return chosen
