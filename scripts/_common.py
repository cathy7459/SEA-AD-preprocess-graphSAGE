from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_project_config
from src.io_utils import ensure_dir
from src.logging_utils import append_run_manifest, get_logger


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regions", nargs="*", default=None)
    return parser


def init_run(script_name: str, config_path: str):
    config = load_project_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    log_dir = ensure_dir(config["paths"]["log_dir"])
    logger = get_logger(script_name, log_dir / f"{script_name}.log")
    return config, logger


def finalize_run(config: dict, script_name: str, config_path: str, outputs: list[str], status: str = "success") -> None:
    append_run_manifest(
        manifest_path=config["paths"]["run_manifest_path"],
        script_name=script_name,
        config_path=config_path,
        seed=int(config["seed"]),
        status=status,
        outputs=outputs,
    )
