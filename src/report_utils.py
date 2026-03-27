from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pandas as pd

from src.io_utils import write_table


def collect_software_versions(packages: list[str], path: str | Path) -> Path:
    rows = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            rows.append({"package": package, "version": version})
        except Exception:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                rows.append({"package": package, "version": version})
            except Exception as exc:
                rows.append({"package": package, "version": f"unavailable: {exc}"})
    return write_table(pd.DataFrame(rows), path)


def figure_manifest_rows(figure_paths: list[Path], description: str) -> pd.DataFrame:
    return pd.DataFrame([{"figure_path": str(path), "description": description} for path in figure_paths])


def append_stage_status(
    path: str | Path,
    stage_name: str,
    status: str,
    message: str,
    n_inputs: int = 0,
    n_outputs: int = 0,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "stage_name": stage_name,
                "status": status,
                "message": message,
                "n_inputs": n_inputs,
                "n_outputs": n_outputs,
            }
        ]
    )
    if out.exists():
        row.to_csv(out, sep="\t", index=False, mode="a", header=False)
    else:
        row.to_csv(out, sep="\t", index=False)
    return out
