from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif out.suffix in {".tsv", ".txt"}:
        df.to_csv(out, sep="\t", index=False)
    else:
        df.to_csv(out, index=False)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    table = Path(path)
    try:
        if table.suffix == ".parquet":
            return pd.read_parquet(table)
        if table.suffix in {".tsv", ".txt"}:
            return pd.read_csv(table, sep="\t")
        return pd.read_csv(table)
    except EmptyDataError:
        return pd.DataFrame()


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _sanitize_h5_key(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("/", "_")
    return value


def _sanitize_mapping_keys(mapping):
    sanitized = {}
    for key, value in mapping.items():
        new_key = _sanitize_h5_key(key)
        if isinstance(value, dict):
            sanitized[new_key] = _sanitize_mapping_keys(value)
        else:
            sanitized[new_key] = value
    mapping.clear()
    mapping.update(sanitized)


def sanitize_anndata_for_h5ad(adata) -> None:
    """Sanitize AnnData metadata keys that future h5 stores will reject.

    anndata/h5py are moving away from allowing forward slashes in stored keys.
    We normalize common mutable containers before write_h5ad so saved files stay
    forward-compatible.
    """
    if hasattr(adata, "uns") and isinstance(adata.uns, dict):
        _sanitize_mapping_keys(adata.uns)

    for attr_name in ["obs", "var"]:
        frame = getattr(adata, attr_name, None)
        if frame is not None and hasattr(frame, "columns"):
            frame.columns = [_sanitize_h5_key(col) for col in frame.columns]

    for attr_name in ["obsm", "varm", "obsp", "varp", "layers"]:
        mapping = getattr(adata, attr_name, None)
        if mapping is not None and hasattr(mapping, "keys"):
            keys = list(mapping.keys())
            for key in keys:
                new_key = _sanitize_h5_key(key)
                if new_key != key:
                    mapping[new_key] = mapping[key]
                    del mapping[key]


def write_h5ad(adata, path: str | Path) -> Path:
    import anndata as ad  # local import to avoid forcing plotting-only scripts to import heavy deps

    if not isinstance(adata, ad.AnnData):
        raise TypeError("write_h5ad expects an AnnData object.")
    sanitize_anndata_for_h5ad(adata)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    return out


def file_size_gb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024**3)


def list_files(root: str | Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    root_path = Path(root)
    files = [p for p in root_path.rglob("*") if p.is_file()]
    if suffixes is None:
        return files
    return [p for p in files if p.suffix.lower() in suffixes]


def safe_exists(path: str | Path) -> bool:
    return Path(path).exists()


def maybe_relpath(path: str | Path, start: str | Path | None = None) -> str:
    try:
        return os.path.relpath(str(path), str(start or Path.cwd()))
    except ValueError:
        return str(path)
