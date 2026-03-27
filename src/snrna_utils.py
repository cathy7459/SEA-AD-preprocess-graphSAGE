from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc

from src.io_utils import list_files


RNASEQ_HINTS = ("rnaseq", "gex", "gene_expression")
ATAC_HINTS = ("atac", "peak", "chromatin")


def classify_snrna_assay(path: str | Path) -> str:
    path_obj = Path(path)
    parts = [part.lower() for part in path_obj.parts[-6:]]
    name_text = path_obj.name.lower()
    if "atacseq" in parts or "atacseq" in name_text:
        return "ATACseq"
    if "rnaseq" in parts or "rnaseq" in name_text:
        return "RNAseq"
    return "unknown"


def discover_snrna_h5ad_files(raw_dir: str | Path) -> list[Path]:
    files = []
    for path in list_files(raw_dir, (".h5ad",)):
        assay = classify_snrna_assay(path)
        path_text = str(path).lower()
        is_donor_object = "donor_objects" in path_text and "final-nuclei" in path_text
        if assay == "RNAseq" and is_donor_object:
            files.append(path)
    return sorted(files)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_region_label(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    mapping = {
        "HUMAN MTG": "MTG",
        "MTG": "MTG",
        "HUMAN MTG_L5": "MTG",
        "HUMAN DFC_ALL_LAYERS": "A9",
        "DFC_ALL_LAYERS": "A9",
        "PFC": "A9",
        "A9": "A9",
        "DLPFC": "A9",
    }
    for key, region in mapping.items():
        if key in upper:
            return region
    return text or "unknown"


def normalize_diagnosis_label(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    if upper in {"REFERENCE", "NEUROTYPICAL", "CONTROL"}:
        return "reference"
    if "NO DEMENTIA" in upper:
        return "no_dementia"
    if "DEMENTIA" in upper:
        return "dementia"
    if upper in {"HIGH", "INTERMEDIATE", "LOW", "NOT AD"}:
        return upper.lower().replace(" ", "_")
    return text.lower().replace(" ", "_") if text else "unknown"


def infer_region_from_snrna_path(path: str | Path) -> str:
    text = str(path).upper()
    if "SEAAD_A9_" in text or "\\PFC\\" in text or "DFC_ALL_LAYERS" in text:
        return "A9"
    if "SEAAD_MTG_" in text or "\\MTG\\" in text:
        return "MTG"
    for region in ["STG", "HIP", "EC", "ITG", "V1C"]:
        if region in text:
            return region
    return "unknown"


def is_primary_snrna_object(path: str | Path) -> bool:
    text = str(path).lower()
    return classify_snrna_assay(path) == "RNAseq" and "donor_objects" in text and "final-nuclei" in text


def harmonize_obs_metadata(obs: pd.DataFrame, file_path: str | Path | None = None) -> pd.DataFrame:
    out = obs.copy()
    rename_map = {col: normalize_column_name(col) for col in out.columns}
    out = out.rename(columns=rename_map)

    donor_col = next((c for c in out.columns if c in {"donor_id", "donorid"}), None)
    if donor_col is None:
        donor_col = next((c for c in out.columns if c.startswith("donor")), None)
    if donor_col is not None:
        out["donor_id"] = out[donor_col].astype(str)
    else:
        out["donor_id"] = Path(file_path).name.split("_")[0] if file_path is not None else "donor-unknown"

    region_col = next((c for c in out.columns if c in {"brain_region", "region"}), None)
    if region_col is not None:
        out["region"] = out[region_col].map(normalize_region_label)
    else:
        out["region"] = normalize_region_label(infer_region_from_snrna_path(file_path or ""))

    cognitive_col = next((c for c in out.columns if c == "cognitive_status"), None)
    path_col = next((c for c in out.columns if c == "overall_ad_neuropathological_change"), None)
    out["diagnosis_raw"] = (
        out[cognitive_col].astype(str)
        if cognitive_col is not None
        else out[path_col].astype(str) if path_col is not None else "unknown"
    )
    out["diagnosis"] = out["diagnosis_raw"].map(normalize_diagnosis_label)

    if "neurotypical_reference" in out.columns:
        ref_mask = out["neurotypical_reference"].astype(str).str.lower() == "true"
        out.loc[ref_mask, "diagnosis"] = "reference"
        out.loc[ref_mask, "diagnosis_raw"] = out.loc[ref_mask, "diagnosis_raw"].where(
            out.loc[ref_mask, "diagnosis_raw"].astype(str).ne("unknown"),
            "Reference",
        )

    major_class_col = next((c for c in out.columns if c in {"class", "broad_cell_type", "supertype"}), None)
    subclass_col = next((c for c in out.columns if c == "subclass"), None)
    out["major_cell_class"] = out[major_class_col].astype(str) if major_class_col is not None else "unknown"
    out["subclass"] = out[subclass_col].astype(str) if subclass_col is not None else "unknown"
    out["source_file"] = str(file_path) if file_path is not None else ""
    return out


def summarize_h5ad_metadata(path: str | Path) -> dict[str, Any]:
    adata = sc.read_h5ad(path, backed="r")
    try:
        obs = harmonize_obs_metadata(adata.obs, file_path=path)
        summary = {
            "file_path": str(path),
            "file_name": Path(path).name,
            "assay": classify_snrna_assay(path),
            "loaded": True,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "donor_id_example": obs["donor_id"].iloc[0] if len(obs) else "unknown",
            "region_example": obs["region"].iloc[0] if len(obs) else infer_region_from_snrna_path(path),
            "diagnosis_labels_raw": sorted(pd.Series(obs["diagnosis_raw"]).dropna().astype(str).unique().tolist()),
            "diagnosis_labels_normalized": sorted(pd.Series(obs["diagnosis"]).dropna().astype(str).unique().tolist()),
            "has_donor_id": int("donor_id" in obs.columns),
            "has_region": int("region" in obs.columns),
            "has_diagnosis": int("diagnosis" in obs.columns),
            "has_subclass": int("subclass" in obs.columns and (obs["subclass"] != "unknown").any()),
            "obs_columns": list(obs.columns),
        }
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    return summary
