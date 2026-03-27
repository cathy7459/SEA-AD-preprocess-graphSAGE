from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

from src.ann_utils import shared_genes

SPATIAL_REGION_ALIASES = {
    "middle-temporal-gyrus": "MTG",
    "caudate_nucleus": "CN",
    "caudate-nucleus": "CN",
    "mtg": "MTG",
    "cn": "CN",
}


def infer_region_from_spatial_path(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]
    for part in parts:
        if part in SPATIAL_REGION_ALIASES:
            return SPATIAL_REGION_ALIASES[part]
    return "unknown"


def infer_ids_from_spatial_path(path: str | Path) -> tuple[str, str, str]:
    parts = Path(path).parts
    donor_id = next((part for part in parts if part.startswith("H") and "." in part), "donor-unknown")
    numeric_parts = [part for part in parts if part.isdigit()]
    section_id = numeric_parts[-1] if numeric_parts else Path(path).stem[:48]
    region = infer_region_from_spatial_path(path)
    return donor_id, region, section_id


def discover_spatial_sections(raw_dir: str | Path, selected_regions: list[str] | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows: list[dict[str, object]] = []
    for matrix_path in raw_dir.rglob("matrix.mtx.gz"):
        if "spatial_transcriptomics" not in str(matrix_path).lower():
            continue
        section_root = matrix_path.parent.parent
        donor_id, region, section_id = infer_ids_from_spatial_path(section_root)
        if selected_regions and region not in selected_regions:
            continue
        features_path = matrix_path.parent / "features.tsv.gz"
        barcodes_path = matrix_path.parent / "barcodes.tsv.gz"
        cells_parquet = section_root / "cells.parquet"
        cells_csv = section_root / "cells.csv.gz"
        cells_path = cells_parquet if cells_parquet.exists() else cells_csv
        metrics_path = section_root / "metrics_summary.csv"
        rows.append(
            {
                "section_root": str(section_root),
                "matrix_path": str(matrix_path),
                "features_path": str(features_path),
                "barcodes_path": str(barcodes_path),
                "cells_path": str(cells_path),
                "metrics_path": str(metrics_path),
                "donor_id": donor_id,
                "region": region,
                "section_id": section_id,
                "has_required_files": all(path.exists() for path in [matrix_path, features_path, barcodes_path, cells_path]),
                "has_metrics_summary": metrics_path.exists(),
            }
        )
    inventory = pd.DataFrame(rows).drop_duplicates(subset=["section_root"]).reset_index(drop=True)
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "section_root",
                "matrix_path",
                "features_path",
                "barcodes_path",
                "cells_path",
                "metrics_path",
                "donor_id",
                "region",
                "section_id",
                "has_required_files",
                "has_metrics_summary",
            ]
        )
    inventory["section_root_length"] = inventory["section_root"].astype(str).str.len()
    inventory = (
        inventory.sort_values(["donor_id", "region", "section_id", "section_root_length"])
        .drop_duplicates(subset=["donor_id", "region", "section_id"], keep="first")
        .drop(columns=["section_root_length"])
        .reset_index(drop=True)
    )
    return inventory


def discover_spatial_anomalies(raw_dir: str | Path, inventory: pd.DataFrame) -> pd.DataFrame:
    known_roots = set(inventory["section_root"].astype(str)) if not inventory.empty else set()
    rows: list[dict[str, object]] = []
    for path in Path(raw_dir).rglob("*"):
        if not path.is_file():
            continue
        text = str(path).lower()
        if "spatial_transcriptomics" not in text:
            continue
        if "cellpose-detected_transcripts.csv" in text:
            donor_id, region, section_id = infer_ids_from_spatial_path(path)
            rows.append(
                {
                    "path": str(path),
                    "donor_id": donor_id,
                    "region": region,
                    "section_id": section_id,
                    "anomaly_type": "transcript_only_file",
                    "anomaly_reason": "Transcript detections exist without a matched cell feature matrix loader in the current pipeline.",
                }
            )
        if "combined_spatialdata_zarr" in text and not any(root in text for root in known_roots):
            donor_id, region, section_id = infer_ids_from_spatial_path(path)
            rows.append(
                {
                    "path": str(path),
                    "donor_id": donor_id,
                    "region": region,
                    "section_id": section_id,
                    "anomaly_type": "zarr_fragment_not_primary_input",
                    "anomaly_reason": "SpatialData/Zarr fragments are present, but the robust primary loader uses Xenium cell_feature_matrix plus cells table.",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["path", "donor_id", "region", "section_id", "anomaly_type", "anomaly_reason"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["region", "donor_id", "section_id", "anomaly_type"]).reset_index(drop=True)


def _read_tsv_gz_lines(path: str | Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _make_unique(values: pd.Series) -> pd.Series:
    counts: dict[str, int] = {}
    out = []
    for value in values.astype(str):
        counts[value] = counts.get(value, 0) + 1
        out.append(value if counts[value] == 1 else f"{value}_{counts[value]}")
    return pd.Series(out, index=values.index)


def read_cells_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".gz":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported cells table format: {path}")


def read_metrics_summary(path: str | Path) -> dict[str, object]:
    table = pd.read_csv(path)
    if table.empty:
        return {}
    return table.iloc[0].to_dict()


def read_spatial_object(path: str | Path) -> ad.AnnData | pd.DataFrame:
    path = Path(path)
    if path.suffix == ".h5ad":
        import scanpy as sc

        return sc.read_h5ad(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported spatial file type: {path}")


def load_xenium_section(section_row: pd.Series | dict[str, object]) -> ad.AnnData:
    row = dict(section_row)
    features = pd.read_csv(
        row["features_path"],
        sep="\t",
        header=None,
        names=["feature_id", "feature_name", "feature_type"],
        compression="gzip",
    )
    barcodes = pd.Series(_read_tsv_gz_lines(row["barcodes_path"]), name="obs_name")
    with gzip.open(row["matrix_path"], "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape == (len(features), len(barcodes)):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match barcode/feature dimensions {(len(barcodes), len(features))} "
            f"for section {row['section_root']}"
        )

    cells = read_cells_table(row["cells_path"]).copy()
    if "cell_id" not in cells.columns:
        raise KeyError(f"cells table is missing 'cell_id': {row['cells_path']}")
    cells["obs_name"] = cells["cell_id"].astype(str)
    cells = cells.set_index("obs_name").reindex(barcodes.astype(str))
    cells.index.name = "obs_name"
    cells["cell_id"] = cells.get("cell_id", cells.index.to_series()).fillna(cells.index.to_series())
    cells["x"] = pd.to_numeric(cells.get("x_centroid", np.nan), errors="coerce")
    cells["y"] = pd.to_numeric(cells.get("y_centroid", np.nan), errors="coerce")
    cells["n_transcripts_raw"] = pd.to_numeric(
        cells.get("transcript_counts", cells.get("total_counts", np.nan)),
        errors="coerce",
    )
    cells["segmentation_qc"] = np.where(cells["n_transcripts_raw"].fillna(0) > 0, 1.0, 0.0)
    cells["donor_id"] = row["donor_id"]
    cells["region"] = row["region"]
    cells["section_id"] = row["section_id"]
    cells["section_root"] = row["section_root"]

    var = features.copy()
    var["feature_name"] = _make_unique(var["feature_name"].fillna(var["feature_id"]).astype(str))
    var = var.set_index("feature_name")
    adata = ad.AnnData(X=matrix, obs=cells, var=var)
    adata.obs_names = barcodes.astype(str).tolist()
    adata.var_names = var.index.astype(str)
    adata.uns["source_paths"] = {
        "matrix_path": row["matrix_path"],
        "features_path": row["features_path"],
        "barcodes_path": row["barcodes_path"],
        "cells_path": row["cells_path"],
        "metrics_path": row["metrics_path"],
    }
    return adata


def spatial_qc_table(
    data: ad.AnnData | pd.DataFrame,
    donor_id: str,
    region: str,
    section_id: str,
    metrics_summary: dict[str, object] | None = None,
) -> pd.DataFrame:
    if isinstance(data, ad.AnnData):
        obs = data.obs.copy()
        obs["obs_name"] = data.obs_names.astype(str)
        counts = np.asarray(data.X.sum(axis=1)).ravel()
        genes = np.asarray((data.X > 0).sum(axis=1)).ravel()
    else:
        obs = data.copy()
        if "obs_name" not in obs.columns:
            obs["obs_name"] = obs.index.astype(str)
        expression_cols = [c for c in obs.columns if c not in {"x", "y", "donor_id", "region", "section_id"}]
        counts = obs[expression_cols].sum(axis=1).to_numpy()
        genes = (obs[expression_cols] > 0).sum(axis=1).to_numpy()
    obs["n_transcripts"] = counts
    obs["n_genes"] = genes
    obs["x"] = pd.to_numeric(obs.get("x", obs.get("x_centroid", np.nan)), errors="coerce")
    obs["y"] = pd.to_numeric(obs.get("y", obs.get("y_centroid", np.nan)), errors="coerce")
    obs["segmentation_qc"] = pd.to_numeric(obs.get("segmentation_qc", 1.0), errors="coerce").fillna(0.0)
    obs["zero_transcript_flag"] = obs["n_transcripts"].eq(0)
    obs["donor_id"] = donor_id
    obs["region"] = region
    obs["section_id"] = section_id
    if metrics_summary:
        for key in ["panel_name", "fraction_transcripts_assigned", "fraction_empty_cells", "num_cells_detected"]:
            if key in metrics_summary:
                obs[key] = metrics_summary[key]
    return obs


def filter_spatial_qc(qc_df: pd.DataFrame, cfg: dict) -> pd.Series:
    return (
        (qc_df["n_transcripts"] >= cfg["min_transcripts_per_cell"])
        & (qc_df["n_genes"] >= cfg["min_genes_per_cell"])
        & (qc_df["segmentation_qc"] >= cfg["min_segmentation_qc"])
    )


def section_qc_summary(qc_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    donor_id = str(qc_df["donor_id"].iloc[0]) if not qc_df.empty else "donor-unknown"
    region = str(qc_df["region"].iloc[0]) if not qc_df.empty else "unknown"
    section_id = str(qc_df["section_id"].iloc[0]) if not qc_df.empty else "unknown"
    n_cells_total = int(len(qc_df))
    n_cells_retained = int(qc_df["qc_pass"].sum()) if "qc_pass" in qc_df.columns else 0
    zero_fraction = float(qc_df["zero_transcript_flag"].mean()) if "zero_transcript_flag" in qc_df.columns and n_cells_total else np.nan
    summary = pd.DataFrame(
        [
            {
                "donor_id": donor_id,
                "region": region,
                "section_id": section_id,
                "n_cells_total": n_cells_total,
                "n_cells_retained": n_cells_retained,
                "retained_fraction": n_cells_retained / n_cells_total if n_cells_total else 0.0,
                "median_n_transcripts": float(qc_df["n_transcripts"].median()) if n_cells_total else np.nan,
                "median_n_genes": float(qc_df["n_genes"].median()) if n_cells_total else np.nan,
                "zero_transcript_fraction": zero_fraction,
                "section_pass": bool(
                    n_cells_retained >= int(cfg["min_cells_per_section"])
                    and (zero_fraction <= float(cfg["max_zero_transcript_fraction"]) if pd.notna(zero_fraction) else False)
                ),
            }
        ]
    )
    return summary


def map_spatial_to_reference(reference: ad.AnnData, query: ad.AnnData, label_col: str = "major_cell_class") -> pd.DataFrame:
    genes = shared_genes(reference, query)
    if not genes:
        raise ValueError("No shared genes between reference and query for mapping.")
    ref = reference[:, genes].to_memory() if getattr(reference, "isbacked", False) else reference[:, genes].copy()
    qry = query[:, genes].copy()
    labels = ref.obs[label_col].astype(str).fillna("unknown")
    centroids = {}
    for label in sorted(labels.unique()):
        mask = labels == label
        centroids[label] = np.asarray(ref[mask].X.mean(axis=0)).ravel()
    centroid_df = pd.DataFrame(centroids).T
    query_matrix = np.asarray(qry.X.todense() if hasattr(qry.X, "todense") else qry.X)
    distances = ((query_matrix[:, None, :] - centroid_df.to_numpy()[None, :, :]) ** 2).sum(axis=2)
    best_idx = distances.argmin(axis=1)
    best_labels = centroid_df.index.to_numpy()[best_idx]
    min_dist = distances.min(axis=1)
    confidence = 1.0 / (1.0 + min_dist)
    return pd.DataFrame(
        {
            "obs_name": qry.obs_names.astype(str),
            "mapped_cell_type": best_labels,
            "mapped_confidence": confidence,
        }
    )


def patchify_spatial_adata(
    adata: ad.AnnData,
    patch_size_um: float,
    min_cells_per_patch: int,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    obs = adata.obs.copy()
    obs["patch_x"] = np.floor(obs["x"].fillna(0).to_numpy() / patch_size_um).astype(int)
    obs["patch_y"] = np.floor(obs["y"].fillna(0).to_numpy() / patch_size_um).astype(int)
    obs["patch_id"] = (
        obs["donor_id"].astype(str)
        + "__"
        + obs["section_id"].astype(str)
        + "__"
        + obs["patch_x"].astype(str)
        + "_"
        + obs["patch_y"].astype(str)
    )
    groups = obs.groupby("patch_id", sort=False)

    patch_metadata_rows: list[dict[str, object]] = []
    patch_blocks = []
    for patch_id, patch_obs in groups:
        if len(patch_obs) < min_cells_per_patch:
            continue
        indices = adata.obs_names.get_indexer(patch_obs.index.astype(str))
        block = adata.X[indices]
        if sparse.issparse(block):
            patch_blocks.append(block.sum(axis=0))
        else:
            patch_blocks.append(np.asarray(block).sum(axis=0, keepdims=True))
        row = {
            "patch_id": patch_id,
            "donor_id": str(patch_obs["donor_id"].iloc[0]),
            "region": str(patch_obs["region"].iloc[0]),
            "section_id": str(patch_obs["section_id"].iloc[0]),
            "x": float(patch_obs["x"].mean()),
            "y": float(patch_obs["y"].mean()),
            "n_cells": int(len(patch_obs)),
            "mean_n_transcripts": float(patch_obs.get("n_transcripts", pd.Series(dtype=float)).mean()) if "n_transcripts" in patch_obs else np.nan,
        }
        if "mapped_confidence" in patch_obs.columns:
            row["mapped_confidence_mean"] = float(pd.to_numeric(patch_obs["mapped_confidence"], errors="coerce").mean())
        if "mapped_cell_type" in patch_obs.columns:
            fractions = patch_obs["mapped_cell_type"].fillna("unknown").value_counts(normalize=True)
            row["dominant_mapped_cell_type"] = str(fractions.index[0])
            row["dominant_mapped_cell_type_fraction"] = float(fractions.iloc[0])
            for label, frac in fractions.items():
                row[f"celltype_fraction__{label}"] = float(frac)
        patch_metadata_rows.append(row)

    if not patch_blocks:
        return sparse.csr_matrix((0, adata.n_vars)), pd.DataFrame(
            columns=["patch_id", "donor_id", "region", "section_id", "x", "y", "n_cells"]
        )
    matrix = sparse.vstack([sparse.csr_matrix(block) for block in patch_blocks]).tocsr()
    metadata = pd.DataFrame(patch_metadata_rows)
    return matrix, metadata
