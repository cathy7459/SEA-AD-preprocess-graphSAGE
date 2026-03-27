from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from src.io_utils import file_size_gb


def read_h5ad_memory_safe(
    path: str | Path,
    force_backed_above_gb: float = 1.0,
    backed_mode: str = "r",
) -> ad.AnnData:
    if file_size_gb(path) >= force_backed_above_gb:
        return sc.read_h5ad(path, backed=backed_mode)
    return sc.read_h5ad(path)


def materialize_if_backed(adata: ad.AnnData) -> ad.AnnData:
    if getattr(adata, "isbacked", False):
        return adata.to_memory()
    return adata


def normalize_log1p_inplace(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    adata = materialize_if_backed(adata)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata)
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(np.asarray(adata.X))
    adata.layers["log1p_norm"] = adata.X.copy()
    return adata


def compute_hvg_mask(adata: ad.AnnData, n_top_genes: int, batch_key: str | None = None) -> list[str]:
    adata = materialize_if_backed(adata)
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            batch_key=batch_key,
            inplace=True,
        )
        return adata.var_names[adata.var["highly_variable"]].tolist()
    except ImportError:
        pass
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="cell_ranger",
            batch_key=batch_key,
            inplace=True,
        )
        return adata.var_names[adata.var["highly_variable"]].tolist()
    except Exception:
        matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene_vars = matrix.var(axis=0)
        top_idx = np.argsort(gene_vars)[::-1][:n_top_genes]
        return adata.var_names[top_idx].tolist()


def shared_genes(reference: ad.AnnData, query: ad.AnnData) -> list[str]:
    return sorted(set(reference.var_names).intersection(set(query.var_names)))
