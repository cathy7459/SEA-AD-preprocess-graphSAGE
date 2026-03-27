from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


def _sum_axis(matrix, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=axis)).ravel()
    return np.asarray(matrix.sum(axis=axis)).ravel()


def _nnz_axis(matrix, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        if axis == 1:
            return np.diff(matrix.tocsr().indptr)
        return np.diff(matrix.tocsc().indptr)
    return np.count_nonzero(matrix, axis=axis)


def compute_basic_qc(
    matrix,
    var_names: Iterable[str],
    obs: pd.DataFrame,
    donor_id: str,
    region: str,
    library_id: str | None = None,
) -> pd.DataFrame:
    var_index = pd.Index([str(v) for v in var_names])
    mt_mask = var_index.str.upper().str.startswith("MT-")
    ribo_mask = var_index.str.upper().str.startswith(("RPS", "RPL"))
    hb_mask = var_index.str.upper().str.startswith(("HB", "HBA", "HBB"))

    total_counts = _sum_axis(matrix, axis=1)
    n_genes = _nnz_axis(matrix, axis=1)
    mt_counts = _sum_axis(matrix[:, mt_mask], axis=1) if mt_mask.any() else np.zeros_like(total_counts)
    ribo_counts = _sum_axis(matrix[:, ribo_mask], axis=1) if ribo_mask.any() else np.zeros_like(total_counts)
    hb_counts = _sum_axis(matrix[:, hb_mask], axis=1) if hb_mask.any() else np.zeros_like(total_counts)
    denom = np.clip(total_counts, 1, None)

    qc = obs.copy()
    qc["n_counts"] = total_counts
    qc["n_genes"] = n_genes
    qc["pct_mt"] = mt_counts / denom * 100.0
    qc["pct_ribo"] = ribo_counts / denom * 100.0
    qc["pct_hb"] = hb_counts / denom * 100.0
    qc["donor_id"] = donor_id
    qc["region"] = region
    qc["library_id"] = library_id or donor_id
    return qc


def donor_wise_filter_mask(qc_df: pd.DataFrame, cfg: dict) -> pd.Series:
    min_genes = qc_df["n_genes"].quantile(cfg["donor_relative_n_genes_floor_quantile"])
    min_counts = qc_df["n_counts"].quantile(cfg["donor_relative_n_counts_floor_quantile"])
    mask = (
        (qc_df["n_genes"] >= max(cfg["min_genes_per_cell"], min_genes))
        & (qc_df["n_counts"] >= max(cfg["min_counts_per_cell"], min_counts))
        & (qc_df["pct_mt"] <= cfg["max_pct_mt"])
        & (qc_df["pct_ribo"] <= cfg["max_pct_ribo"])
        & (qc_df["pct_hb"] <= cfg["max_pct_hb"])
    )
    return mask


def summarize_qc_retention(qc_df: pd.DataFrame) -> pd.DataFrame:
    if qc_df.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["donor_id", "region"] if col in qc_df.columns]
    summary = (
        qc_df.groupby(group_cols, dropna=False)
        .agg(
            cells_total=("qc_pass", "size"),
            cells_retained=("qc_pass", "sum"),
            median_n_genes=("n_genes", "median"),
            median_n_counts=("n_counts", "median"),
            median_pct_mt=("pct_mt", "median"),
        )
        .reset_index()
    )
    return summary
