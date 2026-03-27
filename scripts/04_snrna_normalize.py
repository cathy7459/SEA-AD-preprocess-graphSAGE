from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.ann_utils import compute_hvg_mask, normalize_log1p_inplace
from src.io_utils import ensure_dir, list_files, write_h5ad, write_table
from src.naming import artifact_name
from src.report_utils import append_stage_status


def main() -> None:
    args = build_parser("Normalize donor-region snRNA objects and compute region-wise HVG voting tables.").parse_args()
    config, logger = init_run("04_snrna_normalize", args.config)
    in_dir = Path(config["paths"]["interim_dir"]) / "snrna_qc"
    out_dir = ensure_dir(Path(config["paths"]["processed_dir"]) / "snrna_normalized")
    files = [p for p in list_files(in_dir, (".h5ad",)) if "_qc_" in p.name]
    region_votes: dict[str, Counter] = defaultdict(Counter)
    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    if not files:
        no_input_path = out_dir / "seaad_snrna_normalize_no_inputs_v1.tsv"
        write_table(
            pd.DataFrame([{"reason": "No QC-filtered snRNA .h5ad inputs were found in data/interim/snrna_qc."}]),
            no_input_path,
        )
        append_stage_status(
            stage_status_path,
            "04_snrna_normalize",
            "blocked",
            "Normalization skipped because no QC-filtered snRNA inputs were available.",
            n_inputs=0,
            n_outputs=1,
        )
        finalize_run(config, "04_snrna_normalize", args.config, [str(no_input_path)], "partial")
        return

    for file_path in files:
        adata = sc.read_h5ad(file_path)
        logger.info("Normalizing %s", file_path.name)
        adata = normalize_log1p_inplace(adata)
        hvg_genes = compute_hvg_mask(adata, config["qc"]["snrna"]["hvg_top_n"])
        hvg_use = [gene for gene in hvg_genes if gene in adata.var_names]
        if hvg_use:
            pca_view = adata[:, hvg_use].copy()
            sc.pp.pca(pca_view, n_comps=min(20, max(2, len(hvg_use) - 1)))
            adata.obsm["X_pca_hvg"] = np.asarray(pca_view.obsm["X_pca"])
        region = next((token for token in file_path.stem.split("_") if token in config["analysis"]["supported_regions"]), "unknown")
        region_votes[region].update(hvg_genes)
        donor_token = next((token for token in file_path.stem.split("_") if token.startswith("donor-")), "donor-unknown")
        out_name = artifact_name(config["project_name"], "snrna", region, donor_token, "normalized", config["version"], "h5ad")
        write_h5ad(adata, out_dir / out_name)
        outputs.append(str(out_dir / out_name))
        if "X_pca_hvg" in adata.obsm:
            feature_df = pd.DataFrame(adata.obsm["X_pca_hvg"], index=adata.obs_names)
            feature_df.columns = [f"pca_hvg_{i+1}" for i in range(feature_df.shape[1])]
            feature_df = feature_df.reset_index(names="obs_name")
            for column in ["donor_id", "region", "diagnosis", "major_cell_class", "subclass"]:
                if column in adata.obs.columns:
                    feature_df[column] = adata.obs[column].astype(str).values
            feature_path = out_dir / out_name.replace(".h5ad", "_cell-features.parquet")
            write_table(feature_df, feature_path)
            outputs.append(str(feature_path))

    for region, votes in region_votes.items():
        table = pd.DataFrame({"gene": list(votes.keys()), "n_donor_votes": list(votes.values())})
        table["region"] = region
        table["selected_hvg"] = table["n_donor_votes"] >= config["qc"]["snrna"]["hvg_min_donor_votes"]
        path = out_dir / f"seaad_snrna_{region}_hvg_votes_{config['version']}.tsv"
        write_table(table.sort_values(["selected_hvg", "n_donor_votes"], ascending=[False, False]), path)
        outputs.append(str(path))
    append_stage_status(
        stage_status_path,
        "04_snrna_normalize",
        "success",
        "Normalized snRNA donor-region objects and computed HVG vote tables.",
        n_inputs=len(files),
        n_outputs=len(outputs),
    )
    finalize_run(config, "04_snrna_normalize", args.config, outputs, "success" if outputs else "partial")


if __name__ == "__main__":
    main()
