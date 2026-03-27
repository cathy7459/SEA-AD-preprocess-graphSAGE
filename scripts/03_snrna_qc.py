from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.ann_utils import materialize_if_backed, read_h5ad_memory_safe
from src.io_utils import ensure_dir, read_table, write_h5ad, write_table
from src.naming import artifact_name
from src.qc_utils import compute_basic_qc, donor_wise_filter_mask, summarize_qc_retention
from src.report_utils import append_stage_status
from src.snrna_utils import discover_snrna_h5ad_files, harmonize_obs_metadata


def process_adata(adata: ad.AnnData, region: str, project: str, version: str, out_dir: Path, qc_cfg: dict) -> list[str]:
    outputs: list[str] = []
    obs = harmonize_obs_metadata(adata.obs, file_path=adata.uns.get("source_file", ""))
    donors = sorted(obs["donor_id"].astype(str).unique()) if "donor_id" in obs.columns else ["donor-unknown"]
    for donor_id in donors:
        donor_mask = obs["donor_id"].astype(str) == donor_id if "donor_id" in obs.columns else pd.Series(True, index=obs.index)
        region_mask = obs["region"].astype(str).str.upper() == region.upper() if region != "unknown" else pd.Series(True, index=obs.index)
        mask = donor_mask & region_mask
        if int(mask.sum()) < qc_cfg["min_cells_per_donor_region"]:
            continue
        subset = adata[mask].to_memory() if getattr(adata, "isbacked", False) else adata[mask].copy()
        subset.obs = obs.loc[mask].copy()
        qc_table = compute_basic_qc(subset.X, subset.var_names, subset.obs, donor_id=donor_id, region=region)
        qc_table["qc_pass"] = donor_wise_filter_mask(qc_table, qc_cfg)
        subset = subset[qc_table["qc_pass"].to_numpy()].copy()
        subset.obs = qc_table.loc[qc_table["qc_pass"]].copy()
        # Keep QC intermediates lightweight and readable across anndata versions.
        subset.uns = {
            "source_file": str(adata.uns.get("source_file", "")),
            "source_stage": "03_snrna_qc",
            "assay": "RNAseq",
            "region": region,
            "donor_id": donor_id,
        }
        subset.obsm.clear()
        subset.obsp.clear()
        subset.varm.clear()
        subset.varp.clear()
        h5ad_name = artifact_name(project, "snrna", region, f"donor-{donor_id}", "qc", version, "h5ad")
        table_name = artifact_name(project, "snrna", region, f"donor-{donor_id}", "qc-metrics", version, "tsv")
        write_h5ad(subset, out_dir / h5ad_name)
        write_table(qc_table, out_dir / table_name)
        outputs.extend([str(out_dir / h5ad_name), str(out_dir / table_name)])
    return outputs


def main() -> None:
    args = build_parser("Run donor-wise snRNA QC and save filtered donor-region objects.").parse_args()
    config, logger = init_run("03_snrna_qc", args.config)
    regions = args.regions or config["analysis"]["default_regions"]
    primary_manifest_path = Path(config["artifact_paths"]["snrna_primary_manifest"])
    if primary_manifest_path.exists():
        primary_manifest = read_table(primary_manifest_path)
        raw_files = [Path(path) for path in primary_manifest.get("local_path", pd.Series(dtype=str)).dropna().tolist() if Path(path).exists()]
    else:
        raw_files = discover_snrna_h5ad_files(config["paths"]["raw_dir"])
    out_dir = ensure_dir(Path(config["paths"]["interim_dir"]) / "snrna_qc")
    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]
    if not raw_files:
        download_manifest_path = Path(config["artifact_paths"]["download_manifest"])
        candidate_manifest = out_dir / "seaad_snrna_qc_input_candidates_v1.tsv"
        if download_manifest_path.exists():
            download_manifest = read_table(download_manifest_path)
            candidates = download_manifest.loc[download_manifest["modality"] == "snrna"].copy()
            write_table(candidates, candidate_manifest)
            outputs.append(str(candidate_manifest))
        append_stage_status(
            stage_status_path,
            "03_snrna_qc",
            "blocked",
            "No local processable snRNA .h5ad files were found. Saved candidate input manifest instead of producing empty QC outputs.",
            n_inputs=0,
            n_outputs=len(outputs),
        )
        finalize_run(config, "03_snrna_qc", args.config, outputs, "partial")
        return
    for file_path in raw_files:
        logger.info("QC processing %s", file_path)
        adata = read_h5ad_memory_safe(file_path, config["memory"]["force_backed_h5ad_above_gb"])
        adata.uns["source_file"] = str(file_path)
        obs_preview = harmonize_obs_metadata(adata.obs, file_path=file_path)
        logger.info(
            "Loaded %s with %s cells, %s genes, donor(s)=%s, region(s)=%s, diagnosis=%s",
            file_path.name,
            adata.n_obs,
            adata.n_vars,
            ",".join(sorted(obs_preview["donor_id"].astype(str).unique())[:5]),
            ",".join(sorted(obs_preview["region"].astype(str).unique())[:5]),
            ",".join(sorted(obs_preview["diagnosis"].astype(str).unique())[:5]),
        )
        for region in regions:
            file_outputs = process_adata(
                adata=adata,
                region=region,
                project=config["project_name"],
                version=config["version"],
                out_dir=out_dir,
                qc_cfg=config["qc"]["snrna"],
            )
            outputs.extend(file_outputs)
        if getattr(adata, "isbacked", False):
            adata.file.close()
    qc_metric_tables = list(out_dir.glob("*qc-metrics*.tsv"))
    if qc_metric_tables:
        all_qc = pd.concat([pd.read_csv(p, sep="\t") for p in qc_metric_tables], ignore_index=True)
        summary = summarize_qc_retention(all_qc)
        summary_path = out_dir / "seaad_snrna_qc_summary_v1.tsv"
        write_table(summary, summary_path)
        outputs.append(str(summary_path))
        append_stage_status(
            stage_status_path,
            "03_snrna_qc",
            "success",
            "snRNA donor-region QC completed.",
            n_inputs=len(raw_files),
            n_outputs=len(outputs),
        )
    else:
        reason_path = out_dir / "seaad_snrna_qc_no_outputs_v1.tsv"
        write_table(
            pd.DataFrame(
                [
                    {
                        "reason": "Local snRNA files existed, but no donor-region subset met configured minimum cells or region filters.",
                        "n_raw_files": len(raw_files),
                    }
                ]
            ),
            reason_path,
        )
        outputs.append(str(reason_path))
        append_stage_status(
            stage_status_path,
            "03_snrna_qc",
            "blocked",
            "snRNA raw files were found but produced no QC-passing donor-region outputs.",
            n_inputs=len(raw_files),
            n_outputs=len(outputs),
        )
    finalize_run(config, "03_snrna_qc", args.config, outputs, "success" if qc_metric_tables else "partial")


if __name__ == "__main__":
    main()
