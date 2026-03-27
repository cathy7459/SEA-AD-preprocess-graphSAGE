from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.datasets.processed_single_nucleus_rnaseq_and_atacseq import infer_region_from_path
from src.datasets.spatial_transcriptomics import infer_spatial_platform
from src.io_utils import list_files, read_table, write_table
from src.report_utils import append_stage_status
from src.snrna_utils import classify_snrna_assay, infer_region_from_snrna_path
from src.spatial_utils import discover_spatial_sections, infer_region_from_spatial_path


def main() -> None:
    args = build_parser("Create region-scoped file inventories for snRNA and spatial data.").parse_args()
    config, logger = init_run("02_split_by_region", args.config)
    regions = args.regions or config["analysis"]["default_regions"]
    raw_dir = Path(config["paths"]["raw_dir"])

    rows = []
    primary_manifest_path = Path(config["artifact_paths"]["snrna_primary_manifest"])
    if primary_manifest_path.exists():
        primary_manifest = read_table(primary_manifest_path)
        for row in primary_manifest.to_dict(orient="records"):
            local_path = Path(row["local_path"])
            rows.append(
                {
                    "file_path": str(local_path),
                    "file_name": local_path.name,
                    "region": row.get("region", infer_region_from_snrna_path(local_path)),
                    "modality": "snrna",
                    "platform": "RNAseq",
                    "include_for_analysis": int(local_path.exists()),
                    "download_status": row.get("status", "unknown"),
                    "processable": 1,
                    "exists_locally": int(local_path.exists()),
                    "data_role": row.get("snrna_role", "primary_donor_rnaseq"),
                }
            )
    download_manifest_path = Path(config["artifact_paths"]["download_manifest"])
    if download_manifest_path.exists():
        download_manifest = read_table(download_manifest_path)
        for row in download_manifest.to_dict(orient="records"):
            local_path = Path(row["local_path"]) if pd.notna(row.get("local_path")) else Path(row["local_target_path"])
            suffix = Path(str(row.get("object_key", local_path.name))).suffix.lower()
            modality = str(row.get("modality", "unknown"))
            processable = suffix in {".h5ad", ".parquet", ".csv", ".tsv", ".mtx", ".gz", ".zip"}
            rows.append(
                {
                    "file_path": str(local_path),
                    "file_name": local_path.name,
                    "region": infer_region_from_path(local_path),
                    "modality": modality,
                    "platform": infer_spatial_platform(local_path) if modality == "spatial" else "single-cell",
                    "include_for_analysis": int((infer_region_from_path(local_path) in regions or infer_region_from_path(local_path) == "unknown") and processable),
                    "download_status": row.get("status", "unknown"),
                    "processable": int(processable),
                    "exists_locally": int(local_path.exists()),
                    "data_role": "manifest_inventory",
                }
            )
    for file_path in list_files(raw_dir):
        suffix = file_path.suffix.lower()
        modality = "spatial" if "spatial" in str(file_path).lower() else "snrna"
        assay = classify_snrna_assay(file_path) if modality == "snrna" else infer_spatial_platform(file_path)
        region = infer_region_from_snrna_path(file_path) if modality == "snrna" else infer_region_from_path(file_path)
        processable = suffix in {".h5ad", ".parquet", ".csv", ".tsv", ".mtx", ".gz", ".zip"}
        rows.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "region": region,
                "modality": modality,
                "platform": assay if modality == "snrna" else infer_spatial_platform(file_path),
                "include_for_analysis": int((region in regions or region == "unknown") and processable and (modality != "snrna" or assay == "RNAseq")),
                "download_status": "present_on_disk",
                "processable": int(processable),
                "exists_locally": 1,
                "data_role": "disk_inventory",
            }
        )
    spatial_sections = discover_spatial_sections(raw_dir)
    for _, spatial_row in spatial_sections.iterrows():
        rows.append(
            {
                "file_path": str(spatial_row["section_root"]),
                "file_name": Path(spatial_row["section_root"]).name,
                "region": spatial_row["region"],
                "modality": "spatial",
                "platform": "xenium_cell_feature_matrix",
                "include_for_analysis": int(bool(spatial_row["has_required_files"])),
                "download_status": "present_on_disk",
                "processable": int(bool(spatial_row["has_required_files"])),
                "exists_locally": 1,
                "data_role": "primary_spatial_section",
            }
        )
    inventory = pd.DataFrame(rows).drop_duplicates()
    outputs = []
    inventory_path = Path(config["paths"]["interim_dir"]) / "seaad_region_file_inventory_v1.tsv"
    write_table(inventory, inventory_path)
    outputs.append(str(inventory_path))

    for region in regions:
        region_df = inventory.loc[(inventory["region"] == region) | (inventory["region"] == "unknown")].copy()
        region_path = Path(config["paths"]["interim_dir"]) / f"seaad_region_inventory_{region}_v1.tsv"
        write_table(region_df, region_path)
        outputs.append(str(region_path))
        logger.info("Region %s inventory contains %s files", region, len(region_df))
    append_stage_status(
        config["artifact_paths"]["stage_status_manifest"],
        "02_split_by_region",
        "success",
        "Built region inventories from manifests and local files.",
        n_inputs=len(rows),
        n_outputs=len(outputs),
    )
    finalize_run(config, "02_split_by_region", args.config, outputs)


if __name__ == "__main__":
    main()
