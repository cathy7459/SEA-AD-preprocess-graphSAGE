from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, list_files, read_table, write_table
from src.path_audit_utils import require_existing_readable, require_writable_dir
from src.report_utils import append_stage_status
from src.spatial_utils import map_spatial_to_reference, read_spatial_object


def main() -> None:
    args = build_parser("Map spatial sections to region-matched snRNA references.").parse_args()
    config, logger = init_run("07_spatial_map_to_snrna", args.config)
    spatial_dir = require_existing_readable(Path(config["paths"]["interim_dir"]) / "spatial_qc", "spatial_qc_dir")
    ref_dir = require_existing_readable(Path(config["paths"]["processed_dir"]) / "snrna_annotated", "snrna_annotated_dir")
    out_dir = require_writable_dir(Path(config["paths"]["processed_dir"]) / "spatial_mapping", "spatial_mapping_output_dir")

    spatial_files = [p for p in list_files(spatial_dir, (".h5ad",)) if "_qc_" in p.name]
    ref_files = [p for p in list_files(ref_dir, (".h5ad",)) if "annotated" in p.name]
    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    if not spatial_files:
        no_input_path = out_dir / "seaad_spatial_mapping_no_inputs_v1.tsv"
        write_table(pd.DataFrame([{"reason": "No QC-filtered spatial h5ad inputs were available for mapping."}]), no_input_path)
        append_stage_status(
            stage_status_path,
            "07_spatial_map_to_snrna",
            "blocked",
            "Spatial mapping skipped because spatial QC outputs were missing.",
            n_inputs=0,
            n_outputs=1,
        )
        finalize_run(config, "07_spatial_map_to_snrna", args.config, [str(no_input_path)], "partial")
        return

    for spatial_file in spatial_files:
        region = next((token for token in spatial_file.stem.split("_") if token in set(config["analysis"]["supported_regions"] + ["CN"])), "unknown")
        donor = next((token for token in spatial_file.stem.split("_") if token.startswith("H") or token.startswith("donor-")), "donor-unknown")
        section = next((token for token in spatial_file.stem.split("_") if token.isdigit()), "section-unknown")
        reference_path = next((p for p in ref_files if f"_{region}_" in p.name), None)
        output_path = out_dir / spatial_file.name.replace("_qc_", "_mapped_").replace(".h5ad", ".tsv")
        spatial_data = read_spatial_object(spatial_file)
        if not isinstance(spatial_data, ad.AnnData):
            raise TypeError(f"Expected spatial QC h5ad for mapping, got {type(spatial_data)} from {spatial_file}")
        if "donor_id" in spatial_data.obs.columns and spatial_data.n_obs:
            donor = str(spatial_data.obs["donor_id"].iloc[0])
        if "section_id" in spatial_data.obs.columns and spatial_data.n_obs:
            section = str(spatial_data.obs["section_id"].iloc[0])
        if "region" in spatial_data.obs.columns and spatial_data.n_obs:
            region = str(spatial_data.obs["region"].iloc[0])
        logger.info("Spatial mapping start | donor=%s | region=%s | section=%s | file=%s", donor, region, section, spatial_file.name)

        if reference_path is None:
            template = pd.DataFrame(
                {
                    "obs_name": [],
                    "mapped_cell_type": [],
                    "mapped_confidence": [],
                    "mapping_status": [],
                    "mapping_note": [],
                    "donor_id": [],
                    "region": [],
                    "section_id": [],
                }
            )
            template.loc[0] = ["", "", 0.0, "no_region_reference", f"No region-matched snRNA reference for region={region}", donor, region, section]
            write_table(template, output_path)
            outputs.append(str(output_path))
            continue

        reference = sc.read_h5ad(reference_path)
        if "major_cell_class" not in reference.obs.columns:
            raise KeyError(f"snRNA reference missing required 'major_cell_class': {reference_path}")

        mapped = map_spatial_to_reference(reference, spatial_data, label_col="major_cell_class")
        mapped["mapping_status"] = "mapped"
        mapped["mapping_note"] = ""
        mapped["donor_id"] = donor
        mapped["region"] = region
        mapped["section_id"] = section
        write_table(mapped, output_path)
        outputs.append(str(output_path))
        logger.info(
            "Spatial mapping complete | donor=%s | region=%s | section=%s | mapped_cells=%s",
            donor,
            region,
            section,
            len(mapped),
        )

    mapping_manifest = pd.DataFrame(
        [{"mapping_path": str(path), "file_name": Path(path).name} for path in outputs if path.endswith(".tsv")]
    )
    manifest_path = write_table(mapping_manifest, out_dir / "seaad_spatial_mapping_manifest_v1.tsv")
    outputs.append(str(manifest_path))

    append_stage_status(
        stage_status_path,
        "07_spatial_map_to_snrna",
        "success" if outputs else "blocked",
        "Spatial mapping writes explicit section-level cell-type assignments or an explicit no-reference status.",
        n_inputs=len(spatial_files),
        n_outputs=len(outputs),
    )
    finalize_run(config, "07_spatial_map_to_snrna", args.config, outputs, "success" if outputs else "partial")


if __name__ == "__main__":
    main()
