from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.datasets.donor_metadata import standardize_donor_metadata
from src.io_utils import ensure_dir, write_table
from src.seaad_manifest_utils import build_remote_manifest
from src.snrna_utils import discover_snrna_h5ad_files, summarize_h5ad_metadata
from src.spatial_utils import discover_spatial_sections
from src.web_fetch_utils import build_session, fetch_html, parse_links, stream_download


def main() -> None:
    args = build_parser("Build cohort and donor overlap tables from official SEA-AD metadata.").parse_args()
    config, logger = init_run("00_build_cohort_table", args.config)
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    session = build_session(config["web"]["user_agent"])
    try:
        html = fetch_html(session, config["web"]["base_page"], timeout=config["web"]["timeout_seconds"])
        links = parse_links(html, config["web"]["base_page"])
        manifest = build_remote_manifest(links, config["datasets"], raw_dir)
        donor_rows = manifest.loc[manifest["dataset_id"] == "donor_metadata"]
    except Exception as exc:
        logger.warning("Official donor metadata page fetch failed; falling back to local donor metadata if present: %s", exc)
        local_donor_files = sorted((Path(raw_dir) / "donor_metadata").glob("*.xlsx"))
        if not local_donor_files:
            raise
        donor_rows = pd.DataFrame(
            [
                {
                    "dataset_id": "donor_metadata",
                    "local_target_path": str(local_donor_files[0]),
                    "remote_url": config["web"]["base_page"],
                }
            ]
        )
    outputs: list[str] = []

    cohort_path = Path(config["artifact_paths"]["cohort_table"])
    overlap_path = Path(config["artifact_paths"]["overlap_table"])
    if donor_rows.empty:
        logger.warning("No donor metadata link was discovered on the official page.")
        write_table(pd.DataFrame(columns=["donor_id"]), cohort_path)
        write_table(pd.DataFrame(columns=["donor_id", "region", "modality", "available"]), overlap_path)
        finalize_run(config, "00_build_cohort_table", args.config, [str(cohort_path), str(overlap_path)], "partial")
        return

    donor_target = Path(donor_rows.iloc[0]["local_target_path"])
    if args.overwrite or not donor_target.exists():
        download_result = stream_download(
            session,
            donor_rows.iloc[0]["remote_url"],
            donor_target,
            retries=config["web"]["retries"],
            timeout=config["web"]["timeout_seconds"],
        )
        logger.info("Donor metadata fetch status: %s", download_result["status"])

    donor_df = pd.read_excel(donor_target)
    donor_df = standardize_donor_metadata(donor_df)
    write_table(donor_df, cohort_path)
    outputs.append(str(cohort_path))

    overlap = []
    donor_ids = set(donor_df["donor_id"].astype(str).tolist())
    snrna_files = discover_snrna_h5ad_files(raw_dir)
    for path in snrna_files:
        metadata = summarize_h5ad_metadata(path)
        donor_id = str(metadata["donor_id_example"])
        region = str(metadata["region_example"])
        overlap.append(
            {
                "donor_id": donor_id,
                "region": region,
                "modality": "snrna",
                "available": int(Path(path).exists()),
                "source": "local_primary_snrna_object",
            }
        )
        donor_ids.add(donor_id)

    spatial_sections = discover_spatial_sections(raw_dir)
    for _, row in spatial_sections.iterrows():
        donor_id = str(row["donor_id"])
        overlap.append(
            {
                "donor_id": donor_id,
                "region": str(row["region"]),
                "modality": "spatial",
                "available": int(bool(row["has_required_files"])),
                "source": "local_primary_spatial_section",
            }
        )
        donor_ids.add(donor_id)

    if overlap:
        overlap_df = pd.DataFrame(overlap).drop_duplicates(subset=["donor_id", "region", "modality"], keep="last")
    else:
        regions = args.regions or config["analysis"]["default_regions"]
        overlap_df = pd.DataFrame(
            [
                {
                    "donor_id": donor_id,
                    "region": region,
                    "modality": modality,
                    "available": pd.NA,
                    "source": "metadata_initialized",
                }
                for donor_id in sorted(donor_ids)
                for region in regions
                for modality in config["analysis"]["modalities"]
            ]
        )
    write_table(overlap_df, overlap_path)
    outputs.append(str(overlap_path))
    finalize_run(config, "00_build_cohort_table", args.config, outputs)


if __name__ == "__main__":
    main()
