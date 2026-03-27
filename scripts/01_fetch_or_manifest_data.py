from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, write_table
from src.report_utils import append_stage_status, collect_software_versions
from src.seaad_manifest_utils import build_bucket_manifest, build_remote_manifest, summarize_manifest
from src.snrna_utils import infer_region_from_snrna_path, is_primary_snrna_object
from src.web_fetch_utils import (
    build_session,
    fetch_html,
    head_metadata,
    list_public_s3_bucket,
    parse_links,
    stream_download,
)

def local_manifest_from_disk(raw_dir: Path, datasets: list[dict]) -> pd.DataFrame:
    dataset_lookup = {row["dataset_id"]: row for row in datasets}
    rows = []
    for dataset_dir in raw_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_id = dataset_dir.name
        meta = dataset_lookup.get(dataset_id, {"display_name": dataset_id, "modality": "unknown"})
        for file_path in dataset_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "display_name": meta.get("display_name", dataset_id),
                    "modality": meta.get("modality", "unknown"),
                    "source_page_url": "",
                    "link_title": file_path.name,
                    "remote_url": "",
                    "file_type": file_path.suffix.lstrip(".") or "unknown",
                    "direct_download": 0,
                    "manual_download_needed": 0,
                    "local_target_path": str(file_path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser("Fetch directly accessible SEA-AD data and build a manifest for manual resources.").parse_args()
    config, logger = init_run("01_fetch_or_manifest_data", args.config)
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    manifest_dir = ensure_dir(config["paths"]["manifest_dir"])
    session = build_session(config["web"]["user_agent"])
    offline_mode = False
    try:
        html = fetch_html(session, config["web"]["base_page"], timeout=config["web"]["timeout_seconds"])
        links = parse_links(html, config["web"]["base_page"])
        manifest = build_remote_manifest(links, config["datasets"], raw_dir)
    except Exception as exc:
        logger.warning("Official SEA-AD page fetch failed; attempting offline manifest reuse: %s", exc)
        offline_mode = True
        manifest_path = Path(config["artifact_paths"]["fetch_manifest"])
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, sep="\t")
        else:
            manifest = local_manifest_from_disk(Path(raw_dir), config["datasets"])
        links = []
    manifest_path = Path(config["artifact_paths"]["fetch_manifest"])
    write_table(manifest, manifest_path)

    bucket_rows = []
    if not offline_mode:
        for bucket_name in ["sea-ad-single-cell-profiling", "sea-ad-spatial-transcriptomics", "sea-ad-quantitative-neuropathology"]:
            try:
                logger.info("Listing public bucket objects for %s", bucket_name)
                bucket_rows.extend(list_public_s3_bucket(session, bucket_name, timeout=config["web"]["timeout_seconds"]))
            except Exception as exc:
                logger.warning("Bucket listing failed for %s: %s", bucket_name, exc)
        bucket_manifest = build_bucket_manifest(bucket_rows, raw_dir)
    else:
        bucket_manifest_path = Path(config["artifact_paths"]["bucket_manifest"])
        bucket_manifest = pd.read_csv(bucket_manifest_path, sep="\t") if bucket_manifest_path.exists() else pd.DataFrame()
    bucket_manifest_path = Path(config["artifact_paths"]["bucket_manifest"])
    write_table(bucket_manifest, bucket_manifest_path)

    download_rows = []
    suffix_allowlist = {suffix.lower() for suffix in config["web"]["download_suffix_allowlist"]}
    max_auto_download_gb = float(config["web"]["max_auto_download_gb"])

    for row in manifest.to_dict(orient="records"):
        local_target = Path(row["local_target_path"])
        if offline_mode:
            status = "already_exists" if local_target.exists() else "offline_unavailable"
            result = {"status": status, "local_path": str(local_target)}
            metadata = {"final_url": row["remote_url"], "content_type": row.get("file_type", ""), "content_length": "", "etag": ""}
        else:
            if row["direct_download"]:
                if args.overwrite or not local_target.exists():
                    logger.info("Downloading %s", row["remote_url"])
                    result = stream_download(
                        session,
                        row["remote_url"],
                        local_target,
                        retries=config["web"]["retries"],
                        timeout=config["web"]["timeout_seconds"],
                    )
                else:
                    result = {"status": "already_exists", "local_path": str(local_target)}
                metadata = head_metadata(session, row["remote_url"], timeout=config["web"]["timeout_seconds"])
            else:
                result = {"status": "manual_download_needed", "local_path": str(local_target)}
                metadata = {"final_url": row["remote_url"], "content_type": "landing_page", "content_length": "", "etag": ""}
        download_rows.append({**row, **result, **metadata})

    for row in bucket_manifest.to_dict(orient="records"):
        local_target = Path(row["local_target_path"])
        can_download = row["file_type"] in suffix_allowlist and (
            config["web"]["allow_large_downloads"] or float(row["size_gb"]) <= max_auto_download_gb
        )
        if offline_mode:
            result = {"status": "already_exists" if local_target.exists() else "offline_unavailable", "local_path": str(local_target)}
        elif can_download:
            if args.overwrite or not local_target.exists():
                logger.info("Downloading bucket object %s", row["remote_url"])
                result = stream_download(
                    session,
                    row["remote_url"],
                    local_target,
                    retries=config["web"]["retries"],
                    timeout=config["web"]["timeout_seconds"],
                )
            else:
                result = {"status": "already_exists", "local_path": str(local_target)}
        else:
            reason = "deferred_by_download_policy"
            if row["file_type"] not in suffix_allowlist:
                reason = "unsupported_suffix"
            result = {"status": reason, "local_path": str(local_target)}
        download_rows.append({**row, **result, "final_url": row["remote_url"], "content_type": row["file_type"], "content_length": row["size_bytes"], "etag": ""})

    download_manifest = pd.DataFrame(download_rows)
    download_manifest_path = Path(config["artifact_paths"]["download_manifest"])
    write_table(download_manifest, download_manifest_path)

    snrna_primary = download_manifest.loc[
        download_manifest.get("modality", pd.Series(dtype=str)).astype(str).eq("snrna")
        & download_manifest["local_path"].astype(str).map(is_primary_snrna_object)
    ].copy()
    if not snrna_primary.empty:
        snrna_primary["region"] = snrna_primary["local_path"].map(infer_region_from_snrna_path)
        snrna_primary["donor_id"] = (
            snrna_primary["local_path"].astype(str).str.extract(r"(H[\d\.]+)_SEAAD_", expand=False).fillna("unknown")
        )
        snrna_primary["snrna_role"] = "primary_donor_rnaseq"
    primary_manifest_path = Path(config["artifact_paths"]["snrna_primary_manifest"])
    write_table(snrna_primary, primary_manifest_path)
    summary_path = manifest_dir / "seaad_manifest_summary_v1.tsv"
    manifest_summary = summarize_manifest(manifest)
    if not bucket_manifest.empty:
        bucket_summary = (
            bucket_manifest.groupby(["dataset_id", "modality"], dropna=False)
            .agg(n_bucket_objects=("remote_url", "size"), total_size_gb=("size_gb", "sum"))
            .reset_index()
        )
        manifest_summary = manifest_summary.merge(bucket_summary, on="dataset_id", how="outer")
    write_table(manifest_summary, summary_path)
    versions_path = Path(config["artifact_paths"]["software_versions"])
    collect_software_versions(["anndata", "scanpy", "numpy", "pandas", "torch"], versions_path)
    stage_status_path = Path(config["artifact_paths"]["stage_status_manifest"])
    append_stage_status(
        stage_status_path,
        "01_fetch_or_manifest_data",
        "success",
        "Built landing-page manifest, bucket object manifest, download-status manifest, and explicit primary snRNA input manifest." + (" Ran in offline manifest-reuse mode." if offline_mode else ""),
        n_inputs=len(links),
        n_outputs=len(download_manifest),
    )
    finalize_run(
        config,
        "01_fetch_or_manifest_data",
        args.config,
        [str(manifest_path), str(bucket_manifest_path), str(download_manifest_path), str(primary_manifest_path), str(summary_path), str(versions_path), str(stage_status_path)],
    )


if __name__ == "__main__":
    main()
