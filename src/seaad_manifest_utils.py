from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.naming import sanitize_token
from src.web_fetch_utils import choose_downloadable_links


def classify_dataset(title: str, datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = title.lower()
    for dataset in datasets:
        keywords = [kw.lower() for kw in dataset.get("expected_keywords", [])]
        if any(keyword in lowered for keyword in keywords):
            return dataset
    return None


def build_remote_manifest(
    link_records: list[dict[str, str]],
    datasets: list[dict[str, Any]],
    raw_dir: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    downloadable_urls = {row["url"] for row in choose_downloadable_links(link_records)}
    for record in link_records:
        dataset = classify_dataset(record["title"], datasets)
        if dataset is None:
            continue
        extension = Path(record["url"]).suffix.lstrip(".") or "html"
        local_name = f"{sanitize_token(dataset['dataset_id'])}_{sanitize_token(record['title'])}.{extension}"
        rows.append(
            {
                "dataset_id": dataset["dataset_id"],
                "display_name": dataset["display_name"],
                "modality": dataset["modality"],
                "source_page_url": record["source_page"],
                "link_title": record["title"],
                "remote_url": record["url"],
                "file_type": extension,
                "direct_download": int(record["url"] in downloadable_urls),
                "manual_download_needed": int(record["url"] not in downloadable_urls),
                "local_target_path": str(Path(raw_dir) / dataset["dataset_id"] / local_name),
            }
        )
    manifest = pd.DataFrame(rows).drop_duplicates()
    return manifest.sort_values(["dataset_id", "link_title"]).reset_index(drop=True)


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["dataset_id", "n_links", "n_direct", "n_manual"])
    summary = (
        manifest.groupby("dataset_id", dropna=False)
        .agg(
            n_links=("remote_url", "size"),
            n_direct=("direct_download", "sum"),
            n_manual=("manual_download_needed", "sum"),
        )
        .reset_index()
    )
    return summary


def infer_dataset_from_bucket(bucket_name: str) -> dict[str, str]:
    mapping = {
        "sea-ad-single-cell-profiling": {
            "dataset_id": "processed_single_nucleus_rnaseq_and_atacseq",
            "display_name": "SEA-AD processed single nucleus RNAseq and ATAC-seq",
            "modality": "snrna",
        },
        "sea-ad-spatial-transcriptomics": {
            "dataset_id": "spatial_transcriptomics",
            "display_name": "SEA-AD spatial transcriptomics",
            "modality": "spatial",
        },
        "sea-ad-quantitative-neuropathology": {
            "dataset_id": "quantitative_neuropathology",
            "display_name": "SEA-AD quantitative neuropathology",
            "modality": "pathology",
        },
    }
    return mapping[bucket_name]


def build_bucket_manifest(bucket_rows: list[dict[str, str]], raw_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for item in bucket_rows:
        dataset_meta = infer_dataset_from_bucket(item["bucket_name"])
        key = item["key"]
        local_target = Path(raw_dir) / dataset_meta["dataset_id"] / key
        rows.append(
            {
                "dataset_id": dataset_meta["dataset_id"],
                "display_name": dataset_meta["display_name"],
                "modality": dataset_meta["modality"],
                "bucket_name": item["bucket_name"],
                "remote_url": item["remote_url"],
                "object_key": key,
                "size_bytes": item["size_bytes"],
                "size_gb": round(item["size_bytes"] / (1024**3), 6),
                "last_modified": item["last_modified"],
                "file_type": Path(key).suffix.lower(),
                "local_target_path": str(local_target),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "display_name",
                "modality",
                "bucket_name",
                "remote_url",
                "object_key",
                "size_bytes",
                "size_gb",
                "last_modified",
                "file_type",
                "local_target_path",
            ]
        )
    manifest = pd.DataFrame(rows).sort_values(["dataset_id", "object_key"]).reset_index(drop=True)
    return manifest
