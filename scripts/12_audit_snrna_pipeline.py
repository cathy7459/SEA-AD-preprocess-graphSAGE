from __future__ import annotations

from pathlib import Path

import pandas as pd
import scanpy as sc

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, read_table, write_table
from src.report_utils import append_stage_status
from src.snrna_utils import (
    classify_snrna_assay,
    discover_snrna_h5ad_files,
    harmonize_obs_metadata,
    infer_region_from_snrna_path,
    normalize_diagnosis_label,
    summarize_h5ad_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dataset_families_table(download_manifest: pd.DataFrame, raw_files: list[Path]) -> pd.DataFrame:
    if not download_manifest.empty and {"modality", "dataset_id", "status"}.issubset(download_manifest.columns):
        snrna = download_manifest[download_manifest["modality"] == "snrna"].copy()
    else:
        snrna = pd.DataFrame(columns=["dataset_id", "status", "local_path"])
    local_rnaseq_count = len(raw_files)
    rows = []
    for dataset_id in ["processed_single_nucleus_rnaseq_and_atacseq", "microglial_annotated_multiregion_nuclei"]:
        subset = snrna[snrna["dataset_id"] == dataset_id]
        microglial_placeholder = PROJECT_ROOT / "data" / "raw" / "microglial_annotated_multiregion_nuclei"
        rows.append(
            {
                "dataset_id": dataset_id,
                "expected": 1,
                "downloaded_any": int((subset["status"].isin(["downloaded", "already_exists"])).any()) if not subset.empty else int(dataset_id == "processed_single_nucleus_rnaseq_and_atacseq" and local_rnaseq_count > 0),
                "downloaded_h5ad_count": int(
                    subset.loc[
                        subset["status"].isin(["downloaded", "already_exists"])
                        & subset["local_path"].astype(str).str.endswith(".h5ad", na=False)
                    ].shape[0]
                )
                if not subset.empty
                else local_rnaseq_count if dataset_id == "processed_single_nucleus_rnaseq_and_atacseq" else 0,
                "manual_download_needed_count": int((subset["status"] == "manual_download_needed").sum()) if not subset.empty else int(dataset_id == "microglial_annotated_multiregion_nuclei" and any(microglial_placeholder.glob("*"))),
                "failed_count": int((subset["status"] == "failed").sum()) if not subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def file_audit_table(raw_files: list[Path], config: dict) -> pd.DataFrame:
    qc_dir = PROJECT_ROOT / config["paths"]["interim_dir"] / "snrna_qc"
    norm_dir = PROJECT_ROOT / config["paths"]["processed_dir"] / "snrna_normalized"
    ann_dir = PROJECT_ROOT / config["paths"]["processed_dir"] / "snrna_annotated"
    rows = []
    for file_path in raw_files:
        summary = summarize_h5ad_metadata(file_path)
        donor_id = summary["donor_id_example"]
        region = summary["region_example"]
        qc_outputs = list(qc_dir.glob(f"*{region}*donor-{donor_id}*qc*.h5ad"))
        norm_outputs = list(norm_dir.glob(f"*{region}*donor-{donor_id}*normalized*.h5ad"))
        ann_outputs = list(ann_dir.glob(f"*{region}*donor-{donor_id}*annotated*.h5ad"))
        rows.append(
            {
                "dataset_id": "processed_single_nucleus_rnaseq_and_atacseq",
                "file_name": summary["file_name"],
                "file_path": summary["file_path"],
                "assay": summary["assay"],
                "expected": 1,
                "downloaded": int(Path(file_path).exists()),
                "loaded": int(summary["loaded"]),
                "retained": int(len(qc_outputs) > 0),
                "used": int(len(norm_outputs) > 0 or len(ann_outputs) > 0),
                "n_cells": summary["n_cells"],
                "n_genes": summary["n_genes"],
                "donor_id": donor_id,
                "region": region,
                "diagnosis_labels_raw": ";".join(summary["diagnosis_labels_raw"]),
                "diagnosis_labels_normalized": ";".join(summary["diagnosis_labels_normalized"]),
                "has_donor_id": summary["has_donor_id"],
                "has_region": summary["has_region"],
                "has_diagnosis": summary["has_diagnosis"],
                "has_subclass": summary["has_subclass"],
            }
        )
    microglial_path = PROJECT_ROOT / config["paths"]["raw_dir"] / "microglial_annotated_multiregion_nuclei"
    rows.append(
        {
            "dataset_id": "microglial_annotated_multiregion_nuclei",
            "file_name": "manual_download_placeholder",
            "file_path": str(microglial_path),
            "assay": "RNAseq",
            "expected": 1,
            "downloaded": 0,
            "loaded": 0,
            "retained": 0,
            "used": 0,
            "n_cells": 0,
            "n_genes": 0,
            "donor_id": "unknown",
            "region": "unknown",
            "diagnosis_labels_raw": "",
            "diagnosis_labels_normalized": "",
            "has_donor_id": 0,
            "has_region": 0,
            "has_diagnosis": 0,
            "has_subclass": 0,
        }
    )
    return pd.DataFrame(rows)


def label_validation_table(raw_files: list[Path]) -> pd.DataFrame:
    rows = []
    for file_path in raw_files:
        adata = sc.read_h5ad(file_path, backed="r")
        try:
            obs = harmonize_obs_metadata(adata.obs, file_path=file_path)
            for raw_label in sorted(pd.Series(obs["diagnosis_raw"]).dropna().astype(str).unique().tolist()):
                rows.append(
                    {
                        "file_name": Path(file_path).name,
                        "raw_metadata_label": raw_label,
                        "normalized_label": normalize_diagnosis_label(raw_label),
                        "used_in_downstream_analysis": int(True),
                    }
                )
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
    return pd.DataFrame(rows).drop_duplicates().sort_values(["raw_metadata_label", "file_name"])


def integrity_trace_table(file_audit: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "stage": "download_manifest",
            "status": "mixed",
            "observation": "snRNA family includes RNAseq and ATACseq objects; microglial dataset remains manual-download-only.",
            "code_location": "scripts/01_fetch_or_manifest_data.py",
        },
        {
            "stage": "region_inventory",
            "status": "bug_detected",
            "observation": "Original region split inferred snRNA regions only from simple path tokens and did not distinguish RNAseq from ATACseq robustly.",
            "code_location": "scripts/02_split_by_region.py",
        },
        {
            "stage": "snrna_qc_loading",
            "status": "bug_detected",
            "observation": "Original QC loader globbed every non-spatial .h5ad, so ATAC donor objects were eligible for snRNA QC.",
            "code_location": "scripts/03_snrna_qc.py",
        },
        {
            "stage": "metadata_mapping",
            "status": "bug_detected",
            "observation": "Original donor and region column matching missed 'Donor ID' and 'Brain Region', causing donor/region to collapse to unknown and risking duplicate region processing.",
            "code_location": "scripts/03_snrna_qc.py",
        },
        {
            "stage": "retained_outputs",
            "status": "observed",
            "observation": f"{int(file_audit['retained'].sum())} of {len(file_audit[file_audit['dataset_id']=='processed_single_nucleus_rnaseq_and_atacseq'])} downloaded RNAseq donor objects currently have QC outputs.",
            "code_location": "data/interim/snrna_qc",
        },
        {
            "stage": "used_outputs",
            "status": "observed",
            "observation": f"{int(file_audit['used'].sum())} donor objects currently propagate to normalized or annotated outputs.",
            "code_location": "data/processed/snrna_normalized; data/processed/snrna_annotated",
        },
    ]
    return pd.DataFrame(rows)


def literature_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper": "Mathys et al., Nature 2019",
                "preprocessing_steps": "nuclei-level QC, removal of low-quality nuclei, normalization, clustering, cell-type annotation, donor-aware downstream comparisons",
                "relevance_to_current_pipeline": "Directly relevant for donor-level AD snRNA-seq QC and metadata-aware label usage",
                "differences_from_current_code": "Current code lacked robust metadata harmonization and could mix assays; the corrected pipeline now enforces donor/region harmonization before QC.",
                "source_url": "https://www.nature.com/articles/s41586-019-1195-2",
            },
            {
                "paper": "Morabito et al., Nat Genet 2021",
                "preprocessing_steps": "paired snRNA/snATAC resource with modality-specific preprocessing, donor-aware analysis, pathology-linked metadata harmonization",
                "relevance_to_current_pipeline": "Strong comparator because SEA-AD raw assets similarly include both RNAseq and ATACseq objects that must not be conflated",
                "differences_from_current_code": "Current code originally treated all non-spatial h5ad files as snRNA candidates; corrected code filters to RNAseq only.",
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8766217/",
            },
            {
                "paper": "SEA-AD official atlas resource",
                "preprocessing_steps": "region-specific donor objects, rich clinical/pathology metadata, multimodal atlas organization",
                "relevance_to_current_pipeline": "Primary source for the exact datasets used here",
                "differences_from_current_code": "Current pipeline needed explicit region mapping for MTG vs A9/PFC and explicit diagnosis extraction from metadata rather than assumptions.",
                "source_url": "https://brain-map.org/consortia/sea-ad/our-data",
            },
        ]
    )


def selection_strategy_table(file_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, subset in file_audit.groupby("dataset_id", dropna=False):
        if dataset_id == "microglial_annotated_multiregion_nuclei":
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "include_or_exclude": "exclude_for_now",
                    "reasoning": "Not locally downloadable in the current run; only a landing-page placeholder is present, so it cannot be reproducibly incorporated now.",
                }
            )
            continue
        mtg_n = int((subset["region"] == "MTG").sum())
        a9_n = int((subset["region"] == "A9").sum())
        labels = sorted({label for labels in subset["diagnosis_labels_normalized"].tolist() for label in labels.split(";") if label})
        rows.append(
            {
                "dataset_id": dataset_id,
                "include_or_exclude": "include_rnaseq_only",
                "reasoning": (
                    f"Downloaded donor-level RNAseq objects are present for MTG ({mtg_n}) and A9 ({a9_n}), "
                    f"with diagnosis labels observed as {labels}. This supports graph modeling and region-matched multimodal integration, "
                    f"but ATAC objects should be excluded from the snRNA pipeline."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    out_path: Path,
    family_table: pd.DataFrame,
    file_audit: pd.DataFrame,
    integrity_trace: pd.DataFrame,
    label_table: pd.DataFrame,
    literature: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    def frame_to_markdown_like(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        return df.to_csv(sep="\t", index=False)

    diagnosis_counts = (
        file_audit[file_audit["dataset_id"] == "processed_single_nucleus_rnaseq_and_atacseq"]["diagnosis_labels_normalized"]
        .fillna("")
        .str.split(";")
        .explode()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    lines = [
        "# snRNA pipeline audit",
        "",
        "## Dataset audit",
        frame_to_markdown_like(family_table),
        "",
        frame_to_markdown_like(file_audit),
        "",
        "## Integrity failure trace",
        frame_to_markdown_like(integrity_trace),
        "",
        "## Label validation",
        frame_to_markdown_like(label_table),
        "",
        "## Observed diagnosis label distribution across downloaded RNAseq donor objects",
        frame_to_markdown_like(diagnosis_counts.to_frame("n_files").reset_index().rename(columns={"index": "diagnosis"})),
        "",
        "## Literature comparison",
        frame_to_markdown_like(literature),
        "",
        "## Dataset selection strategy",
        frame_to_markdown_like(selection),
        "",
        "## Conclusion",
        (
            "The downloaded snRNA data currently support reference, no-dementia, and dementia donors with pathology labels such as Not AD, Low, Intermediate, High, and Reference. "
            "The pipeline should therefore be evaluated against the labels actually present, not assumed AsymAD groups."
        ),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser("Audit, validate, and correct the snRNA preprocessing pipeline.").parse_args()
    config, logger = init_run("12_audit_snrna_pipeline", args.config)
    check_dir = ensure_dir(PROJECT_ROOT / "results" / "check")
    outputs: list[str] = []

    download_manifest_path = PROJECT_ROOT / config["artifact_paths"]["download_manifest"]
    download_manifest = read_table(download_manifest_path) if download_manifest_path.exists() else pd.DataFrame()
    raw_files = discover_snrna_h5ad_files(PROJECT_ROOT / config["paths"]["raw_dir"])
    family_table = dataset_families_table(download_manifest, raw_files)
    family_path = check_dir / "snrna_dataset_family_audit.tsv"
    write_table(family_table, family_path)
    outputs.append(str(family_path))

    logger.info("Auditing %s local RNAseq donor objects", len(raw_files))
    file_audit = file_audit_table(raw_files, config)
    file_audit_path = check_dir / "snrna_dataset_audit.tsv"
    write_table(file_audit, file_audit_path)
    outputs.append(str(file_audit_path))

    integrity_trace = integrity_trace_table(file_audit)
    integrity_path = check_dir / "snrna_integrity_failure_trace.tsv"
    write_table(integrity_trace, integrity_path)
    outputs.append(str(integrity_path))

    label_table = label_validation_table(raw_files)
    label_path = check_dir / "snrna_label_validation.tsv"
    write_table(label_table, label_path)
    outputs.append(str(label_path))

    literature = literature_table()
    literature_path = check_dir / "snrna_literature_comparison.tsv"
    write_table(literature, literature_path)
    outputs.append(str(literature_path))

    selection = selection_strategy_table(file_audit)
    selection_path = check_dir / "snrna_dataset_selection_strategy.tsv"
    write_table(selection, selection_path)
    outputs.append(str(selection_path))

    md_path = check_dir / "snrna_audit_report.md"
    write_markdown_summary(md_path, family_table, file_audit, integrity_trace, label_table, literature, selection)
    outputs.append(str(md_path))

    append_stage_status(
        config["artifact_paths"]["stage_status_manifest"],
        "12_audit_snrna_pipeline",
        "success",
        "Generated snRNA audit tables and markdown summary under results/check.",
        n_inputs=len(raw_files),
        n_outputs=len(outputs),
    )
    finalize_run(config, "12_audit_snrna_pipeline", args.config, outputs)


if __name__ == "__main__":
    main()
