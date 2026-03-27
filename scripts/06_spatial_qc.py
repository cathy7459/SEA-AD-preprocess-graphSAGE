from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, write_h5ad, write_table
from src.path_audit_utils import audit_paths, check_path, require_existing_readable, require_writable_dir
from src.report_utils import append_stage_status
from src.spatial_utils import (
    discover_spatial_anomalies,
    discover_spatial_sections,
    filter_spatial_qc,
    load_xenium_section,
    read_metrics_summary,
    section_qc_summary,
    spatial_qc_table,
)


def zscore_flags(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["donor_id", "region", "section_id", "metric", "value", "zscore", "flag_reason"])
    rows: list[dict[str, object]] = []
    for metric in ["retained_fraction", "median_n_transcripts", "median_n_genes", "zero_transcript_fraction"]:
        if metric not in summary.columns:
            continue
        series = pd.to_numeric(summary[metric], errors="coerce")
        std = float(series.std(ddof=0)) if series.notna().any() else 0.0
        if std == 0 or pd.isna(std):
            continue
        zscores = (series - float(series.mean())) / std
        flagged = summary.loc[zscores.abs() >= threshold].copy()
        flagged["metric"] = metric
        flagged["value"] = series.loc[flagged.index].to_numpy()
        flagged["zscore"] = zscores.loc[flagged.index].to_numpy()
        flagged["flag_reason"] = f"|z| >= {threshold}"
        rows.extend(flagged[["donor_id", "region", "section_id", "metric", "value", "zscore", "flag_reason"]].to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser("Run section-wise spatial QC using Xenium cell feature matrices and save filtered objects.").parse_args()
    config, logger = init_run("06_spatial_qc", args.config)

    raw_dir = require_existing_readable(config["paths"]["raw_dir"], "raw_dir")
    out_dir = require_writable_dir(Path(config["paths"]["interim_dir"]) / "spatial_qc", "spatial_qc_output_dir")
    check_root = require_writable_dir(Path("results/check"), "results_check_dir")
    spatial_qc_check_dir = require_writable_dir(check_root / "spatial_qc", "spatial_qc_check_dir")
    distribution_dir = require_writable_dir(check_root / "spatial_distribution", "spatial_distribution_dir")
    confounder_dir = require_writable_dir(check_root / "confounders", "confounder_dir")
    graph_input_dir = require_writable_dir(check_root / "graph_inputs", "graph_input_dir")
    outlier_dir = require_writable_dir(check_root / "spatial_outlier", "spatial_outlier_dir")

    path_rows = [
        check_path(raw_dir, "raw_dir"),
        check_path(out_dir, "spatial_qc_output_dir"),
        check_path(spatial_qc_check_dir, "spatial_qc_check_dir"),
        check_path(distribution_dir, "spatial_distribution_dir"),
        check_path(confounder_dir, "confounder_dir"),
        check_path(graph_input_dir, "graph_input_dir"),
        check_path(outlier_dir, "spatial_outlier_dir"),
    ]
    path_audit_path = audit_paths(path_rows, spatial_qc_check_dir / "seaad_spatial_path_audit_v1.tsv")

    selected_regions = args.regions
    inventory = discover_spatial_sections(raw_dir, selected_regions=None)
    inventory_path = write_table(inventory, spatial_qc_check_dir / "seaad_spatial_section_inventory_v1.tsv")
    anomalies = discover_spatial_anomalies(raw_dir, inventory)
    anomaly_path = write_table(anomalies, outlier_dir / "seaad_spatial_input_anomalies_v1.tsv")

    outputs: list[str] = [str(path_audit_path), str(inventory_path), str(anomaly_path)]
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    if inventory.empty:
        append_stage_status(
            stage_status_path,
            "06_spatial_qc",
            "blocked",
            "No Xenium cell_feature_matrix sections were discoverable under data/raw/spatial_transcriptomics.",
            n_inputs=0,
            n_outputs=len(outputs),
        )
        finalize_run(config, "06_spatial_qc", args.config, outputs, "partial")
        return

    if selected_regions:
        inventory = inventory.loc[inventory["region"].isin(selected_regions) | inventory["region"].eq("unknown")].copy()
    if inventory.empty:
        message_path = write_table(
            pd.DataFrame(
                [{"reason": "Spatial sections were found, but all were excluded by the selected region filter.", "selected_regions": ",".join(selected_regions or [])}]
            ),
            spatial_qc_check_dir / "seaad_spatial_region_filter_block_v1.tsv",
        )
        outputs.append(str(message_path))
        append_stage_status(
            stage_status_path,
            "06_spatial_qc",
            "blocked",
            "Spatial sections exist but none matched the current region filter.",
            n_inputs=0,
            n_outputs=len(outputs),
        )
        finalize_run(config, "06_spatial_qc", args.config, outputs, "partial")
        return

    qc_summaries: list[pd.DataFrame] = []
    missingness_rows: list[dict[str, object]] = []
    graph_input_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for _, row in inventory.iterrows():
        section_root = Path(row["section_root"])
        logger.info(
            "Spatial section load start | donor=%s | region=%s | section=%s | root=%s",
            row["donor_id"],
            row["region"],
            row["section_id"],
            section_root,
        )
        required_files = [row["matrix_path"], row["features_path"], row["barcodes_path"], row["cells_path"]]
        for required_path in required_files:
            require_existing_readable(required_path, "spatial_input_file")

        try:
            adata = load_xenium_section(row)
        except Exception as exc:
            logger.exception("Spatial section load failed for %s", section_root)
            failure_rows.append(
                {
                    "section_root": str(section_root),
                    "donor_id": row["donor_id"],
                    "region": row["region"],
                    "section_id": row["section_id"],
                    "failure_stage": "load_xenium_section",
                    "failure_message": str(exc),
                }
            )
            continue

        metrics_summary = read_metrics_summary(row["metrics_path"]) if Path(row["metrics_path"]).exists() else {}
        qc = spatial_qc_table(
            adata,
            donor_id=str(row["donor_id"]),
            region=str(row["region"]),
            section_id=str(row["section_id"]),
            metrics_summary=metrics_summary,
        )
        qc["qc_pass"] = filter_spatial_qc(qc, config["qc"]["spatial"])
        summary = section_qc_summary(qc, config["qc"]["spatial"])
        summary["section_root"] = str(section_root)
        summary["cells_path"] = str(row["cells_path"])
        summary["matrix_path"] = str(row["matrix_path"])
        qc_summaries.append(summary)

        missingness_rows.append(
            {
                "donor_id": row["donor_id"],
                "region": row["region"],
                "section_id": row["section_id"],
                "missing_x_fraction": float(qc["x"].isna().mean()),
                "missing_y_fraction": float(qc["y"].isna().mean()),
                "missing_transcripts_fraction": float(qc["n_transcripts"].isna().mean()),
                "missing_genes_fraction": float(qc["n_genes"].isna().mean()),
            }
        )

        qc_path = out_dir / f"seaad_spatial_{row['region']}_{row['section_id']}_qc-metrics_{config['version']}.tsv"
        write_table(qc, qc_path)
        outputs.append(str(qc_path))

        filtered = adata[qc["qc_pass"].to_numpy()].copy()
        filtered.obs = qc.loc[qc["qc_pass"]].copy()
        filtered.uns["section_metrics_summary"] = metrics_summary
        filtered.uns["section_pass"] = bool(summary["section_pass"].iloc[0])
        out_path = out_dir / f"seaad_spatial_{row['region']}_{row['section_id']}_qc_{config['version']}.h5ad"
        write_h5ad(filtered, out_path)
        outputs.append(str(out_path))

        graph_input_rows.append(
            {
                "donor_id": row["donor_id"],
                "region": row["region"],
                "section_id": row["section_id"],
                "spatial_qc_h5ad": str(out_path),
                "n_cells_retained": int(filtered.n_obs),
                "n_genes": int(filtered.n_vars),
                "section_pass": bool(summary["section_pass"].iloc[0]),
            }
        )
        logger.info(
            "Spatial section QC complete | donor=%s | region=%s | section=%s | retained=%s/%s",
            row["donor_id"],
            row["region"],
            row["section_id"],
            int(filtered.n_obs),
            int(adata.n_obs),
        )

    summary_df = pd.concat(qc_summaries, ignore_index=True) if qc_summaries else pd.DataFrame()
    summary_path = write_table(summary_df, spatial_qc_check_dir / "seaad_spatial_section_qc_summary_v1.tsv")
    outputs.append(str(summary_path))

    donor_summary = (
        summary_df.groupby(["donor_id", "region"], dropna=False)
        .agg(
            n_sections=("section_id", "nunique"),
            n_cells_retained=("n_cells_retained", "sum"),
            median_section_retained_fraction=("retained_fraction", "median"),
        )
        .reset_index()
        if not summary_df.empty
        else pd.DataFrame(columns=["donor_id", "region", "n_sections", "n_cells_retained", "median_section_retained_fraction"])
    )
    donor_summary_path = write_table(donor_summary, distribution_dir / "seaad_spatial_donor_summary_v1.tsv")
    outputs.append(str(donor_summary_path))

    region_summary = (
        summary_df.groupby("region", dropna=False)
        .agg(
            n_sections=("section_id", "nunique"),
            n_donors=("donor_id", "nunique"),
            n_cells_retained=("n_cells_retained", "sum"),
            median_zero_transcript_fraction=("zero_transcript_fraction", "median"),
        )
        .reset_index()
        if not summary_df.empty
        else pd.DataFrame(columns=["region", "n_sections", "n_donors", "n_cells_retained", "median_zero_transcript_fraction"])
    )
    region_summary_path = write_table(region_summary, distribution_dir / "seaad_spatial_region_summary_v1.tsv")
    outputs.append(str(region_summary_path))

    missingness_path = write_table(pd.DataFrame(missingness_rows), spatial_qc_check_dir / "seaad_spatial_missingness_summary_v1.tsv")
    outputs.append(str(missingness_path))

    graph_inputs_path = write_table(pd.DataFrame(graph_input_rows), graph_input_dir / "seaad_spatial_graph_input_candidates_v1.tsv")
    outputs.append(str(graph_inputs_path))

    if not summary_df.empty:
        confounder_candidates = summary_df[
            [col for col in ["retained_fraction", "median_n_transcripts", "median_n_genes", "zero_transcript_fraction"] if col in summary_df.columns]
        ].copy()
        confounder_corr = confounder_candidates.corr(numeric_only=True).reset_index().rename(columns={"index": "feature"})
    else:
        confounder_corr = pd.DataFrame(columns=["feature"])
    confounder_path = write_table(confounder_corr, confounder_dir / "seaad_spatial_section_confounder_correlation_v1.tsv")
    outputs.append(str(confounder_path))

    outliers = zscore_flags(summary_df, float(config["qc"]["spatial"]["outlier_zscore_threshold"]))
    outlier_path = write_table(outliers, outlier_dir / "seaad_spatial_section_outliers_v1.tsv")
    outputs.append(str(outlier_path))

    failure_path = write_table(pd.DataFrame(failure_rows), outlier_dir / "seaad_spatial_section_failures_v1.tsv")
    outputs.append(str(failure_path))

    successful_sections = int(len(graph_input_rows))
    append_stage_status(
        stage_status_path,
        "06_spatial_qc",
        "success" if successful_sections else "blocked",
        "Spatial QC now uses Xenium cell_feature_matrix + cells tables as primary section inputs, with explicit path audits and anomaly reports.",
        n_inputs=int(len(inventory)),
        n_outputs=len(outputs),
    )
    finalize_run(config, "06_spatial_qc", args.config, outputs, "success" if successful_sections else "partial")


if __name__ == "__main__":
    main()
