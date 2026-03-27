from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, list_files, read_table, write_table
from src.patient_utils import aggregate_patient_features, cell_type_fraction_table
from src.report_utils import append_stage_status


def main() -> None:
    args = build_parser("Aggregate donor-region summaries into patient-level feature tables.").parse_args()
    config, logger = init_run("09_aggregate_patient_features", args.config)
    graph_dir = Path(config["paths"]["processed_dir"]) / "graphs"
    out_dir = ensure_dir(Path(config["paths"]["processed_dir"]) / "patient_features")
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]
    snrna_norm_dir = Path(config["paths"]["processed_dir"]) / "snrna_normalized"
    overlap_table_path = Path(config["artifact_paths"]["overlap_table"])

    summary_files = [p for p in list_files(graph_dir, (".tsv",)) if "summary" in p.name]
    meta_files = [p for p in list_files(graph_dir, (".parquet",)) if "node-metadata" in p.name]
    if not summary_files and not meta_files:
        status_path = out_dir / "seaad_patient_aggregation_no_inputs_v1.tsv"
        write_table(pd.DataFrame([{"reason": "No graph summaries or node metadata were available for patient aggregation."}]), status_path)
        append_stage_status(
            stage_status_path,
            "09_aggregate_patient_features",
            "blocked",
            "Patient aggregation skipped because graph outputs were missing.",
            n_inputs=0,
            n_outputs=1,
        )
        finalize_run(config, "09_aggregate_patient_features", args.config, [str(status_path)], "partial")
        return
    graph_summaries = pd.concat([read_table(p) for p in summary_files], ignore_index=True) if summary_files else pd.DataFrame()
    fraction_frames = []
    for meta_path in meta_files:
        meta = read_table(meta_path)
        donor = str(meta.get("donor_id", pd.Series(["donor-unknown"])).iloc[0])
        region = str(meta.get("region", pd.Series(["unknown"])).iloc[0])
        frac = cell_type_fraction_table(meta, donor_id=donor, region=region)
        if not frac.empty:
            fraction_frames.append(frac)
    fractions = pd.concat(fraction_frames, ignore_index=True) if fraction_frames else pd.DataFrame()
    spatial_patch_feature_rows = []
    for meta_path in meta_files:
        meta = read_table(meta_path)
        fraction_cols = [col for col in meta.columns if col.startswith("celltype_fraction__")]
        if fraction_cols and {"donor_id", "region"}.issubset(meta.columns):
            donor = str(meta["donor_id"].iloc[0])
            region = str(meta["region"].iloc[0])
            averaged = meta[fraction_cols].mean(axis=0)
            row = {"donor_id": donor, "region": region}
            row.update({f"spatial_patch__{col}": float(val) for col, val in averaged.items()})
            spatial_patch_feature_rows.append(row)

    mapping_files = [p for p in list_files(Path(config["paths"]["processed_dir"]) / "spatial_mapping", (".tsv",))]
    mapping_summary = None
    if mapping_files:
        mapping = pd.concat([read_table(p) for p in mapping_files], ignore_index=True)
        if not mapping.empty and "mapped_confidence" in mapping.columns:
            mapping["donor_id"] = mapping.get("donor_id", "donor-unknown")
            mapping["region"] = mapping.get("region", "unknown")
            mapping_summary = (
                mapping.groupby(["donor_id", "region"], dropna=False)["mapped_confidence"]
                .mean()
                .reset_index(name="mapped_confidence_mean")
            )
    patient_features = aggregate_patient_features(graph_summaries, fractions, mapping_summary)
    overlap_note_outputs: list[str] = []
    if config["analysis"].get("keep_only_multimodal_overlap", False) and overlap_table_path.exists() and not patient_features.empty:
        overlap_df = read_table(overlap_table_path)
        if not overlap_df.empty and {"donor_id", "region", "modality", "available"}.issubset(overlap_df.columns):
            available = overlap_df.loc[pd.to_numeric(overlap_df["available"], errors="coerce").fillna(0).astype(int) == 1].copy()
            strict_pairs = (
                available.groupby(["donor_id", "region"], dropna=False)["modality"].nunique().reset_index(name="n_modalities")
            )
            strict_pairs = strict_pairs.loc[strict_pairs["n_modalities"] >= 2]
            if not strict_pairs.empty:
                eligible_donors = sorted(strict_pairs["donor_id"].astype(str).unique())
                patient_features = patient_features.loc[patient_features["donor_id"].astype(str).isin(eligible_donors)].copy()
                overlap_note = pd.DataFrame([{"mode": "strict_donor_region_overlap", "n_eligible_donors": len(eligible_donors)}])
            else:
                donor_level = available.groupby("donor_id", dropna=False)["modality"].nunique().reset_index(name="n_modalities")
                donor_level = donor_level.loc[donor_level["n_modalities"] >= 2]
                if not donor_level.empty:
                    eligible_donors = sorted(donor_level["donor_id"].astype(str).unique())
                    patient_features = patient_features.loc[patient_features["donor_id"].astype(str).isin(eligible_donors)].copy()
                    overlap_note = pd.DataFrame(
                        [{"mode": "fallback_donor_level_overlap", "n_eligible_donors": len(eligible_donors), "note": "No strict donor-region multimodal overlap was present; retained donors with both modalities across any region."}]
                    )
                else:
                    overlap_note = pd.DataFrame(
                        [{"mode": "no_multimodal_overlap_found", "n_eligible_donors": 0, "note": "No multimodal overlap was found; retained all donors to avoid silently producing an empty patient matrix."}]
                    )
            overlap_note_path = out_dir / f"seaad_patient_overlap_selection_note_{config['version']}.tsv"
            write_table(overlap_note, overlap_note_path)
            overlap_note_outputs.append(str(overlap_note_path))
    cell_feature_files = [p for p in list_files(snrna_norm_dir, (".parquet",)) if "cell-features" in p.name]
    if cell_feature_files:
        cell_features = pd.concat([read_table(p) for p in cell_feature_files], ignore_index=True)
        pca_cols = [c for c in cell_features.columns if c.startswith("pca_hvg_")]
        if pca_cols and {"donor_id", "region"}.issubset(cell_features.columns):
            region_means = (
                cell_features.groupby(["donor_id", "region"], dropna=False)[pca_cols]
                .mean()
                .reset_index()
            )
            wide = region_means.pivot_table(index="donor_id", columns="region", values=pca_cols, fill_value=0)
            wide.columns = [f"{col[0]}__{col[1]}" for col in wide.columns]
            patient_features = patient_features.merge(wide.reset_index(), on="donor_id", how="outer")
    if spatial_patch_feature_rows:
        spatial_patch_features = pd.DataFrame(spatial_patch_feature_rows)
        wide = spatial_patch_features.pivot_table(index="donor_id", columns="region", values=[col for col in spatial_patch_features.columns if col.startswith("spatial_patch__")], fill_value=0)
        wide.columns = [f"{col[0]}__{col[1]}" for col in wide.columns]
        patient_features = patient_features.merge(wide.reset_index(), on="donor_id", how="outer")
    feat_path = out_dir / f"seaad_patient_graphready_features_{config['version']}.parquet"
    legacy_feat_path = out_dir / f"seaad_patient_overlap{len(config['analysis']['default_regions'])}regions_features_{config['version']}.parquet"
    write_table(patient_features, feat_path)
    write_table(patient_features, legacy_feat_path)
    frac_path = out_dir / f"seaad_patient_cell_type_fractions_{config['version']}.tsv"
    write_table(fractions, frac_path)
    sum_path = out_dir / f"seaad_patient_graph_summaries_{config['version']}.tsv"
    write_table(graph_summaries, sum_path)
    append_stage_status(
        stage_status_path,
        "09_aggregate_patient_features",
        "success",
        "Aggregated graph and composition summaries into patient-level features.",
        n_inputs=len(summary_files) + len(meta_files),
        n_outputs=4 + len(overlap_note_outputs),
    )
    finalize_run(config, "09_aggregate_patient_features", args.config, [str(feat_path), str(legacy_feat_path), str(frac_path), str(sum_path), *overlap_note_outputs])


if __name__ == "__main__":
    main()
