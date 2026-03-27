from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, list_files, read_table, write_table
from src.plotting import barplot, boxplot_by_region, heatmap, scatter_spatial, stacked_bar, violin_by_region
from src.report_utils import append_stage_status

sns.set_theme(style="whitegrid", context="talk")


def save_scatter(df: pd.DataFrame, x: str, y: str, hue: str, out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    n_groups = df[hue].nunique(dropna=False)
    legend = n_groups <= 20
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=10, linewidth=0, alpha=0.7, ax=ax, legend=legend)
    ax.set_title(title)
    if legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_table(df, out_base.with_suffix(".tsv"))


def interpret_embedding(coords: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = coords.dropna(subset=["embed_1", "embed_2"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["label", "n_groups", "silhouette", "interpretation"])
    eval_df = usable.sample(min(len(usable), 5000), random_state=1729) if len(usable) > 5000 else usable
    xy = eval_df[["embed_1", "embed_2"]].to_numpy()
    for column in label_columns:
        if column not in eval_df.columns:
            continue
        labels = eval_df[column].fillna("missing").astype(str)
        n_groups = labels.nunique()
        if n_groups < 2 or n_groups >= len(eval_df):
            score = np.nan
        else:
            score = float(silhouette_score(xy, labels))
        interpretation = "not_evaluable"
        if pd.notna(score):
            if score >= 0.35:
                interpretation = "strong_separation"
            elif score >= 0.15:
                interpretation = "moderate_separation"
            else:
                interpretation = "weak_or_mixed_signal"
        rows.append({"label": column, "n_groups": int(n_groups), "silhouette": score, "interpretation": interpretation})
    return pd.DataFrame(rows)


def build_spatial_umap(graph_dir: Path, out_dir: Path, seed: int, donor_target: int, max_nodes: int) -> list[str]:
    meta_files = [p for p in list_files(graph_dir, (".parquet",)) if "_spatial_node-metadata_" in p.name]
    feat_files = [p for p in list_files(graph_dir, (".parquet",)) if "_spatial_node-features_" in p.name]
    if not meta_files or not feat_files:
        message_path = write_table(
            pd.DataFrame([{"reason": "Spatial graph node metadata/features were unavailable for UMAP diagnostics."}]),
            out_dir / "seaad_spatial_umap_missing_inputs_v1.tsv",
        )
        return [str(message_path)]

    meta = pd.concat([read_table(path) for path in meta_files], ignore_index=True)
    features = pd.concat([read_table(path) for path in feat_files], ignore_index=True)
    merged = meta.merge(features, on="node_id", how="inner")
    donors = sorted(merged["donor_id"].dropna().astype(str).unique())
    rng = np.random.default_rng(seed)
    chosen = donors if len(donors) <= donor_target else sorted(rng.choice(donors, size=donor_target, replace=False).tolist())
    chosen_path = write_table(pd.DataFrame({"donor_id": chosen}), out_dir / "seaad_spatial_umap_selected_donors_v1.tsv")

    merged = merged.loc[merged["donor_id"].astype(str).isin(chosen)].copy()
    if len(merged) > max_nodes:
        merged = merged.sample(max_nodes, random_state=seed).reset_index(drop=True)

    feature_cols = [col for col in merged.columns if col.startswith("spatial_feat_")]
    if not feature_cols:
        message_path = write_table(
            pd.DataFrame([{"reason": "Spatial node feature matrix did not contain spatial_feat_* columns."}]),
            out_dir / "seaad_spatial_umap_missing_feature_columns_v1.tsv",
        )
        return [str(chosen_path), str(message_path)]

    scaler = StandardScaler(with_mean=True, with_std=True)
    matrix = scaler.fit_transform(merged[feature_cols].fillna(0).to_numpy(dtype=float))
    method = "umap"
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.3, metric="euclidean")
        embedding = reducer.fit_transform(matrix)
    except Exception as exc:
        method = f"pca_fallback: {exc.__class__.__name__}"
        reducer = PCA(n_components=2, random_state=seed)
        embedding = reducer.fit_transform(matrix)

    merged["embed_1"] = embedding[:, 0]
    merged["embed_2"] = embedding[:, 1]
    coords_path = write_table(
        merged[["node_id", "donor_id", "region", "section_id", "embed_1", "embed_2"] + [col for col in ["diagnosis", "mapped_confidence_mean"] if col in merged.columns]],
        out_dir / "seaad_spatial_umap_coordinates_v1.tsv",
    )
    metadata_path = write_table(merged.drop(columns=feature_cols), out_dir / "seaad_spatial_umap_metadata_v1.tsv")
    feature_summary_path = write_table(
        pd.DataFrame(
            [
                {
                    "n_selected_donors": len(chosen),
                    "n_nodes_used": len(merged),
                    "n_feature_columns": len(feature_cols),
                    "embedding_method": method,
                }
            ]
        ),
        out_dir / "seaad_spatial_umap_feature_summary_v1.tsv",
    )

    outputs = [str(chosen_path), str(coords_path), str(metadata_path), str(feature_summary_path)]
    for hue, title_suffix in [
        ("section_id", "Section ID"),
        ("donor_id", "Donor ID"),
        ("region", "Region"),
        ("diagnosis", "Diagnosis"),
    ]:
        if hue not in merged.columns:
            continue
        out_base = out_dir / f"seaad_spatial_umap_by_{hue}_v1"
        save_scatter(merged[["embed_1", "embed_2", hue]].copy(), "embed_1", "embed_2", hue, out_base, f"Spatial embedding colored by {title_suffix}")
        outputs.extend([str(out_base.with_suffix(".png")), str(out_base.with_suffix(".pdf")), str(out_base.with_suffix(".tsv"))])

    interpretation = interpret_embedding(merged, ["section_id", "donor_id", "region", "diagnosis"])
    interpretation["embedding_method"] = method
    if not interpretation.empty and interpretation["silhouette"].notna().any():
        best = interpretation.sort_values("silhouette", ascending=False, na_position="last").iloc[0]
        interpretation["dominant_signal"] = best["label"]
    interpretation_path = write_table(interpretation, out_dir / "seaad_spatial_umap_interpretation_v1.tsv")
    outputs.append(str(interpretation_path))
    return outputs


def main() -> None:
    args = build_parser("Generate publication-style QC, graph-input, and spatial UMAP diagnostics.").parse_args()
    config, logger = init_run("11_qc_report_and_plots", args.config)
    fig_dir = ensure_dir(config["paths"]["figure_dir"])
    check_root = ensure_dir("results/check")
    spatial_qc_dir = ensure_dir(check_root / "spatial_qc")
    spatial_distribution_dir = ensure_dir(check_root / "spatial_distribution")
    confounder_dir = ensure_dir(check_root / "confounders")
    graph_input_dir = ensure_dir(check_root / "graph_inputs")
    outlier_dir = ensure_dir(check_root / "spatial_outlier")
    spatial_umap_dir = ensure_dir(check_root / "spatial_UMAP")

    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    snrna_tables = [read_table(p) for p in list_files(Path(config["paths"]["interim_dir"]) / "snrna_qc", (".tsv",)) if "qc-metrics" in p.name]
    if snrna_tables:
        snrna_qc = pd.concat(snrna_tables, ignore_index=True)
        violin_by_region(snrna_qc, "n_genes", Path(fig_dir) / "snrna_n_genes_by_region", "snRNA n_genes by region")
        violin_by_region(snrna_qc, "n_counts", Path(fig_dir) / "snrna_n_counts_by_region", "snRNA n_counts by region")
        violin_by_region(snrna_qc, "pct_mt", Path(fig_dir) / "snrna_pct_mt_by_region", "snRNA pct_mt by region")
        donor_retained = snrna_qc.groupby(["donor_id", "region"], dropna=False)["qc_pass"].sum().reset_index(name="cells_retained")
        boxplot_by_region(donor_retained, "cells_retained", Path(fig_dir) / "snrna_cells_retained_by_region", "Donor-wise retained snRNA cells")
        outputs.extend([str(Path(fig_dir) / name) for name in ["snrna_n_genes_by_region", "snrna_n_counts_by_region", "snrna_pct_mt_by_region", "snrna_cells_retained_by_region"]])

    spatial_summary_path = spatial_qc_dir / "seaad_spatial_section_qc_summary_v1.tsv"
    if spatial_summary_path.exists():
        spatial_summary = read_table(spatial_summary_path)
        if not spatial_summary.empty:
            boxplot_by_region(spatial_summary, "median_n_transcripts", spatial_distribution_dir / "seaad_spatial_median_transcripts_by_region_v1", "Spatial median transcripts by region")
            boxplot_by_region(spatial_summary, "median_n_genes", spatial_distribution_dir / "seaad_spatial_median_genes_by_region_v1", "Spatial median genes by region")
            boxplot_by_region(spatial_summary, "retained_fraction", spatial_distribution_dir / "seaad_spatial_retained_fraction_by_region_v1", "Spatial retained fraction by region")
            outputs.extend(
                [
                    str(spatial_distribution_dir / "seaad_spatial_median_transcripts_by_region_v1"),
                    str(spatial_distribution_dir / "seaad_spatial_median_genes_by_region_v1"),
                    str(spatial_distribution_dir / "seaad_spatial_retained_fraction_by_region_v1"),
                ]
            )

    mapping_files = [read_table(p) for p in list_files(Path(config["paths"]["processed_dir"]) / "spatial_mapping", (".tsv",))]
    if mapping_files:
        mapping = pd.concat(mapping_files, ignore_index=True)
        if "mapped_confidence" in mapping.columns and not mapping.empty:
            if "region" not in mapping.columns:
                mapping["region"] = "unknown"
            boxplot_by_region(mapping, "mapped_confidence", fig_dir / "spatial_mapped_confidence_by_region", "Mapped confidence by region")
            outputs.append(str(fig_dir / "spatial_mapped_confidence_by_region"))

    fraction_path = Path(config["paths"]["processed_dir"]) / "patient_features" / f"seaad_patient_cell_type_fractions_{config['version']}.tsv"
    if fraction_path.exists():
        fractions = read_table(fraction_path)
        if not fractions.empty:
            stacked_bar(fractions, fig_dir / "cell_type_composition_by_donor_region", "Cell-type composition by donor and region")
            heatmap(fractions, "region", "major_cell_class", "fraction", fig_dir / "region_cell_type_heatmap", "Region-by-cell-type mean fraction")
            outputs.extend([str(fig_dir / "cell_type_composition_by_donor_region"), str(fig_dir / "region_cell_type_heatmap")])

    graph_summary_files = [read_table(p) for p in list_files(Path(config["paths"]["processed_dir"]) / "graphs", (".tsv",)) if "_summary_" in p.name]
    if graph_summary_files:
        graph_summaries = pd.concat(graph_summary_files, ignore_index=True)
        graph_summary_path = write_table(graph_summaries, graph_input_dir / "seaad_graph_input_summary_v1.tsv")
        heatmap(graph_summaries, "donor_id", "region", "n_nodes", graph_input_dir / "seaad_graph_nodes_heatmap_v1", "Donor-region graph size summary")
        outputs.extend([str(graph_summary_path), str(graph_input_dir / "seaad_graph_nodes_heatmap_v1")])

    patient_feature_dir = Path(config["paths"]["processed_dir"]) / "patient_features"
    patient_feature_path = patient_feature_dir / f"seaad_patient_graphready_features_{config['version']}.parquet"
    if patient_feature_path.exists():
        patient_features = read_table(patient_feature_path)
        if not patient_features.empty:
            long_df = patient_features.melt(id_vars="donor_id", var_name="feature", value_name="value")
            long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
            long_df = long_df.dropna(subset=["value"])
            heatmap(long_df, "donor_id", "feature", "value", fig_dir / "patient_feature_heatmap", "Patient-level feature heatmap")
            outputs.append(str(fig_dir / "patient_feature_heatmap"))

    spatial_meta_files = [read_table(p) for p in list_files(Path(config["paths"]["processed_dir"]) / "graphs", (".parquet",)) if "_spatial_node-metadata_" in p.name]
    if spatial_meta_files:
        spatial_df = pd.concat(spatial_meta_files, ignore_index=True)
        if {"x", "y"}.issubset(spatial_df.columns):
            hue = "dominant_mapped_cell_type" if "dominant_mapped_cell_type" in spatial_df.columns else "region"
            scatter_spatial(spatial_df.head(5000), hue, fig_dir / "spatial_tissue_map", "Spatial tissue map")
            outputs.append(str(fig_dir / "spatial_tissue_map"))

    outlier_path = outlier_dir / "seaad_spatial_section_outliers_v1.tsv"
    if outlier_path.exists():
        outliers = read_table(outlier_path)
        if not outliers.empty:
            outlier_counts = outliers.groupby("metric", dropna=False).size().reset_index(name="n_flagged_sections")
            barplot(
                outlier_counts,
                x="metric",
                y="n_flagged_sections",
                out_base=outlier_dir / "seaad_spatial_outlier_counts_v1",
                title="Spatial outlier counts by metric",
            )
            outputs.append(str(outlier_dir / "seaad_spatial_outlier_counts_v1"))

    confounder_path = confounder_dir / "seaad_spatial_section_confounder_correlation_v1.tsv"
    if confounder_path.exists():
        confounders = read_table(confounder_path)
        if not confounders.empty and "feature" in confounders.columns:
            confounder_long = confounders.melt(id_vars="feature", var_name="correlated_with", value_name="correlation")
            heatmap(
                confounder_long,
                "feature",
                "correlated_with",
                "correlation",
                confounder_dir / "seaad_spatial_confounder_heatmap_v1",
                "Spatial section confounder correlation heatmap",
            )
            outputs.append(str(confounder_dir / "seaad_spatial_confounder_heatmap_v1"))

    outputs.extend(
        build_spatial_umap(
            graph_dir=Path(config["paths"]["processed_dir"]) / "graphs",
            out_dir=spatial_umap_dir,
            seed=int(config["seed"]),
            donor_target=int(config["qc"]["spatial"]["umap_sample_donors"]),
            max_nodes=int(config["qc"]["spatial"]["umap_max_nodes"]),
        )
    )

    append_stage_status(
        stage_status_path,
        "11_qc_report_and_plots",
        "success" if outputs else "blocked",
        "Generated QC plots, graph-input diagnostics, outlier summaries, and donor-sampled spatial embedding checks.",
        n_inputs=0,
        n_outputs=len(outputs),
    )
    finalize_run(config, "11_qc_report_and_plots", args.config, outputs, "success" if outputs else "partial")


if __name__ == "__main__":
    main()
