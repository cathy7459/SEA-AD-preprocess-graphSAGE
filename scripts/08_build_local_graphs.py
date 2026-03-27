from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from _common import build_parser, finalize_run, init_run
from src.graph_utils import (
    build_graph_payload,
    edge_table,
    expression_embedding,
    feature_table,
    graph_summary,
    knn_edges,
    save_graph_payload,
    spatial_edges,
)
from src.io_utils import ensure_dir, list_files, read_table, write_table
from src.path_audit_utils import require_existing_readable, require_writable_dir
from src.report_utils import append_stage_status
from src.spatial_utils import patchify_spatial_adata


def save_graph_artifacts(
    *,
    node_ids: pd.Series,
    features: np.ndarray,
    metadata: pd.DataFrame,
    expr_edges: np.ndarray,
    spatial_knn_edges: np.ndarray | None,
    donor: str,
    region: str,
    modality: str,
    out_dir: Path,
    version: str,
) -> list[str]:
    payload = build_graph_payload(features, metadata, expression_edge_index=expr_edges, spatial_edge_index=spatial_knn_edges)
    graph_name = "cellgraph" if modality == "snrna" else "patchgraph"
    pt_path = out_dir / f"seaad_graph_{region}_{donor}_{graph_name}_{version}.pt"
    save_graph_payload(payload, pt_path)

    metadata = metadata.copy()
    metadata.insert(0, "node_id", node_ids.astype(str).to_numpy())
    meta_path = out_dir / f"seaad_graph_{region}_{donor}_{modality}_node-metadata_{version}.parquet"
    write_table(metadata, meta_path)

    feature_path = out_dir / f"seaad_graph_{region}_{donor}_{modality}_node-features_{version}.parquet"
    write_table(feature_table(node_ids, features, prefix=f"{modality}_feat"), feature_path)

    edge_frames = [edge_table(expr_edges, "expression")]
    if spatial_knn_edges is not None:
        edge_frames.append(edge_table(spatial_knn_edges, "spatial"))
    edge_path = out_dir / f"seaad_graph_{region}_{donor}_{modality}_edge-table_{version}.parquet"
    write_table(pd.concat(edge_frames, ignore_index=True), edge_path)

    summary_path = out_dir / f"seaad_graph_{region}_{donor}_{modality}_summary_{version}.tsv"
    write_table(graph_summary(payload, donor, region, modality), summary_path)
    return [str(pt_path), str(meta_path), str(feature_path), str(edge_path), str(summary_path)]


def build_snrna_graph(file_path: Path, config: dict, out_dir: Path) -> list[str]:
    adata = sc.read_h5ad(file_path)
    if "X_pca_hvg" in adata.obsm:
        features = np.asarray(adata.obsm["X_pca_hvg"])
    else:
        features = expression_embedding(adata.X, n_components=config["qc"]["graph"]["expression_feature_dim"], seed=config["seed"])
    expr_edges = knn_edges(features, config["qc"]["graph"]["expression_knn"]) if adata.n_obs > 1 else np.zeros((2, 0), dtype=np.int64)
    metadata = adata.obs.reset_index(names="obs_name")
    region = next((token for token in file_path.stem.split("_") if token in config["analysis"]["supported_regions"]), "unknown")
    donor = next((token for token in file_path.stem.split("_") if token.startswith("H") or token.startswith("donor-")), "donor-unknown")
    return save_graph_artifacts(
        node_ids=metadata["obs_name"],
        features=features,
        metadata=metadata,
        expr_edges=expr_edges,
        spatial_knn_edges=None,
        donor=donor,
        region=region,
        modality="snrna",
        out_dir=out_dir,
        version=config["version"],
    )


def _normalize_patch_counts(matrix: sparse.csr_matrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    counts = matrix.astype(np.float32)
    libsize = np.asarray(counts.sum(axis=1)).ravel()
    libsize[libsize == 0] = 1.0
    normalized = counts.multiply(1e4 / libsize[:, None])
    normalized.data = np.log1p(normalized.data)
    return normalized.toarray()


def build_spatial_graph(qc_path: Path, mapping_path: Path | None, config: dict, out_dir: Path) -> list[str]:
    adata = sc.read_h5ad(qc_path)
    if mapping_path and mapping_path.exists():
        mapped = read_table(mapping_path)
        if not mapped.empty and "obs_name" in mapped.columns:
            mapped = mapped.set_index("obs_name")
            for column in ["mapped_cell_type", "mapped_confidence", "mapping_status", "mapping_note"]:
                if column in mapped.columns:
                    adata.obs[column] = mapped.reindex(adata.obs_names)[column].to_numpy()

    patch_matrix, patch_meta = patchify_spatial_adata(
        adata,
        patch_size_um=float(config["qc"]["spatial"]["patch_size_um"]),
        min_cells_per_patch=int(config["qc"]["spatial"]["min_cells_per_patch"]),
    )
    if patch_meta.empty:
        donor = str(adata.obs["donor_id"].iloc[0]) if "donor_id" in adata.obs.columns and adata.n_obs else "donor-unknown"
        region = str(adata.obs["region"].iloc[0]) if "region" in adata.obs.columns and adata.n_obs else "unknown"
        note_path = out_dir / f"seaad_graph_{region}_{donor}_spatial_patchgraph_empty_{config['version']}.tsv"
        write_table(
            pd.DataFrame(
                [
                    {
                        "donor_id": donor,
                        "region": region,
                        "reason": "No spatial patches met the configured min_cells_per_patch threshold.",
                    }
                ]
            ),
            note_path,
        )
        return [str(note_path)]

    normalized = _normalize_patch_counts(patch_matrix)
    patch_features = expression_embedding(
        normalized,
        n_components=int(config["qc"]["spatial"]["patch_expression_dim"]),
        seed=int(config["seed"]),
    )

    extra_cols = [col for col in patch_meta.columns if col.startswith("celltype_fraction__")]
    extra_cols += [col for col in ["mapped_confidence_mean", "n_cells", "mean_n_transcripts"] if col in patch_meta.columns]
    if extra_cols:
        extra_values = patch_meta[extra_cols].fillna(0).to_numpy(dtype=float)
        features = np.concatenate([patch_features, extra_values], axis=1)
    else:
        features = patch_features

    expr_edges = knn_edges(features, config["qc"]["graph"]["expression_knn"]) if len(patch_meta) > 1 else np.zeros((2, 0), dtype=np.int64)
    spatial_knn = spatial_edges(patch_meta[["x", "y"]].to_numpy(dtype=float), config["qc"]["graph"]["spatial_knn"])
    donor = str(patch_meta["donor_id"].iloc[0])
    region = str(patch_meta["region"].iloc[0])
    return save_graph_artifacts(
        node_ids=patch_meta["patch_id"],
        features=features,
        metadata=patch_meta,
        expr_edges=expr_edges,
        spatial_knn_edges=spatial_knn,
        donor=donor,
        region=region,
        modality="spatial",
        out_dir=out_dir,
        version=config["version"],
    )


def main() -> None:
    args = build_parser("Build donor-region local graphs for snRNA cells and spatial patches.").parse_args()
    config, logger = init_run("08_build_local_graphs", args.config)
    out_dir = require_writable_dir(Path(config["paths"]["processed_dir"]) / "graphs", "graphs_output_dir")
    graph_check_dir = require_writable_dir(Path("results/check/graph_inputs"), "graph_inputs_check_dir")
    selected_regions = set(args.regions or [])
    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    snrna_dir = require_existing_readable(Path(config["paths"]["processed_dir"]) / "snrna_annotated", "snrna_annotated_dir")
    spatial_qc_dir = require_existing_readable(Path(config["paths"]["interim_dir"]) / "spatial_qc", "spatial_qc_dir")
    mapping_dir = Path(config["paths"]["processed_dir"]) / "spatial_mapping"

    snrna_inputs = list_files(snrna_dir, (".h5ad",))
    spatial_inputs = [p for p in list_files(spatial_qc_dir, (".h5ad",)) if "_qc_" in p.name]
    if selected_regions:
        snrna_inputs = [p for p in snrna_inputs if any(f"_{region}_" in p.name for region in selected_regions)]
        spatial_inputs = [p for p in spatial_inputs if any(f"_{region}_" in p.name for region in selected_regions)]

    if not snrna_inputs and not spatial_inputs:
        no_input_path = out_dir / "seaad_graph_build_no_inputs_v1.tsv"
        write_table(pd.DataFrame([{"reason": "No annotated snRNA or QC-filtered spatial inputs were available for graph construction."}]), no_input_path)
        append_stage_status(
            stage_status_path,
            "08_build_local_graphs",
            "blocked",
            "Graph construction skipped because upstream graph-ready inputs were missing.",
            n_inputs=0,
            n_outputs=1,
        )
        finalize_run(config, "08_build_local_graphs", args.config, [str(no_input_path)], "partial")
        return

    graph_inventory_rows: list[dict[str, object]] = []
    for file_path in snrna_inputs:
        logger.info("Graph build start | modality=snrna | file=%s", file_path.name)
        artifact_paths = build_snrna_graph(file_path, config, out_dir)
        outputs.extend(artifact_paths)
        graph_inventory_rows.append({"modality": "snrna", "input_path": str(file_path), "n_artifacts": len(artifact_paths)})

    for qc_path in spatial_inputs:
        mapping_path = mapping_dir / qc_path.name.replace("_qc_", "_mapped_").replace(".h5ad", ".tsv")
        logger.info("Graph build start | modality=spatial | file=%s", qc_path.name)
        artifact_paths = build_spatial_graph(qc_path, mapping_path if mapping_path.exists() else None, config, out_dir)
        outputs.extend(artifact_paths)
        graph_inventory_rows.append({"modality": "spatial", "input_path": str(qc_path), "mapping_path": str(mapping_path), "n_artifacts": len(artifact_paths)})

    inventory_path = write_table(pd.DataFrame(graph_inventory_rows), graph_check_dir / "seaad_graph_input_inventory_v1.tsv")
    outputs.append(str(inventory_path))

    append_stage_status(
        stage_status_path,
        "08_build_local_graphs",
        "success",
        "Built graph-ready artifacts: node metadata, node features, edge tables, and serialized payloads for snRNA cells and spatial patches.",
        n_inputs=len(snrna_inputs) + len(spatial_inputs),
        n_outputs=len(outputs),
    )
    finalize_run(config, "08_build_local_graphs", args.config, outputs, "success")


if __name__ == "__main__":
    main()
