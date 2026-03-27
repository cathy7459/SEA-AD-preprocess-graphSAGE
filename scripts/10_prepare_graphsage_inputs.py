from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, list_files, write_table
from src.report_utils import append_stage_status


def main() -> None:
    args = build_parser("Prepare GraphSAGE-ready graph manifests and feature paths.").parse_args()
    config, logger = init_run("10_prepare_graphsage_inputs", args.config)
    graph_dir = Path(config["paths"]["processed_dir"]) / "graphs"
    patient_dir = Path(config["paths"]["processed_dir"]) / "patient_features"
    out_dir = ensure_dir(Path(config["paths"]["processed_dir"]) / "graphsage_inputs")
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    graph_files = list_files(graph_dir, (".pt", ".parquet", ".tsv"))
    rows = []
    for file_path in graph_files:
        tokens = file_path.stem.split("_")
        region = next((token for token in tokens if token in set(config["analysis"]["supported_regions"] + ["CN"])), "unknown")
        donor = next((token for token in tokens if token.startswith("H") or token.startswith("donor-")), "donor-unknown")
        artifact_role = "other"
        if "node-metadata" in file_path.name:
            artifact_role = "node_metadata"
        elif "node-features" in file_path.name:
            artifact_role = "node_features"
        elif "edge-table" in file_path.name:
            artifact_role = "edge_table"
        elif "summary" in file_path.name:
            artifact_role = "summary"
        elif file_path.suffix == ".pt":
            artifact_role = "pyg_payload"
        rows.append(
            {
                "artifact_path": str(file_path),
                "artifact_name": file_path.name,
                "region": region,
                "donor_id": donor,
                "artifact_type": file_path.suffix.lstrip("."),
                "artifact_role": artifact_role,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / f"seaad_graphsage_graph_manifest_{config['version']}.tsv"
    write_table(manifest, manifest_path)

    patient_files = list_files(patient_dir, (".parquet", ".tsv"))
    patient_rows = []
    for path in patient_files:
        role = "other"
        if "graphready_features" in path.name:
            role = "graphready_features"
        elif "overlap" in path.name and "features" in path.name:
            role = "legacy_graphready_features"
        elif "cell_type_fractions" in path.name:
            role = "cell_type_fractions"
        elif "graph_summaries" in path.name:
            role = "graph_summaries"
        patient_rows.append({"artifact_path": str(path), "artifact_name": path.name, "artifact_type": path.suffix.lstrip("."), "artifact_role": role})
    patient_manifest = pd.DataFrame(patient_rows)
    patient_manifest_path = out_dir / f"seaad_graphsage_patient_manifest_{config['version']}.tsv"
    write_table(patient_manifest, patient_manifest_path)
    append_stage_status(
        stage_status_path,
        "10_prepare_graphsage_inputs",
        "success" if len(graph_files) + len(patient_files) > 0 else "blocked",
        "Prepared GraphSAGE manifests from available graph and patient artifacts.",
        n_inputs=len(graph_files) + len(patient_files),
        n_outputs=2,
    )
    finalize_run(config, "10_prepare_graphsage_inputs", args.config, [str(manifest_path), str(patient_manifest_path)])


if __name__ == "__main__":
    main()
