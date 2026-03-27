from __future__ import annotations

from pathlib import Path

import pandas as pd
import scanpy as sc

from _common import build_parser, finalize_run, init_run
from src.io_utils import ensure_dir, list_files, write_h5ad, write_table
from src.report_utils import append_stage_status
from src.taxonomy_utils import attach_existing_taxonomy, write_annotation_template


def main() -> None:
    args = build_parser("Attach taxonomy-compatible annotations to normalized snRNA objects.").parse_args()
    config, logger = init_run("05_snrna_annotate", args.config)
    in_dir = Path(config["paths"]["processed_dir"]) / "snrna_normalized"
    out_dir = ensure_dir(Path(config["paths"]["processed_dir"]) / "snrna_annotated")
    files = [p for p in list_files(in_dir, (".h5ad",)) if "normalized" in p.name]
    outputs: list[str] = []
    stage_status_path = config["artifact_paths"]["stage_status_manifest"]

    if not files:
        no_input_path = out_dir / "seaad_snrna_annotate_no_inputs_v1.tsv"
        write_table(pd.DataFrame([{"reason": "No normalized snRNA inputs were found."}]), no_input_path)
        append_stage_status(
            stage_status_path,
            "05_snrna_annotate",
            "blocked",
            "Annotation skipped because normalized snRNA inputs were missing.",
            n_inputs=0,
            n_outputs=1,
        )
        finalize_run(config, "05_snrna_annotate", args.config, [str(no_input_path)], "partial")
        return

    for file_path in files:
        adata = sc.read_h5ad(file_path)
        obs = attach_existing_taxonomy(adata.obs)
        adata.obs = obs
        out_h5ad = out_dir / file_path.name.replace("normalized", "annotated")
        write_h5ad(adata, out_h5ad)
        outputs.append(str(out_h5ad))

        ann_table = obs.reset_index(names="obs_name")
        ann_path = out_dir / file_path.name.replace(".h5ad", "_annotations.tsv").replace("normalized", "annotated")
        write_table(ann_table, ann_path)
        outputs.append(str(ann_path))

        if (ann_table["major_cell_class"] == "unknown").all():
            template_path = out_dir / file_path.name.replace(".h5ad", "_annotation_template.tsv").replace("normalized", "annotated")
            write_annotation_template(template_path, adata.obs_names.astype(str).tolist())
            outputs.append(str(template_path))
            logger.warning("No usable taxonomy columns found in %s; wrote template.", file_path.name)
    append_stage_status(
        stage_status_path,
        "05_snrna_annotate",
        "success",
        "Annotated normalized snRNA objects or wrote explicit annotation templates.",
        n_inputs=len(files),
        n_outputs=len(outputs),
    )
    finalize_run(config, "05_snrna_annotate", args.config, outputs, "success" if outputs else "partial")


if __name__ == "__main__":
    main()
