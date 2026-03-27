# SEA-AD GraphSAGE Preprocessing Pipeline

This repository provides a fully reproducible, laptop-oriented preprocessing pipeline for SEA-AD single-nucleus RNA-seq and spatial transcriptomics data preparation prior to GraphSAGE-based Alzheimer's disease subtype discovery. The workflow is designed for 16 GB laptops where available RAM may fall to 6-8 GB, so all major steps are region-wise, donor-wise, or section-wise and save intermediate outputs aggressively.

## Biological objective

The immediate goal is to produce graph-ready multimodal donor-region representations from Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD) data. The downstream use case is hierarchical GraphSAGE analysis where:

1. Cell-level graphs are built within each donor-region.
2. Donor-region graph summaries are aggregated into donor or patient representations.
3. Patient-level embeddings can later be clustered into candidate Alzheimer's disease subtypes.
4. Those subtypes can be evaluated against clinical prognosis, pathology progression, and survival endpoints.

This repository intentionally stops at preprocessing and graph-ready data preparation. It does not train GraphSAGE models.

## Data provenance

The primary official source page is the SEA-AD data portal:

- https://brain-map.org/consortia/sea-ad/our-data

The pipeline programmatically parses this page and records direct URLs or source landing pages for:

- donor metadata
- processed single nucleus RNA-seq / ATAC-seq resources
- spatial transcriptomics resources
- microglial annotated multiregion nuclei data

When the official source exposes a direct downloadable object URL, the pipeline downloads it with retries and resumable streaming. When the official source exposes only a landing page or a controlled-access portal, the pipeline writes a manifest row flagged as `manual_download_needed` and continues.

## Design principles

- Memory-aware by default: avoids full-atlas concatenation, prefers sparse matrices, backed reading, chunked summaries, and donor/region/section iteration.
- Reproducible: all thresholds, seeds, and region choices live in YAML configs; scripts are restartable and record manifests and logs.
- Modality-aware: snRNA-seq and spatial data are processed independently before alignment.
- Graph-ready: outputs donor-region local graphs and patient-level summary matrices suitable for later GraphSAGE workflows.
- Publication-oriented: produces QC summaries, figure tables, and paper-friendly plots in PNG and PDF formats.

## Repository layout

```text
.
|- README.md
|- LICENSE
|- .gitignore
|- requirements.txt
|- environment/
|  |- environment.yml
|- configs/
|  |- project.yaml
|  |- qc.yaml
|  |- paths.yaml
|  |- datasets.yaml
|- data/
|  |- raw/
|  |- interim/
|  |- processed/
|- docs/
|  |- pipeline.md
|  |- reproducibility.md
|- notebooks/
|  |- 00_data_inventory.ipynb
|  |- 01_qc_exploration.ipynb
|  |- 02_annotation_check.ipynb
|  |- 03_spatial_mapping_check.ipynb
|  |- 04_graph_summary_review.ipynb
|- results/
|  |- figures/
|  |- logs/
|  |- manifests/
|- scripts/
|  |- 00_build_cohort_table.py
|  |- 01_fetch_or_manifest_data.py
|  |- 02_split_by_region.py
|  |- 03_snrna_qc.py
|  |- 04_snrna_normalize.py
|  |- 05_snrna_annotate.py
|  |- 06_spatial_qc.py
|  |- 07_spatial_map_to_snrna.py
|  |- 08_build_local_graphs.py
|  |- 09_aggregate_patient_features.py
|  |- 10_prepare_graphsage_inputs.py
|  |- 11_qc_report_and_plots.py
|  |- 12_audit_snrna_pipeline.py
|- src/
   |- ann_utils.py
   |- config_utils.py
   |- graph_utils.py
   |- io_utils.py
   |- logging_utils.py
   |- naming.py
   |- patient_utils.py
   |- path_audit_utils.py
   |- plotting.py
   |- qc_utils.py
   |- report_utils.py
   |- seaad_manifest_utils.py
   |- snrna_utils.py
   |- spatial_utils.py
   |- taxonomy_utils.py
   |- web_fetch_utils.py
   |- datasets/
```

## Environment setup

Conda:

```bash
conda env create -f environment/environment.yml
conda activate seaad-graphsage-pre
```

Pip:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Recommended run order

```bash
python scripts/00_build_cohort_table.py --config configs/project.yaml
python scripts/01_fetch_or_manifest_data.py --config configs/project.yaml
python scripts/02_split_by_region.py --config configs/project.yaml
python scripts/03_snrna_qc.py --config configs/project.yaml
python scripts/04_snrna_normalize.py --config configs/project.yaml
python scripts/05_snrna_annotate.py --config configs/project.yaml
python scripts/06_spatial_qc.py --config configs/project.yaml
python scripts/07_spatial_map_to_snrna.py --config configs/project.yaml
python scripts/08_build_local_graphs.py --config configs/project.yaml
python scripts/09_aggregate_patient_features.py --config configs/project.yaml
python scripts/10_prepare_graphsage_inputs.py --config configs/project.yaml
python scripts/11_qc_report_and_plots.py --config configs/project.yaml
```

## Expected outputs

- cohort and overlap tables
- download and source manifests
- donor-region snRNA QC summaries and filtered `.h5ad`
- donor-region normalized `.h5ad` with sparse normalized layers
- region-level HVG vote tables
- section-wise spatial QC tables and filtered objects
- spatial-to-snRNA mapping tables
- donor-region graph artifacts (`.pt`, `.parquet`, `.tsv`)
- patient-level aggregated feature matrix
- QC figures and their underlying source tables

## Reproducibility notes

- The pipeline records software versions, seeds, input manifests, and run metadata under `results/manifests/`.
- All important parameters are stored in YAML files under `configs/`.
- Logs are written per script to `results/logs/`.
- Direct downloads and manual-download requirements are both preserved in tabular manifests for auditability.

## Limitations

- Some official SEA-AD resources are controlled-access or exposed only via landing pages. These are captured in manifests and not silently ignored.
- Exact taxonomy transfer may require external references or services not always feasible on a laptop. The annotation stage therefore supports direct mapping when available and otherwise writes a pluggable expected-input format.
- Spatial file formats vary across releases; the loader is defensive and focused on robust Xenium-style section inputs.

## Extending to GraphSAGE training

The output of `10_prepare_graphsage_inputs.py` is intended to feed a later modeling workflow. A typical extension would:

1. load donor-region graphs,
2. train a GraphSAGE encoder at the cell or patch level,
3. pool donor-region embeddings,
4. build donor-level or patient-level graphs,
5. cluster patient embeddings and test associations with clinical outcomes.
