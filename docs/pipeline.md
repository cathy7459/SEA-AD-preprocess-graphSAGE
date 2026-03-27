# Pipeline overview

## Scope

This repository prepares SEA-AD snRNA-seq and spatial transcriptomics data for later GraphSAGE analysis. The implementation is intentionally laptop-first and avoids giant in-memory atlas objects.

## Stage-by-stage workflow

1. `00_build_cohort_table.py`
   - parses official manifests and metadata tables
   - standardizes donor, region, and modality labels
   - builds donor-region overlap tables

2. `01_fetch_or_manifest_data.py`
   - scrapes the official SEA-AD source page
   - resolves direct URLs when available
   - downloads accessible files with retry/resume support
   - records inaccessible or controlled resources in a manifest

3. `02_split_by_region.py`
   - inventories raw files by modality
   - creates region-scoped file manifests
   - avoids loading whole resources into memory

4. `03_snrna_qc.py`
   - reads donor-region snRNA objects in backed or chunk-safe mode
   - computes QC metrics
   - applies donor-wise configurable filters
   - saves filtered objects and QC tables

5. `04_snrna_normalize.py`
   - normalizes donor-region counts safely
   - stores sparse normalized layers
   - computes donor-level HVGs and region-level vote tables

6. `05_snrna_annotate.py`
   - uses taxonomy-compatible columns when available
   - supports plug-in reference mapping
   - writes major class, subclass, and confidence columns

7. `06_spatial_qc.py`
   - processes one section at a time
   - computes section QC metrics
   - filters low-quality cells or spots
   - writes section-wise outputs

8. `07_spatial_map_to_snrna.py`
   - aligns spatial data to region-specific snRNA references
   - uses nearest-centroid mapping over shared gene space when feasible
   - otherwise writes template outputs for later mapping

9. `08_build_local_graphs.py`
   - creates donor-region local graphs only
   - builds expression similarity and spatial adjacency edges
   - writes PyG-ready artifacts and graph summaries

10. `09_aggregate_patient_features.py`
    - converts donor-region outputs into donor-level feature matrices
    - includes cell-type composition, graph summaries, and region summaries

11. `10_prepare_graphsage_inputs.py`
    - standardizes graph artifact paths and metadata
    - writes final training manifests and patient feature tables

12. `11_qc_report_and_plots.py`
    - produces publication-style QC and composition figures
    - saves all figure source tables alongside PNG/PDF outputs

## Low-memory implementation notes

- Prefer `scanpy.read_h5ad(..., backed="r")` for large `.h5ad`.
- Process one donor-region or one section at a time.
- Avoid full concatenation of atlas-wide AnnData objects.
- Keep matrices sparse whenever possible.
- Store intermediate summaries as TSV or Parquet.
- Build neighborhood graphs only within donor-region units.

## Region strategy

The default configuration starts with two regions (`MTG`, `A9`) but supports up to four. In four-region mode, each region is processed sequentially. The overlap table keeps only donors suitable for multimodal integration if configured to do so.

## Dataset-specific fetchers

Dataset-specific code lives in `src/datasets/` so that each official dataset family has a dedicated handler:

- donor metadata
- processed single nucleus RNA-seq and ATAC-seq
- spatial transcriptomics
- microglial annotated multiregion nuclei

This keeps provenance explicit and matches the requirement that multi-dataset logic should not be hidden in a single opaque code file.
