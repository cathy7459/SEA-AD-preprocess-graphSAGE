# Reproducibility

## Configuration

All analysis parameters are externalized under `configs/`:

- `project.yaml`: paths, seeds, region lists, and memory policy
- `qc.yaml`: snRNA, spatial, and graph thresholds
- `paths.yaml`: canonical artifact paths
- `datasets.yaml`: official dataset families and source-page expectations

## Run metadata

Each script appends a row to `results/manifests/run_manifest.tsv` containing:

- script name
- timestamp
- seed
- hostname
- Python version
- config path
- status
- output paths

`results/manifests/software_versions_v1.tsv` stores package versions as observed during execution.

## Determinism

- A fixed seed is set at the beginning of each script.
- Randomized steps such as PCA or randomized SVD are seeded.
- Neighbor graph construction is local to donor-region and does not depend on nondeterministic global ordering.

## Data provenance audit trail

`01_fetch_or_manifest_data.py` writes:

- a source manifest containing official page URL, discovered file title, direct URL if available, and expected local target
- a download-status manifest including checksum metadata where available

Controlled-access or landing-page-only resources are preserved with `manual_download_needed=1` rather than silently skipped.

## Memory-aware reproducibility

Reproducibility on a laptop requires defensive resource handling. This repository therefore:

- logs when backed reading is used,
- records file sizes before loading,
- falls back to chunk-safe operations above configurable thresholds,
- writes intermediate outputs after each donor-region or section unit.

## Figures

Each figure written to `results/figures/` is accompanied by the underlying long-format table in TSV format. This supports both publication reuse and figure regeneration without rerunning upstream preprocessing.
