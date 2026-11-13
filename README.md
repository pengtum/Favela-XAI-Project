# Favela-XAI-Project

Code for:

> **Building-scale morphological signatures of informal settlements**
> Peng Luo, Fábio Duarte, Luiz Carlos Teixeira Coelho, Carlo Ratti

## Overview

From open building footprints of ten favela sites in Rio de Janeiro, the code
computes 25 morphometric indicators per building (dimension, shape, spatial
distribution, configuration and connectivity; momepy on a morphological
tessellation), trains one XGBoost classifier per site against the official 2013
favela boundaries, and uses Tree SHAP to attribute each building's classification
to the 25 indicators. From these attributions it derives the site-level
signatures (Fig. 3), the boundary profiles of each indicator (Fig. 4) and the
within-site attribution plane (Fig. 5). The supplementary analyses (spatial
cross-validation, boundary robustness, label-free clustering, cross-site
transfer, block complexity, 2013/2019 boundary overlay, terrain ablation) are
included as well.

This repository contains code only. The input data are open municipal layers
(see [Data](#data)).

## Repository layout

```
Favela-XAI-Project/
├── config.yaml               paths and parameters of every analysis
├── environment.yml           analysis environment (Python 3.10; conda)
├── environment-download.yml  OpenStreetMap download environment (osmnx; Python 3.11)
├── requirements.txt          pip equivalent of environment.yml
├── pyproject.toml            installs the shared package src/favela_xai
├── src/favela_xai/           shared code: indicators, labels, classifiers, Tree SHAP, PCA,
│                             signed distance and boundary profiles, spatial CV, statistics
├── main/                     main-text analyses (Figs 1-5)
│   ├── 01_features/          footprints, tessellation, 25 indicators, signed distance
│   ├── 02_classification/    per-site XGBoost, Tree SHAP, boundary error rates
│   ├── 03_site_signatures/   site-level importance and dominant features
│   ├── 04_boundary_profiles/ boundary profiles and transition metrics
│   └── 05_attribution_plane/ attribution PCA
├── appendix/                 one folder per Supplementary section (S2, S5-S14),
│                             including the Extended Data figures
└── data/README.md            where to download the raw data
```

Each folder has a README listing the items it produces and its run order. All
scripts read `config.yaml` (paths relative to the repository root) and write to
`outputs/<folder>/`.

## Installation

```bash
conda env create -f environment.yml
conda activate favela-xai
```

The OpenStreetMap streets for the block-complexity comparison (SI S9) are
downloaded once in a separate environment: `conda env create -f environment-download.yml`.

## Data

- **Raw data.** Building footprints 2013 and the official favela boundaries 2013
  and 2019 are openly available from the Rio de Janeiro open-data portal
  (Data.Rio), as are the administrative boundaries. `data/README.md` gives the
  links and the expected locations under `data/raw/`.
- **Terrain model.** The 5 m LiDAR digital terrain model (SI S11) is available
  from the Instituto Municipal de Urbanismo Pereira Passos (IPP) on request; set
  `paths.raw.elevation_raster` in `config.yaml`.
- **Derived data.** The per-building indicators and Tree-SHAP attributions are
  available from the corresponding author upon reasonable request.

## Running

From the repository root, in the `favela-xai` environment:

| Order | Folder | Needs |
|---|---|---|
| 0 | `appendix/S02_site_selection/01_context_layers.py --download` | internet; administrative layers for Fig. 1a and SI S2 |
| 1 | `main/01_features` | raw footprints and 2013 boundaries |
| 2 | `main/02_classification` | main/01 |
| 3 | `main/03_site_signatures`, `main/04_boundary_profiles`, `main/05_attribution_plane` | main/02 |
| 4 | `appendix/S05`, `S06`, `S07`, `S08`, `S10` | main/01-05 |
| 5 | `appendix/S09_block_complexity_k` (step 01 in `favela-xai-download`) | main/02 |
| 6 | `appendix/S11_terrain` | the terrain model |
| 7 | `appendix/S02_site_selection` | main/01 (and S11 for the slope columns of Table S1) |
| 8 | `appendix/S12_spatial_cv`, `S13_boundary_robustness`, `S14_label_free` | main/02, main/04 |

Scripts are numbered in run order within each folder; the folder READMEs note the
few cross-folder dependencies (e.g. `main/05_attribution_plane/03` uses SI S9 step 03).

## Manuscript items and the scripts that produce them

### Main text

| Item | Folder | Script(s) |
|---|---|---|
| Fig. 1 Overview of the framework (map panels) | `main/01_features` | 08, 09 |
| Fig. 2 Favela-detection performance | `main/02_classification` | 06 (data 02-04) |
| Fig. 3 Site-level signatures | `main/03_site_signatures` | 02 (data 01) |
| Fig. 4 Boundary profile of the signature | `main/04_boundary_profiles` | 02 (data 01) |
| Fig. 5 Building-level attribution planes | `main/05_attribution_plane` | 02 (PCA 01) |

### Extended Data

| Item | Folder | Script(s) |
|---|---|---|
| ED Fig. 1 Misclassification rate by signed distance | `appendix/S05_boundary_errors` | 01 |
| ED Fig. 2 Per-site feature importance | `appendix/S06_attribution_analyses` | 01 |
| ED Fig. 3 Site-level rankings under spatial cross-validation | `appendix/S12_spatial_cv` | 13 (with `ed3_schematic.py`) |
| ED Fig. 4 Site-level importance shares under spatial cross-validation | `appendix/S12_spatial_cv` | 14 |
| ED Fig. 5 Shifted boundaries and label controls | `appendix/S13_boundary_robustness` | 07 |

### Supplementary figures

| Item | Folder | Script(s) |
|---|---|---|
| Fig. S1 Study favelas compared with all favelas | `appendix/S02_site_selection` | 05 |
| Fig. S2 Spatial distribution of classification outcomes | `appendix/S05_boundary_errors` | 03 |
| Fig. S3 Drivers of false-positive misclassification | `appendix/S05_boundary_errors` | 04 |
| Figs S4-S5 Case studies | `appendix/S05_boundary_errors` | 05 |
| Fig. S6 Classification probability across the boundary | `appendix/S06_attribution_analyses` | 02 |
| Fig. S7 Directional SHAP dependence | `appendix/S06_attribution_analyses` | 03 |
| Fig. S8 Cross-site heterogeneity of the SHAP dependence | `appendix/S06_attribution_analyses` | 04 |
| Fig. S9 Direction of the boundary transition | `appendix/S06_attribution_analyses` | 06 |
| Fig. S10 PC1 mapped within each site | `appendix/S07_pca` | 02 |
| Figs S11-S12 Cross-site transfer | `appendix/S08_transfer` | 01, 02 |
| Fig. S13 Block-complexity index k | `appendix/S09_block_complexity_k` | 06 |
| Figs S14-S18 Terrain ablation | `appendix/S11_terrain` | 03-05 |
| Figs S19-S21 Spatial cross-validation | `appendix/S12_spatial_cv` | 06, 08, 09 |
| Fig. S22 Raw-indicator and SHAP boundary profiles | `appendix/S13_boundary_robustness` | 04 |
| Fig. S23 Label-free clustering | `appendix/S14_label_free` | 04 |

### Supplementary tables

| Item | Folder | Script(s) |
|---|---|---|
| Table S1 Study-site context | `appendix/S02_site_selection` | 06 |
| Table S2 The 25 indicators | `main/01_features` | 06 |
| Table S3 Per-site classification performance | `main/02_classification` | 02 |
| Table S4 Misclassification rate by distance band | `appendix/S05_boundary_errors` | 02 |
| Table S5 Boundary-transition metrics | `main/04_boundary_profiles` | 03 |
| Tables S6-S7 Block complexity | `appendix/S09_block_complexity_k` | 03, 04 |
| Table S8 Buildings in the 2013 and 2019 boundaries | `appendix/S10_boundaries_2013_2019` | 01 |
| Table S9 Terrain ablation | `appendix/S11_terrain` | 06 |
| Tables S10-S16 Spatial cross-validation | `appendix/S12_spatial_cv` | 05, 07, 11 |
| Table S17 Sensitivity of the sign-change distance | `appendix/S13_boundary_robustness` | 08 |
| Table S18 Label-free clustering | `appendix/S14_label_free` | 03 |

## Citation

Please cite the paper above.

## License

MIT (see `LICENSE`).
