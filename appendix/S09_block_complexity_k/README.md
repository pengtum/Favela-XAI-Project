# appendix/S09_block_complexity_k: comparison with the block-complexity index k

Re-implements the per-building block-complexity index *k* of Bettencourt et al.
on the ten sites (OpenStreetMap street blocks, a tessellation within each block,
depth of each building's cell from the block boundary) and compares *k* with the
25 morphological indicators on the same buildings. Main results (SI S9): the 25
indicators separate favela from formal-settlement buildings more accurately than
*k* alone, within each site and pooled across sites, and the direction of the
relationship between *k* and the favela label differs between sites.

## Manuscript items

| Item | Script |
|---|---|
| Methods 4.9 and SI S9 "Implementation summary": blocks, k per building, share of buildings outside every block | 01, 02 |
| SI Table S6: per-site protocol (rows "Block complexity k alone", "PC1 + PC2 of the raw 25 indicators", "Raw 25 indicators"); mean per-site AUC quoted in the Discussion | 03 |
| SI Table S7: pooled cross-site protocol (rows "Block complexity k alone", "Raw 25 indicators") | 04 |
| SI S9 and S9.2: distribution of k by label, per-site Cohen's d and AUC of k | 05 |
| SI Fig. S13: distribution of k by label | 06 |

## Run order

From the repository root, after `main/01_features` and `main/02_classification`:

```bash
# download environment (environment-download.yml)
python appendix/S09_block_complexity_k/01_download_osm_streets.py        # street network per site [--force]
# analysis environment (environment.yml)
python appendix/S09_block_complexity_k/02_block_complexity_k.py          # blocks, tessellation, k per building, block coverage
python appendix/S09_block_complexity_k/03_persite_protocol.py            # SI Table S6
python appendix/S09_block_complexity_k/04_pooled_protocol.py             # SI Table S7
python appendix/S09_block_complexity_k/05_k_distribution.py              # SI S9.2 statistics, data of Fig. S13
python appendix/S09_block_complexity_k/06_figure_S13_k_distribution.py   # SI Fig. S13
```

`k_common.py` holds the helpers shared by 03-06 (reading k and the per-site
tables, the two building subsets, number formatting); it is not run on its own.

## Inputs

- Per-site building tables with the 25 indicators (`main/01_features`, `products.features_gpkg`).
- Per-building Tree SHAP of the main classifiers (`main/02_classification`, `products.shap_gpkg`), used only to define the building subsets.
- OpenStreetMap street network per site, written by 01 to `<paths.raw.osm_streets_dir>/<site>.gpkg` (© OpenStreetMap contributors, ODbL). OpenStreetMap changes over time; the paper used extracts downloaded on 8 May 2026. Record the date of a new download in `block_complexity.osm_access_date`.

## Outputs (`outputs/appendix/S09_block_complexity_k/`)

| File | Content | Script |
|---|---|---|
| `block_complexity_k.csv` | per building: site, uID, label, block, block size, k | 02 |
| `block_coverage.csv` | per site: buildings in a block, share outside every block, blocks, buildings with a valid k | 02 |
| `k_persite_metrics.csv`, `k_persite_summary.csv`, `table_bettencourt_persite.tex` | per-site accuracy and AUC of each feature set; SI Table S6 | 03 |
| `k_pooled_auc.csv`, `table_bettencourt_pooled.tex` | fold AUCs; SI Table S7 | 04 |
| `k_distribution.csv`, `k_histogram.csv`, `k_in_text_numbers.txt` | statistics of k by label, per site and pooled; histogram of Fig. S13a; numbers quoted in SI S9 | 05 |
| `Figure_R1_k_distribution.pdf` | SI Fig. S13 | 06 |

## Definitions and settings (`config.yaml`)

- Blocks (`block_complexity`): polygonised OpenStreetMap street network (polygons > 100 m²); each building is assigned to the block containing its centroid.
- k: tessellation within each block; cells are adjacent when they share > 1 cm of boundary; cells touching the block boundary (0.5 m buffer) have k = 1, the others their breadth-first-search depth from those cells + 1.
- Building subsets (`k_common.py`): valid k (k >= 1) with complete Tree-SHAP attributions for the distribution of k; in addition all 25 indicators present for the classification comparison.
- Classifier (`block_complexity.classifier`, the classifier of this comparison, separate from the main per-site models of Methods 4.4): XGBoost, 200 trees, max depth 6, learning rate 0.1, seed 42, intercept = favela share of the training rows; per-site stratified 80/20 split (seed 42); pooled stratified 5-fold cross-validation (seed 42). The splits and folds are drawn on the buildings in the row order of the per-building tables.
- PC1 + PC2 of the raw 25 indicators (`pca.raw_benchmark`): two-component PCA of the unstandardised 25 indicators, fitted within each site.
- Download and layout settings: `appendix_S09_block_complexity_k`.
