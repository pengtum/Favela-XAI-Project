# main/05_attribution_plane: the two-dimensional attribution plane

Standardises the 25 Tree-SHAP attribution columns over the buildings of all
ten sites, fits one principal component analysis to the pooled matrix and
draws each site's buildings on the first two components (Methods 4.6, Fig. 5).
The PC1 scores are also the input of SI Fig. S10 (`appendix/S07_pca`).

## Manuscript items

| Item | Script |
|---|---|
| Methods 4.6: pooled, standardised PCA of the attributions, first two components kept | 01 |
| Fig. 5: ten panels on common PC1/PC2 axes (PC1 and PC2 explain 17.9% and 8.5% of the variance); density regions = the 15% of grid cells with the highest kernel density of each class; up to 2,000 points per class | 01 (PCA), 02 (drawing) |
| Results 2.4: in every site the favela and formal clouds are displaced in their centres of mass while overlapping | 01, 02 |
| Results 2.4 and SI Section S7: raw-indicator benchmark (mean per-site accuracy 0.782 and AUC 0.821, against 0.932 and 0.974 for the 25 indicators; SI Table S6) | computed in `appendix/S09_block_complexity_k/03_persite_protocol.py`, printed by 03 |
| SI Section S7, SI Fig. S10: PC1 thresholded within each site at the ROC-Youden optimum | 01 (scores); thresholds and maps in `appendix/S07_pca` |

## Run order

From the repository root, after `main/02_classification`:

```bash
python main/05_attribution_plane/01_attribution_pca.py         # pooled standardised PCA: scores, variance, loadings
python main/05_attribution_plane/02_figure5.py                 # Fig. 5 and per-site panel statistics
python main/05_attribution_plane/03_raw_benchmark_numbers.py   # Results 2.4 benchmark (after appendix/S09_block_complexity_k)
```

## Inputs

- `outputs/main/02_classification/<site>_buildings_shap.gpkg`: `label`, `uID`
  and the 25 `shap_<indicator>` columns (positive = toward favela).
- `outputs/appendix/S09_block_complexity_k/k_persite_summary.csv` (script 03
  only): per-site benchmark summary of SI Table S6.

## Outputs (`outputs/main/05_attribution_plane/`)

| File | Content | Script |
|---|---|---|
| `attribution_pca_scores.parquet` | one row per building: `site`, `row`, `uID`, `label`, `is_favela`, `PC1`, `PC2` | 01 |
| `explained_variance.csv` | the 25 components: explained variance ratio, cumulative ratio, eigenvalue, sign flag | 01 |
| `loadings.csv` | component × indicator: eigenvector and loading (eigenvector × √eigenvalue) | 01 |
| `scaler_params.csv` | pooled mean and s.d. of each SHAP column | 01 |
| `persite_summary.csv` | per site: class means of PC1 and PC2, PC1-only AUC | 01 |
| `figure5.pdf`, `figure5.svg` | Fig. 5 (183 mm wide) | 02 |
| `panel_stats.csv` | per site: out-of-fold AUC of a logistic regression on PC1 and PC2 (numerical summary of each panel's class separation) | 02 |
| `raw_benchmark_summary.csv` | rows "PC1 + PC2 of the raw 25 indicators" and "Raw 25 indicators" of SI Table S6 | 03 |

## Settings (`config.yaml`: `pca`, `main_05_attribution_plane`, `display_orders.figure5_panels`)

- PCA: z-score of each attribution column over the pooled buildings (population s.d., scikit-learn `StandardScaler`), full PCA with 25 components, PC1 and PC2 kept (Methods 4.6).
- Component signs (`orient_pc1`, `orient_pc2`): PC1 oriented so that the pooled favela mean is above the formal mean; PC2 with its largest loading positive.
- Density regions: Gaussian kernel density of each class on a 150 × 150 grid over the 1st-99th percentile box, from at most 20,000 buildings per site (seed 42); region = grid cells above the 85th percentile of the class density.
- Points: random samples of up to 2,000 buildings per class (seed 42).
- Raw-indicator benchmark (`pca.raw_benchmark`, `block_complexity.persite`): per site, two-component PCA of the 25 unstandardised raw indicators, evaluated on the SI S9 subset (buildings with a valid block-complexity index k); the 25-indicator reference (0.932 / 0.974) is the "Raw 25 indicators" row of SI Table S6 on the same subset.
