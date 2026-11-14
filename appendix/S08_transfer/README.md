# appendix/S08_transfer: cross-site classifier transfer

Applies each site's main classifier, unchanged, to the held-out test split of
every other site (10 × 10 balanced-accuracy matrix), and tests which
between-site similarities go with successful transfer (seven measures, Mantel
tests with site-identity permutations, Benjamini-Hochberg correction).

## Manuscript items

| Item | Script |
|---|---|
| SI Fig. S11: 10 × 10 transfer balanced-accuracy matrix (rows = training site, columns = target site) | 01 |
| SI Section S8 and Discussion: mean cross-site balanced accuracy over the 90 ordered pairs 0.574 (site-bootstrap 95% CI 0.529-0.649) against 0.896 within sites; 24 of 90 pairs below chance | 01 |
| Table of all 100 ordered pairs (accuracy, balanced accuracy and AUC); its balanced accuracies are the values of SI Fig. S11 | 01 |
| SI Fig. S12 and SI Section S8: Mantel correlation of transfer with seven between-site measures; similarity of the favela-formal contrast is the strongest (r = 0.59, BH q < 0.001) | 02 |
| Methods, "Cross-site classifier transferability" | 01, 02 |

## Run order

From the repository root, after `main/01_features` and `main/02_classification`:

```bash
python appendix/S08_transfer/01_transfer_matrix.py      # transfer matrix, site bootstrap, table of all pairs, SI Fig. S11
python appendix/S08_transfer/02_transfer_similarity.py  # site descriptors, seven measures, Mantel tests and BH q, SI Fig. S12
```

Both scripts require all ten sites.

## Inputs

- `products.features_gpkg` (`main/01_features`): label and the 25 indicators of every building.
- `products.model_json` (`main/02_classification`): the main per-site models (XGBoost, 200 trees, max depth 6, learning rate 0.1, seed 42).
- `products.shap_gpkg` (`main/02_classification`; script 02): the 25 Tree-SHAP columns (positive = toward favela).
- `products.site_metrics` (`main/02_classification`; optional): SI Table S3, used to check the diagonal accuracy.

The held-out test split of each site is re-derived with
`favela_xai.model.main_split` (stratified 80/20, seed 42), as in
`main/02_classification`.

## Outputs (`outputs/appendix/S08_transfer/`)

| File | Content | Script |
|---|---|---|
| `transfer_pairwise_metrics.csv` | 100 ordered pairs: target test size, favela test buildings and share, accuracy, balanced accuracy, AUC | 01 |
| `transfer_pairwise_metrics_labelled.csv` | the same table with descriptive column headers (UTF-8 with BOM) | 01 |
| `transfer_balanced_accuracy_matrix.csv` | 10 × 10 balanced accuracy, full precision (input of 02) | 01 |
| `transfer_summary.log` | within- and cross-site means, site-bootstrap intervals and ranges; pairs below chance; favela shares; mean transfer by training and target site | 01 |
| `Figure_R1_transfer_balanced_accuracy.pdf` | SI Fig. S11 | 01 |
| `transfer_predictors.csv` | seven measures: Mantel r, p, BH q, permutation-floor flag, q < 0.05 flag | 02 |
| `site_descriptors.csv` | per site and indicator: favela fabric f, formal fabric n, attribution share a; favela share π | 02 |
| `Figure_R1_transfer_similarity.pdf` | SI Fig. S12 | 02 |

## Settings (`config.yaml`: `transfer`, `appendix_S08_transfer`)

- Classifiers applied without refitting; predicted favela when P(favela) ≥ 0.5; balanced accuracy = mean of favela and formal-settlement recall; AUC with favela as the positive class.
- `transfer.target_imputation`: `target_training_medians` (missing values of the target test rows filled with the target site's training medians) | `source_training_medians` (with those of the applied model's site).
- Site bootstrap of the cross-site mean: sites resampled with replacement, cells of the same drawn site dropped; 10,000 resamples (seed 20260917); 2.5 and 97.5 percentiles.
- Indicator standardisation: (x − pooled median) / pooled IQR over the buildings of all ten sites (IQR 0 → 1).
- Site descriptors: f_s and n_s = standardised medians of favela and formal buildings; a_s = mean \|SHAP\| over favela buildings, scaled to sum 1; π_s = favela share.
- Seven measures: cos(f_a − n_a, f_b − n_b); same dominant indicator (argmax a); cos(a_a, a_b); Spearman of a_a and a_b; favela- and formal-fabric mean absolute difference; \|π_a − π_b\|.
- Mantel test: Spearman over the 90 off-diagonal cells; site identities permuted on rows and columns; two-sided on \|r\|; p = (c + 1)/(P + 1), P = 9,999 (seed 42); Benjamini-Hochberg across the seven measures. Fig. S12b: orange bars where the uncorrected p < 0.05.
