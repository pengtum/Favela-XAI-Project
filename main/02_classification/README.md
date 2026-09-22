# main/02_classification: per-site classifiers, Tree SHAP and misclassification at the boundary

Fits one XGBoost classifier per site on the 25 indicators, decomposes every
building's prediction into the contributions of the 25 indicators with Tree
SHAP, reports the held-out test performance, and measures how the
misclassification rate depends on the signed distance to the official 2013
favela boundary, under random five-fold cross-validation and with whole
neighbourhoods withheld.

## Manuscript items

| Item | Script |
|---|---|
| Methods 4.4: per-site XGBoost (200 trees, max depth 6, learning rate 0.1), stratified 80/20 split, seed 42 | 01 |
| Methods 4.6: Tree SHAP, one N × 25 attribution matrix per site (input of main/03-05, the Extended Data figures and most SI analyses) | 01 |
| SI Table S3, SI Section S4, Results 2.1: per-site test performance; mean accuracy 93.3%, per-site F1 0.79-0.98 | 02 |
| Methods 4.7: out-of-fold predictions (stratified five-fold, XGBoost 100 trees, seed 432) | 03 |
| Results 2.1, SI Section S5, SI Table S4, Extended Data Fig. 1: 26.8% misclassified within 6 m of the boundary against 3.7% at 90 m or more (ratio 7.2, 95% CI 3.9-19.7), near > far in all ten sites | 04 |
| Results 2.1, SI Sections S5 and S12: with 250 m blocks and a 200 m buffer, 51.3% against 15.4% (ratio 3.3, 2.0-6.6), near > far in all nine scored sites | 05 |
| Fig. 2: (a) outcome maps of Faz Quem Quer and Morro da Guaíba, (b) per-site test accuracy (mean 0.933), (c) misclassification rate in 6 m signed-distance bins | 06 |

SI Table S4 and Extended Data Fig. 1 are laid out from the tables of step 04
in `appendix/S05_boundary_errors`.

## Run order

From the repository root, after `main/01_features`:

```bash
python main/02_classification/01_train_models.py         # per-site models, split, Tree SHAP
python main/02_classification/02_table_S3.py             # SI Table S3 and the Results 2.1 accuracies
python main/02_classification/03_oof_predictions.py      # out-of-fold predictions (100 trees, seed 432)
python main/02_classification/04_boundary_error_rates.py # error rate by signed distance, site bootstrap
python main/02_classification/05_blocked_error_rates.py  # the same with 250 m blocks and a 200 m buffer
python main/02_classification/06_figure2.py              # Fig. 2
```

01 accepts `--sites <key> ...`. 02 needs 01; 04 needs 03; 06 needs 01-03.
03 and 05 are independent of 01 and 02.

## Inputs

- Per-site building tables and signed distances from `main/01_features`
  (`products.features_gpkg`, `products.signed_distance`)
- Official 2013 favela polygons (Fig. 2a outline): `paths.raw.favelas_2013`
- Settings: `config.yaml`, sections `model_main`, `shap`, `model_cv`, `oof`,
  `boundary_errors`, `spatial_cv`, `main_02_classification`

## Outputs (`outputs/main/02_classification/`)

| File | Content | Script |
|---|---|---|
| `<site>_model.json`, `<site>_imputer_medians.csv`, `<site>_split.csv` | fitted model, training medians for imputation, test-set flag per building | 01 |
| `<site>_buildings_shap.gpkg` | per-site attribution table: `uID`, `label`, `is_test`, `predicted_label`, `prob_favela`, the 25 indicators, 25 `shap_<indicator>` columns (log-odds, positive = toward favela), geometry | 01 |
| `table_S3_site_metrics.csv` / `.tex`, `table_S3_summary.txt` | SI Table S3 and the numbers quoted in Results 2.1 and SI Section S4 | 02 |
| `oof_predictions.parquet`, `oof_summary.csv` | out-of-fold favela probability, prediction and error of every building; per-site summary | 03 |
| `boundary_error_*.csv`, `site_counts_near_far.csv`, `boundary_error_summary.txt` | rates by 6 m bin and distance band, pooled and per site, with Wilson and site-bootstrap intervals; numbers of Results 2.1 and SI Section S5 | 04 |
| `blocked_*.csv` | the same rates under the blocked design, with fold and coverage tables | 05 |
| `figure2.pdf`, `figure2c_values.csv` | Fig. 2 and the values of panel c | 06 |

## Settings with alternatives

- `main_02_classification.figure2.panel_a_buildings` (all_buildings | test_only):
  colour every building by its prediction (default, as published), or only the held-out test buildings.
- `main_02_classification.figure2.misclassified_colours` (by_class | single):
  separate colours for the two error types, or one colour for both.
- `model_cv.fit_formal_as_class_1` (true | false): class coding of the
  cross-validation model; the favela probability is returned in both cases.
