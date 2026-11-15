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
| Methods 4.4: per-site XGBoost with the xgboost defaults (100 trees, max depth 6, learning rate 0.3), seed 432, unstratified 80/20 split with seed 432 | 01 |
| Methods 4.6: Tree SHAP, one N × 25 attribution matrix per site (input of main/03-05, the Extended Data figures and most SI analyses) | 01 |
| SI Table S3, SI Section S4, Results 2.1: per-site test performance; mean accuracy over the ten sites, range of the per-site F1 | 02 |
| Methods 4.7: out-of-fold predictions (stratified five-fold, seed 432; XGBoost with the tree settings and seed of 01 (100 trees, max depth 6, learning rate 0.3, seed 432) and the intercept of each fit set to the share of class 1 among its training rows, `model_cv.base_score`) | 03 |
| Results 2.1, SI Section S5, SI Table S4, Extended Data Fig. 1: 26.8% misclassified within 6 m of the boundary against 3.7% at 90 m or more (ratio 7.3, 95% CI 3.9-19.5), near > far in all ten sites | 04 |
| Results 2.1, SI Sections S5 and S12: with 250 m blocks and a 200 m buffer, 51.6% against 15.5% (ratio 3.3, 2.0-6.5), near > far in all nine scored sites | 05 |
| Fig. 2: (a) outcome maps of Faz Quem Quer and Morro da Guaíba, (b) per-site test accuracy and the ten-site mean, (c) misclassification rate in 6 m signed-distance bins | 06 |

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

## Row order

The per-building tables of `main/01_features` (`products.features_gpkg`) are
read in their own row order, and every output of this folder follows that
order. Only the split of the main models refers to `uID`:

- **01, main per-site models.** The 80/20 split is drawn with
  `sklearn.model_selection.train_test_split` (`model_main`: unstratified,
  seed 432) on the buildings of the site ordered by `uID`, and the training
  rows are passed to XGBoost in the order `train_test_split` returns them
  (`favela_xai.model.main_split`). This is how the analysed models were
  fitted, whatever the row order of the table. Sorting the training rows
  changes the fitted model in the larger sites, because xgboost's hist
  quantile sketch depends on the row order.
- **03 and 05, out-of-fold and blocked predictions** (and SI S9, S12, S14 and
  S15): folds and random draws are assigned by row position, so they use the
  rows in table order.
- **The per-building tables of the analysis** list the buildings in the row
  order in which the per-site tables of the study were stored: uID order in
  nine sites; in Rocinha another order (all 36,048 buildings at another
  position). The out-of-fold results of the paper were computed in that
  order. A table rebuilt from the raw layers by `main/01_features` lists the
  buildings in uID order. That order does not change the main models, but it
  changes Rocinha's out-of-fold folds and random draws, and with them, slightly,
  the results of 03, 05 and SI S9, S12, S14 and S15.

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
- `model_main.fit_formal_as_class_1`, `model_cv.fit_formal_as_class_1`
  (true | false): class coding of the fit. true = formal settlement fitted as
  class 1, the coding of the analysed models; the favela probability and Tree
  SHAP toward favela are returned in both cases (the saved
  `<site>_model.json` keeps the coding of the fit).
- `model_cv.base_score` (training_label_mean | xgboost_default): intercept of
  every validation fit; training_label_mean = share of class 1 among the
  training rows, with which the validation results were computed.
