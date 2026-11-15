# appendix/S11_terrain: terrain ablation

Tests whether the building-scale morphological signal is a proxy for terrain:
per-building elevation and slope from a 5 m digital terrain model, a
terrain-only classifier, a combined morphology + terrain classifier, and the
overlap of the favela and non-favela terrain distributions. The scripts compute
the per-site test accuracy of the three classifiers (Morph: 25 indicators;
Terrain: nine terrain features; Combined: both), the Tree-SHAP feature
importance of the Terrain and Combined classifiers, and the Spearman
correlations of the per-site slope gap and terrain overlap with per-site
accuracy. The results are reported in Supplementary Section S11.

## Manuscript items

| Item | Script |
|---|---|
| Methods 4.10 "Terrain ablation"; Results paragraph on terrain (terrain-only mean accuracy; accuracy difference between the Combined and Morph classifiers) | 02, 06 |
| SI Table S9: Morph / Terrain / Combined accuracy per site, and the in-text numbers of SI S11 | 06 |
| SI Fig. S14: slope gap vs per-site accuracy (Experiment 1) | 03 |
| SI Fig. S15: terrain distribution overlap (OVL) vs accuracy | 04 |
| SI Fig. S16: Tree-SHAP importance of the terrain-only classifier (Experiment 2) | 05 |
| SI Fig. S17: accuracy of the three classifiers | 05 |
| SI Fig. S18: top-20 features of the combined classifier (Experiment 3) | 05 |
| SI Table S1: mean building slope of favela and formal-settlement buildings (`exp1_global_site_summary.csv`) | 02 |

## Run order

From the repository root, after `main/01_features` and `main/02_classification`:

```bash
python appendix/S11_terrain/01_building_terrain.py                # slope raster, nine terrain features per building [--overwrite]
python appendix/S11_terrain/02_terrain_ablation.py                # Morph / Terrain / Combined classifiers, Tree SHAP, slope gap
python appendix/S11_terrain/03_fig_S14_slope_gap.py               # SI Fig. S14 and Spearman correlations
python appendix/S11_terrain/04_fig_S15_terrain_separability.py    # SI Fig. S15 and OVL table
python appendix/S11_terrain/05_fig_S16_S17_S18.py                 # SI Figs S16-S18
python appendix/S11_terrain/06_table_S9.py                        # SI Table S9 and in-text numbers
```

`terrain_common.py` holds the helpers shared by these scripts (file names,
feature lists, classifier and split of the ablation, accuracy source, figure
site names); it is not run on its own. Step 01 computes the slope over the
whole terrain model in memory (several GB of RAM) and writes it once; later
runs reuse it.

## Inputs

- 5 m LiDAR digital terrain model (GeoTIFF, EPSG:31983), available from the Instituto Municipal de Urbanismo Pereira Passos (IPP) on request; set `paths.raw.elevation_raster` in `config.yaml`.
- Per-site building tables of the main analysis (`main/01_features`, `products.features_gpkg`).
- Per-site test accuracy of the main classifiers, SI Table S3 (`main/02_classification`, `products.site_metrics`); used by 03, 04 and 06.

## Outputs (`outputs/appendix/S11_terrain/`)

Accuracies in the tables are fractions (0-1); the figures draw percentages.

| File | Content | Script |
|---|---|---|
| `slope_5m.tif` | slope (degrees) on the grid of the terrain model | 01 |
| `<site>_buildings_with_terrain.gpkg` | building table with `elev_mean`, `elev_min`, `elev_max`, `elev_range`, `elev_std`, `slope_mean`, `slope_max`, `slope_median`, `slope_std` | 01 |
| `summary_three_model_comparison.csv` | per site: test accuracy, favela F1 and macro F1 of the three classifiers | 02 |
| `exp1_global_site_summary.csv` | per site: mean slope of favela and non-favela buildings, slope gap | 02 |
| `exp1_pooled_test_buildings.csv` | test buildings with label, Morph prediction and terrain features | 02 |
| `exp2_terrain_shap_importance.csv`, `exp3_combined_shap_importance.csv` | mean \|Tree SHAP\| per feature and site, and the mean over sites | 02 |
| `exp1_global_correlations.csv`, `exp1_fig3_slope_gap_vs_accuracy.pdf` | Spearman correlations; SI Fig. S14 | 03 |
| `terrain_overlap_by_site.csv`, `terrain_overlap_correlations.csv`, `terrain_separability_vs_accuracy.pdf` / `.svg` | OVL per site and its correlation with accuracy; SI Fig. S15 | 04 |
| `exp2_fig6_terrain_shap_importance.pdf`, `summary_fig8_three_model_comparison.pdf`, `exp3_fig7_combined_shap_top20.pdf` | SI Figs S16, S17, S18 | 05 |
| `table_S9_terrain_only_tabular.tex`, `S11_in_text_numbers.txt` | SI Table S9 (`tabular` environment); numbers quoted in SI S11 and the Results | 06 |

## Definitions and settings (`config.yaml`: `terrain`, `appendix_S11_terrain`, `model_main`, `shap`)

- Terrain features (SI S11 "Data"): slope in degrees from finite-difference gradients over the 5 m cells; nine zonal statistics per footprint (`rasterstats.zonal_stats`, `all_touched`).
- Ablation: per site, Morph (25 indicators), Terrain (9 features) and Combined (34 features) classifiers on one shared train/test split; missing values replaced by training medians.
- Tree SHAP: exact Tree SHAP (log-odds) on the test buildings; mean \|SHAP\| per site, then the mean over sites.
- Slope gap: \|mean slope of favela buildings - mean slope of non-favela buildings\| over all buildings of the site.
- OVL: integral of min(f_favela, f_non-favela) of Gaussian kernel-density estimates of mean slope and mean elevation.
- `terrain.classifier`: `model_main` (the main per-site model of Methods 4.4: same settings, split, training-row order and class coding as `main/02_classification`, so the Morph column of SI Table S9 equals SI Table S3) | `model_cv` (XGBoost, 100 trees, seed 432, intercept of `model_cv.base_score`, unstratified 80/20 split with seed 432).
- `appendix_S11_terrain.ovl_population`: `all_buildings` | `test_set` (buildings used for the OVL).
- `appendix_S11_terrain.figure_accuracy`: `table_S3` | `ablation_morph` (accuracy drawn in Figs S14-S15).
- `appendix_S11_terrain.morph_feature_labels`: `keys` | `table_S2` (tick labels of the indicators in Fig. S18).
