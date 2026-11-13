# appendix/S12_spatial_cv: spatial cross-validation

Re-validates the per-site classifiers with spatially separated folds and tests whether the
site-level attribution rankings hold under that separation. Designs: square blocks of
100-1,000 m in three grid placements with training buffers of 0-400 m and matched
random-removal controls; hexagons, k-means units, five k-means folds and ten folds; the
held-out favela component; the pooled leave-site-out. Every fold is refitted (XGBoost, 100
trees, seed 432, training-median imputation) and explained with out-of-fold Tree SHAP.

## Manuscript items

| Item | Script |
|---|---|
| Methods 'Validation under spatial separation'; SI Section S12 | all |
| Supplementary Tables S10-S14 (designs, blocks x buffers, sensitivity designs, scoring coverage, fold summary) | 05 |
| Supplementary Fig. S19 (accuracy under spatial separation) | 05, 06 |
| Supplementary Table S15 and Fig. S20 (distance to training, error by distance) | 07, 08 |
| Supplementary Fig. S21 (held-out favela component) | 09 |
| Supplementary Table S16 (ranking stability) | 11 |
| Extended Data Fig. 3 (rankings under spatial cross-validation) | 11, 13 (panel a: `ed3_schematic.py`) |
| Extended Data Fig. 4 (per-site importance shares) | 11, 14 |
| Per-fold record summarised in SI Table S14 | 15 |

Main results (both scoring criteria, site-bootstrap 95% intervals):

* Balanced accuracy 0.644 (0.582-0.714, nine sites) for the primary design (250 m blocks, 200 m buffer) against 0.898 for a random 80/20 split; held-out favela component 0.650; pooled leave-site-out 0.702 (step 05).
* Spearman's rho between the out-of-fold and the main-text site rankings: 0.874, 0.902 and 0.864 for 250, 500 and 1,000 m blocks with the 200 m buffer, against 0.910 for the split-refit null (step 11).
* The three leading indicators keep their direction in 73 of 81 site-indicator-placement combinations (step 12).

Steps 05, 07, 09 and 11-14 also write the values quoted in the text and captions to `in_text_numbers.txt` or `*caption_numbers.txt`. The blocked boundary error rates (Results 2.1, SI S5) use the same fold maker and are computed in `main/02_classification/05_blocked_error_rates.py`.

## Run order

From the repository root, after `main/01_features` and `main/02_classification`:

```bash
python appendix/S12_spatial_cv/01_build_site_tables.py    # per-site tables: centroids, labels, signed distance, indicators, main-model SHAP
python appendix/S12_spatial_cv/02_component_groups.py     # held-out favela component groups
python appendix/S12_spatial_cv/03_run_folds.py            # fit every accuracy fold (--part, --sites, --n_jobs, --status)
python appendix/S12_spatial_cv/04_collect_folds.py        # fold record, out-of-fold predictions, leave-site-out summary
python appendix/S12_spatial_cv/05_accuracy_tables.py      # scoring; Supplementary Tables S10-S14
python appendix/S12_spatial_cv/06_figure_S19.py           # Supplementary Fig. S19
python appendix/S12_spatial_cv/07_distances.py            # Supplementary Table S15; data of Fig. S20
python appendix/S12_spatial_cv/08_figure_S20.py           # Supplementary Fig. S20
python appendix/S12_spatial_cv/09_figure_S21.py           # Supplementary Fig. S21
python appendix/S12_spatial_cv/10_oof_shap.py             # out-of-fold Tree SHAP of every ranking run (--sites, --runs, --n_jobs, --status)
python appendix/S12_spatial_cv/11_ranking_stability.py    # Supplementary Table S16; ranking values
python appendix/S12_spatial_cv/12_direction_survival.py   # direction of the attributions
python appendix/S12_spatial_cv/13_figure_ED3.py           # Extended Data Fig. 3
python appendix/S12_spatial_cv/14_figure_ED4.py           # Extended Data Fig. 4
python appendix/S12_spatial_cv/15_supplementary_data.py   # per-fold record (two CSV files)
```

Steps 03 and 10 fit several thousand models. They write one shard per site and job and skip existing shards, so an interrupted run resumes with the same command, and different `--sites` can run in parallel.

Helpers (imported): `s12_common.py` (folders, designs, scoring, LaTeX tables), `s12_figstyle.py` (style of Figs S19-S20), `ed3_schematic.py` (Extended Data Fig. 3a). Fold makers, buffers, matched controls, scoring criteria and the site bootstrap are in `favela_xai.spatial_cv`, `favela_xai.model` and `favela_xai.stats`.

## Inputs and outputs

Inputs: per-site feature tables and signed distances (`main/01_features`), Tree SHAP of the main per-site models (`main/02_classification`), and the official 2013 favela polygons (`paths.raw.favelas_2013`; source in `data/README.md`).

Outputs, under `outputs/appendix/S12_spatial_cv/`:

* `tables/`: LaTeX files of Supplementary Tables S10-S16; `figures/`: PDF (and PNG preview) of Supplementary Figs S19-S21 and Extended Data Figs 3-4;
* `folds_all.parquet` (fold record), `predictions_oof.parquet`, `supplementary_data/` (per-fold record);
* `accuracy/`, `distances/`, `component/`, `ranking/`, `direction/`: plotted values and quoted numbers;
* intermediate files: `site_tables/`, `component_groups/`, `fold_shards/`, `prediction_shards/`, `shap_fold_sums/`, `shap_buildings/`, `logs/`.

## Settings

All settings are in `config.yaml`: `spatial_cv` (block sizes, placements, buffers, primary design, alternative units, scoring, distances, ranking seeds), `model_cv` (classifier of every fold) and `appendix_S12_spatial_cv` (polygon fields, ranking runs, figure options). `spatial_cv.ranking.null_classifier` selects the model of the split-refit null:

* `model_cv`: validation-pipeline model on an unstratified 80/20 split;
* `model_main`: the Methods 4.4 model on a stratified 80/20 split.
