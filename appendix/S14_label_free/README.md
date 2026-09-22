# appendix/S14_label_free: label-free clustering of the morphological indicators

Tests how much of the morphological contrast is recovered without the favela
labels (Supplementary Section S14; Main Methods "Label-free clustering of the
morphological indicators"). In each site the buildings are clustered on the 25
indicators alone (rank transform to a standard normal distribution; k-means and
Ward with 2 to 6 clusters; three weightings of the indicators). The labels and
the official 2013 boundary are used only afterwards, to score and to orient
the clusters.

## Manuscript items

* Supplementary Table S18, label-free clustering by weighting scheme: 02 (data), 03 (table).
* Supplementary Fig. S23, label-free clustering against the official boundary: 02 (data), 04 (figure).
* Values in SI Section S14: printed at the end of 01 and 02.

## Main results

* The labels are only weakly recovered: mean best adjusted Rand index at most
  0.08 and mean best balanced accuracy 0.63 to 0.65, against 0.904 for the
  supervised classifier; mean score AUC 0.55 to 0.63, against 0.969 (02, 03).
* The label-free contrast differs from the supervised attribution: convexity
  is among the three most separating indicators in every site, the two
  rankings are negatively correlated on average (Spearman's ρ = −0.36), share
  their leading indicator in none of the ten sites, and the cluster contrast
  has no consistent direction across sites (mean pairwise cosine −0.10; 01).
* A gradient at the boundary remains: the pooled profile of the cluster score
  correlates with the supervised profile (r = 0.879, 0.942, 0.952 for the three
  weightings), with an inside-outside step of 0.17 to 0.48 SD of the score
  (01, 02).

## Run order

From the repository root, after `main/01_features`, `main/02_classification`
and (for 03 and 04) `appendix/S12_spatial_cv`:

```bash
python appendix/S14_label_free/01_clusters_and_attribution.py   # equal-weight clusters: alignment, contrast vs attribution
python appendix/S14_label_free/02_weighting_sensitivity.py      # three weightings: Table S18 values, profiles, per-site r
python appendix/S14_label_free/03_table_S18.py                  # Supplementary Table S18
python appendix/S14_label_free/04_figure_S23.py                 # Supplementary Fig. S23
```

`label_free_common.py` holds the shared helpers (per-site table, rank
transform, weightings, clustering and scores, 6 m boundary profiles,
supervised reference); it is not run on its own.

## Inputs and outputs

* Inputs: label, 25 indicators and Tree SHAP values of the main model
  (`products.shap_gpkg`) and out-of-fold favela probabilities
  (`products.oof_predictions`), both from `main/02_classification`; signed
  distance to the 2013 boundary (`products.signed_distance`, `main/01_features`);
  fold table of the SI S12 cross-validation (`products.spatial_cv_folds`,
  `appendix/S12_spatial_cv`; steps 03 and 04 only).
* Outputs (`outputs/appendix/S14_label_free/`): `table_S18_label_free.tex`
  (Supplementary Table S18), `Figure_R2_labelfree.pdf` (Supplementary Fig. S23,
  with a PNG preview), and the CSV/NPZ tables listed in each script's docstring.
* Settings: `config.yaml` sections `label_free` and `appendix_S14_label_free`.
