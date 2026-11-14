# appendix/S15_adjacency_sensitivity: sensitivity to the adjacency ratio

Tests how far the site-level attribution rankings, the boundary transition and
the concentration of errors at the boundary depend on the adjacency ratio
(Supplementary Section S15). The adjacency ratio (SI Section S3, item 19) is the
momepy building-adjacency statistic on the queen contiguity graph of the
tessellation cells, the setting of the analysis
(`indicators.adjacency_contiguity: tessellation`). Every site is analysed in
three specifications:

* reference: the 25 indicators as analysed;
* footprint contiguity: the adjacency ratio recomputed on the queen contiguity
  graph of the building footprints (`indicators.adjacency_contiguity: buildings`,
  the alternative setting of the configuration), the other 24 indicators as
  analysed;
* omitted: the other 24 indicators.

All three use the out-of-fold protocol of the boundary error analysis (Methods
4.7; stratified five-fold cross-validation within each site, as in
`main/02_classification/03_oof_predictions.py`), so that the prediction and the
attributions of every building come from a model not trained on it.

## Manuscript items

| Item | Script |
|---|---|
| SI Table S19: rank correlation with the reference (per site and pooled), sites keeping at least two of their three leading indicators, x₀ and w of the summed attributions, misclassification within 6 m and at 90 m or more | 01 |
| SI Section S15 and Methods 4.3: the values quoted in the text | 01 (`in_text_numbers.txt`) |

## Main results (SI S15)

* Rankings: Spearman's ρ with the reference is 0.93 on average over the ten
  sites (range 0.85-0.98) with footprint contiguity and 0.94 (0.87-0.99) over
  the 24 shared indicators with the indicator omitted, and 0.98 for the pooled
  ranking in both. Every site keeps at least two of its three leading
  indicators in both specifications, and, with the indicator omitted, its
  leading indicator among the other 24. SI S15 compares this agreement with that of the five-fold
  refits on 250, 500 and 1,000 m blocks with the 200 m buffer (0.86-0.90) and of
  random five-fold refits (0.95) in SI Table S16 (`appendix/S12_spatial_cv`,
  step 11).
* Boundary transition: x₀ of the summed spatial-distribution attributions is
  +4.9 m in the reference, +5.0 m with footprint contiguity and +4.6 m with the
  indicator omitted; for all indicators +4.3, +4.4 and +3.9 m. The six profiles
  have widths of 14.0-18.3 m. ΔS > 0 in all nine evaluable sites in every
  specification.
* Errors: with footprint contiguity 26.4% of the buildings within 6 m of the
  line are misclassified, against 3.6% at 90 m or more (26.8% and 3.7% in the
  reference, the rates of Fig. 2c); the rate is higher near the line in all ten
  sites in both specifications.

## Run order

From the repository root, after `main/01_features` (steps 01-05):

```bash
python appendix/S15_adjacency_sensitivity/01_adjacency_sensitivity.py   # SI Table S19 and the SI S15 values
```

Options: `--sites <key> ...` (a subset of the sites), `--n_jobs N` (XGBoost
threads) and `--recompute_adjacency`. The footprint-contiguity ratio of each
site is computed once with `favela_xai.features.compute_indicators` (which
computes all 25 indicators; only the adjacency ratio is kept) and stored in
`adjacency_footprint_contiguity/`; later runs reuse it unless
`--recompute_adjacency` is given. The out-of-fold fits (ten sites, three
specifications, five folds) take a few minutes. If
`main/02_classification/03_oof_predictions.py` has been run, the out-of-fold
probabilities of the reference specification are checked against its output.

## Inputs

- Per-site building tables of `main/01_features` (`products.features_gpkg`): `uID`, `label` and the 25 indicators.
- Footprints and tessellation cells of `main/01_features/02_tessellation.py` (`<site>_buildings_tessellated.gpkg`, `<site>_tessellation.gpkg`), for the footprint-contiguity ratio.
- Signed distance to the official 2013 favela boundary (`products.signed_distance`, `main/01_features/05_signed_distance.py`).
- Optional: out-of-fold predictions of `main/02_classification` (`products.oof_predictions`), for the check of the reference specification.

## Outputs (`outputs/appendix/S15_adjacency_sensitivity/`)

| File | Content |
|---|---|
| `table_S19_adjacency_sensitivity.tex` | SI Table S19 |
| `in_text_numbers.txt` | the values quoted in SI Section S15 and Methods 4.3, and the result of the check of the reference |
| `rankings_per_site.csv` | site × specification: Spearman's ρ with the reference, the three leading indicators of the reference and of the specification, their overlap, same leading indicator |
| `rankings_pooled.csv` | indicator: pooled shares (mean of the site shares) of the reference over 25 and over 24 indicators and of the two other specifications |
| `importance_shares.csv` | site × specification × indicator: mean \|SHAP\| over the favela buildings and share |
| `boundary_transition.csv` | specification × group (spatial-distribution indicators, all indicators): ΔS, x₀, w, sites with ΔS > 0 among the evaluable sites |
| `boundary_transition_by_site.csv` | specification × group × site: ΔS of the site profile |
| `boundary_profiles.csv` | specification × group × 2.5 m bin: cross-site mean and s.d. of the summed attributions, number of sites |
| `boundary_errors.csv`, `boundary_errors_by_site.csv` | reference and footprint contiguity, pooled and per site: buildings and misclassified buildings within 6 m and at 90 m or more, rates, sites with the higher rate near the line |
| `adjacency_footprint_contiguity/<site>.parquet` | `uID` and the adjacency ratio on footprint contiguity |

## Definitions and settings (`config.yaml`)

- Folds and classifier: `oof` and `model_cv`, the configuration of the out-of-fold error analysis (`favela_xai.model.fit_predict_cv`). The folds depend on the labels only, so they are the same in the three specifications. Attributions: exact Tree SHAP of each fold's model on its held-out buildings, log-odds, positive toward favela (`favela_xai.attribution.tree_shap`).
- Adjacency ratio: `indicators` (200 m neighbourhood band; `adjacency_contiguity` must be `tessellation`, the analysed setting). The script recomputes the ratio with `adjacency_contiguity` set to `buildings` in a copy of the configuration; `config.yaml` itself is not changed.
- Rankings: share of each indicator in the mean \|SHAP\| over the favela buildings of a site (`importance.population`, as in Extended Data Fig. 2), compared with the reference over the indicators the two specifications share (the reference shares renormalised over the 24 for the omitted specification); pooled ranking = mean of the site shares; three leading indicators (`spatial_cv.ranking.top_k`).
- Boundary transition: summed attributions of the indicators of the spatial-distribution dimension (SI Table S2; four indicators, three when the ratio is omitted) and of all indicators; profiles, ΔS, x₀ and w as for Fig. 4 and SI Table S5 (`boundary_profiles`, `favela_xai.boundary`).
- Boundary errors: a building is misclassified when its out-of-fold favela probability (≥ 0.5 = favela) differs from its label; distance bands from `boundary_errors.band_edges_m` and `near_far_bands` (within 6 m; 90 m or more). As in SI Table S19, the error rows are given for the two specifications of the indicator.
- The folder has no settings of its own.
