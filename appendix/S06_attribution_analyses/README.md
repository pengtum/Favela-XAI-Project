# appendix/S06_attribution_analyses: attribution importance, dependence and boundary direction

Site-level summaries of the Tree-SHAP attributions of the main per-site
classifiers (SI Section S6): per-site importance rankings, the classification
probability across the official 2013 favela boundary, the pooled and per-site
SHAP dependence of the 25 indicators, and the direction of the boundary
transition in each site's attribution space. Script 07 computes the recorded
heights of machine-room and ordinary building polygons quoted in SI Section S6.

## Manuscript items

| Item | Script |
|---|---|
| Extended Data Fig. 2 (Results 2.2): per-site mean \|SHAP\| over favela buildings, indicators in a common cross-site order, coloured by morphometric dimension; interbuilding distance and adjacency lead in Rocinha and Catumbi, squareness in Barão | 01 |
| SI Fig. S6 (SI S6, Methods 4.8): share of buildings classified favela against signed distance; about 0.9 deep inside the polygons and near zero well outside, dropping sharply at the line but staying above its far-field level over the first metres outside | 02 |
| SI Fig. S7: pooled binned-mean SHAP dependence of the 25 indicators, band spanning the per-site trends | 03 |
| SI Fig. S8: per-site dependence curves and heterogeneity score (above 2 for interbuilding distance and adjacency) | 04 |
| SI Section S6.1: inward (26%) and outward (33%) displacement of the near-boundary groups, net skew +7 ± 25 points across sites, sensitivity sweep | 05 |
| SI Fig. S9: displacement and net skew by site | 06 |
| SI Section S6: recorded heights of machine-room and ordinary building polygons (among the polygons with a recorded height, median 14.8 m against 4.0 m; machine rooms higher in each of the ten sites) | 07 |

Computed elsewhere: the per-site importance values of ED Fig. 2
(`main/03_site_signatures/01_importance_tables.py`); SI Table S5 and the
transition values of SI S6 (`main/04_boundary_profiles`).

## Run order

From the repository root, after `main/01_features`, `main/02_classification`
and `main/03_site_signatures`:

```bash
python appendix/S06_attribution_analyses/01_persite_importance.py        # ED Fig. 2
python appendix/S06_attribution_analyses/02_boundary_probability.py      # SI Fig. S6
python appendix/S06_attribution_analyses/03_shap_dependence.py           # SI Fig. S7
python appendix/S06_attribution_analyses/04_dependence_heterogeneity.py  # SI Fig. S8
python appendix/S06_attribution_analyses/05_boundary_direction.py        # SI Section S6.1 values
python appendix/S06_attribution_analyses/06_figure_boundary_direction.py # SI Fig. S9 (after 05)
python appendix/S06_attribution_analyses/07_recorded_heights.py          # SI Section S6 recorded heights (needs main/01_features only)
```

`attribution_common.py` holds the shared helpers (per-site tables, signed
distance join, panel and site order, figure style).

## Inputs

- `products.shap_gpkg` (`main/02_classification`): per building `uID`, `label`, predicted class, the 25 indicators and the 25 `shap_<indicator>` columns (positive = toward favela).
- `products.signed_distance` (`main/01_features`): signed distance to the official 2013 favela boundary (negative inside).
- `products.importance_favela` (`main/03_site_signatures`; script 01 only): per-site mean \|SHAP\| over favela buildings.
- `products.features_gpkg` (`main/01_features`; script 07 only): per building the cadastre class `Tipo` and the recorded height `ALTURA` (m) of the 2013 building layer.

## Outputs (`outputs/appendix/S06_attribution_analyses/`)

| File | Content | Script |
|---|---|---|
| `persite_importance_favela.csv`, `Figure_X_persite_importance_favela.pdf` | per-site importance and rank in the common order; ED Fig. 2 | 01 |
| `boundary_probability_profile.csv`, `Figure_R1_boundary_probability.pdf` | per 2.5 m bin: pooled and per-site share classified favela; SI Fig. S6 | 02 |
| `dependence_pooled_trends.csv`, `Figure_X_shap_dependence.pdf` | per indicator and bin: pooled trend, per-site minimum and maximum; SI Fig. S7 | 03 |
| `dependence_heterogeneity_scores.csv`, `dependence_persite_curves.csv`, `Figure_X_dependence_heterogeneity_scan.pdf` | heterogeneity scores and per-site curves; SI Fig. S8 | 04 |
| `boundary_direction_by_site.csv`, `boundary_direction_sensitivity.csv`, `boundary_direction_summary.txt` | per-site displacements and net skew, sweep, reported values | 05 |
| `Figure_R1_boundary_direction.pdf` | SI Fig. S9 | 06 |
| `recorded_heights_by_site.csv` | per site and over all sites: polygons, polygons with a recorded height and their median height, for machine rooms and ordinary building polygons | 07 |

## Settings (`config.yaml`: `importance`, `boundary_profiles`, `dependence`, `boundary_direction`, `appendix_S06_attribution_analyses`)

- Attributions: exact, path-dependent Tree SHAP (no background sample; log-odds) of the main per-site XGBoost models (Methods 4.4, 4.6).
- ED Fig. 2: mean \|SHAP\| over each site's favela buildings; common order = mean over sites of the per-site values.
- SI Fig. S6: 2.5 m bins on [−40, 40] m, bins with fewer than 25 buildings omitted; band = per-site minimum to maximum. `probability.buildings`: `all` (every building) | `test` (held-out test buildings only).
- SI Fig. S7: 12 equal bins between the pooled 1st and 99th percentiles, at least 20 buildings per bin; up to 1,200 points per class drawn (seed 42). `dependence_tails_in_end_bins`: `true` (values beyond the range counted in the end bins) | `false` (left out).
- SI Fig. S8: 15 equal bins between the pooled 1st and 98th percentiles, at least 25 buildings per bin; score = mean over bins of the cross-site s.d. (ddof 0) of the binned means.
- Panel order (SI Figs S7, S8): dimensions by the mean of their indicators' pooled mean \|SHAP\|, indicators by their own within a dimension (as Fig. 4).
- SI S6.1: reference classes d < −50 m (favela) and d > +50 m (formal); boundary zone \|d\| ≤ 20 m split by label; at least 30 buildings per group. With C_G the mean SHAP vector of group G and v = C_formal − C_favela: t(G) = (C_G − C_favela)·v / v·v; inward = t(near favela), outward = 1 − t(near formal), net skew = outward − inward. Pooled = unweighted mean over sites; ± = s.d. of the per-site net skews.
- Sweep: reference threshold 50-150 m × boundary zone 10-30 m. `direction.sweep_design`: `grid` (every combination) | `one_at_a_time` (each list with the other setting at its primary value). `direction.min_group_rule`: `flag` (keep and flag a site below the minimum group size) | `exclude` (leave it out of that design point).
- SI S6 recorded heights (script 07): machine rooms = `Tipo` containing `CASA_DE_MAQUINA`; ordinary building polygons = `Tipo` containing `EDIFICACAO` (`EDIFICACAO`, `EDIFICACAO_FAVELA`); polygons with `ALTURA` > 0; medians over the polygons of all sites and within each site.
