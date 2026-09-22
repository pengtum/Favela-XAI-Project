# main/03_site_signatures: site-specific morphological signatures

Summarises the per-building Tree-SHAP attributions of the ten per-site
classifiers: which indicators carry the favela classification in each site,
which indicator dominates each favela building, and how two contrasting
favelas (Rocinha and Barão) differ in space and in their SHAP dependence.

## Manuscript items

| Item | Script |
|---|---|
| Fig. 3a: dominant feature of every favela building in the eight other sites | 02 |
| Fig. 3b: Rocinha and Barão, dominant-feature map and Tree-SHAP maps of mean interbuilding distance and squareness, with SHAP-dependence insets | 02 |
| Results 2.2: importance ranking (mean interbuilding distance and adjacency lead across sites; squareness in Barão; perimeter wall in Grotão de Costa Barros; interbuilding distance and adjacency in Rocinha and Catumbi) | 01 |
| Results 2.2: reversal of the interbuilding-distance dependence between Barão (toward favela) and Rocinha (toward formal) | 02 |
| Extended Data Fig. 2: per-site importance for the favela class (values; the figure is drawn by `appendix/S06_attribution_analyses/01_persite_importance.py`) | 01 |
| SI Section S6, first paragraph: global ranking and secondary tier | 01 |

## Run order

From the repository root, after `main/02_classification` (the two scripts are
independent):

```bash
python main/03_site_signatures/01_importance_tables.py   # per-site importance, cross-site order, dominant-feature counts
python main/03_site_signatures/02_figure3.py             # Fig. 3 and the binned dependence curves of its insets
```

## Inputs

- `outputs/main/02_classification/<site>_buildings_shap.gpkg`: label, the 25
  indicators, the 25 `shap_<indicator>` columns (positive = toward favela) and
  the footprint geometry.
- Official 2013 favela polygons (`data/raw/Limite_Favelas_2013/`, see
  `data/README.md`): boundary line on the Fig. 3b SHAP maps.

## Outputs (`outputs/main/03_site_signatures/`)

| File | Content | Script |
|---|---|---|
| `persite_importance_favela.csv` | site × indicator: mean \|SHAP\| over the site's favela buildings, share of the site total, rank | 01 |
| `persite_top3.csv` | the three leading indicators of each site | 01 |
| `crosssite_importance.csv` | indicator: cross-site mean of the per-site values (Extended Data Fig. 2 order) and mean \|SHAP\| pooled over all buildings (Fig. 4 panel order) | 01 |
| `dominant_feature_counts.csv` | site × indicator: favela buildings whose dominant feature is that indicator | 01 |
| `figure3.pdf`, `figure3.svg` | Fig. 3 (the SVG keeps the text editable) | 02 |
| `figure3_legend_features.csv` | the colour-coded dominant features and their favela-building counts | 02 |
| `figure3_dependence_curves.csv` | binned SHAP dependence of the Fig. 3b indicators, pooled and per site | 02 |

## Definitions and settings (`config.yaml`)

- Attributions: exact Tree SHAP (log-odds) of the per-site XGBoost models (Methods 4.4, 4.6).
- Site importance (`importance`): mean \|SHAP\| over the site's favela buildings and its share of the site total; the cross-site order is the mean over the ten sites.
- Dominant feature (`importance`): the indicator with the largest signed SHAP value toward favela.
- Fig. 3 layout (`main_03_site_signatures`): panel-b sites and indicators, panel-a order, zoom window, number of colour-coded features, SHAP colour limit (percentile of \|SHAP\|) and inset binning.
- `main_03_site_signatures.outline`: `official_2013` draws the official 2013 favela boundary (the line of the signed distance, Methods 4.8); `label_hull` draws a hull of the favela-labelled footprints.
