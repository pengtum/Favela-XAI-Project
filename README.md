# Favela-XAI-Project

Code for:

> **Building-scale morphological signatures of informal settlements**
> Peng Luo, Fábio Duarte, Luiz Carlos Teixeira Coelho, Carlo Ratti

Using only open building footprints, we compute 25 morphometric indicators for
188,735 buildings across ten favelas in Rio de Janeiro, train per-site
XGBoost classifiers to distinguish favela from formal buildings, and use
Tree SHAP to recover, for every building, which footprint features drive its
classification.

## Repository structure

```
└── code/
    ├── figures/           # figure-generation scripts (Figures 1–4 + supplementary)
    ├── manuscript_prep/   # manuscript-preparation utilities
    └── README.md          # per-script documentation
```

## Data

All source data are produced by the Instituto Municipal de Urbanismo Pereira
Passos (IPP) of the City of Rio de Janeiro.

- **Building footprints (2013)** — *Edificações 2013*, Data.Rio:
  <https://www.data.rio/datasets/124f1607b57d4292af5c0dfc2ff8572c_0/about>
- **Favela boundary polygons** — *Limite Favelas 2013* / *2019*, Data.Rio:
  <https://www.data.rio/datasets/PCRJ::limite-favelas-2013/about> ·
  <https://www.data.rio/datasets/PCRJ::limite-favelas-2019/about>
- **5 m LiDAR digital terrain model** (terrain ablation only): not openly
  distributed — request from IPP: <https://www.rio.rj.gov.br/web/ipp>
- **Derived per-building datasets** (the 25 morphological indicators and
  Tree-SHAP attributions): available from the corresponding author
  (fduarte@mit.edu) upon reasonable request.

## Reproducing the analysis

### Step 0 — Environment

Python 3.10+. Install the dependencies:

```bash
pip install geopandas momepy osmnx xgboost scikit-learn shap scipy matplotlib pandas pyarrow
```

Versions used in the paper: `geopandas` 0.14, `momepy` 0.7, `osmnx` 2.1,
`xgboost` 1.7, `scikit-learn` 1.3, `shap` 0.45, `scipy` 1.11.

### Step 1 — Source data

Download from [Data.Rio](https://www.data.rio): *Edificações 2013* (building
footprints), *Limite Favelas 2013* (ground-truth favela polygons), and
*Limite Favelas 2019* (boundary-revision analysis only). Clip the footprints
to the ten study sites (site composition in Supplementary Section S2) and
label each building `favela` / `not-favela` by intersecting its centroid with
the 2013 favela polygons.

### Step 2 — Morphological indicators

For each site, compute the 25 footprint-level indicators with `momepy`
(definitions and formulas: Supplementary Section S3). This yields one
building × 25 feature table per site.

### Step 3 — Per-site classification

For each site independently: 80/20 stratified train/test split, XGBoost
binary classifier on the 25 indicators (hyperparameters in Methods,
"Explainable-AI framework"). Held-out test accuracy should reproduce
Supplementary Table S2 (mean ≈ 0.933).

### Step 4 — Tree SHAP attribution

Apply Tree SHAP (`shap.TreeExplainer`) to each site's trained model over all
buildings of that site, yielding the per-building 25-dimensional attribution
vector ("deep features"; positive = toward favela).

### Step 5 — Figures and analyses

Scripts in `code/figures/` regenerate the manuscript figures; each script
documents its expected inputs at the top (`code/README.md` maps scripts to
figures). They read the per-site building tables produced in Steps 2–4 —
edit the `PROJ`/path constants at the top of each script to point to your
local copies. Key entry points:

- `regen_fig2_*.py` — detection performance (Figure 2);
- `regen_fig3_final.py`, `regen_dominant_feature_map.py` — site-specific signatures (Figure 3);
- `regen_fig4_spillover_metrics.py` — boundary-spillover profiles and transition metrics (Figure 4);
- `regen_boundary.py`, `regen_bettencourt_dist.py`, `regen_slope_gap.py`, … — supplementary figures.

## Citation

Please cite the paper above. A citation file will be added once the paper is
published.

## License

MIT (see `LICENSE`).
