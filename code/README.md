# Code — *Building-scale morphological signatures of informal settlements*

Scripts used to produce the figures and to prepare the manuscript for
submission to *Nature Cities*. Organised into two folders:

```
code/
├── figures/          # scripts that generate the manuscript figures
└── manuscript_prep/  # one-off scripts used to edit the LaTeX / build submission assets
```

> **Important — paths.** The `figures/` scripts here are **copies for reference /
> code-availability**. The canonical, runnable copies live in `../Figures/`, and
> most of them resolve input/output paths **relative to that directory**
> (`HERE = os.path.dirname(os.path.abspath(__file__))`) or to absolute project
> paths. To re-run a figure script, run it from `../Figures/` (or copy it back
> there) so it finds its data and writes its PDF next to the other figures.

## Data sources (not included here)

The figure scripts read from the project's analysis outputs, chiefly:

- **Per-building SHAP geopackages** — `results/all_features/<site>/<site>_updated_buildings_with_shap.gpkg`
  (EPSG:31983; columns: `label`, `predicted_label`, the 25 raw morphological
  indicators, 25 `shap_<feature>` attributions, geometry; positive SHAP = toward favela).
- **Terrain / Bettencourt / temporal CSVs** — under `Final analysis/…`
  (e.g. `terrain analysis/experiments/exp1_global_site_summary.csv`,
  `Experiment_COVID19/Data/sensitivity_bettencourt_k*.csv`).
- Pre-computed intermediates in `../Figures/` (e.g. `boundary_error_data.parquet`).

## figures/ — figure-generation scripts

**Figure 1 — concept / pipeline** (Peng assembles the final composite in Illustrator from these element maps)
- `regen_fig1.py`, `regen_fig1_maps.py` — study-area + per-dimension feature element maps
- `regen_fig1_sitemap.py` — ten-site half-disk study map

**Figure 2 — building-level detection performance**
- `prep_fig2_data.py`, `prep_fig2_basemap.py` — data + basemap prep
- `regen_fig2_final.py`, `regen_fig2_v2.py`, `regen_fig2_variants.py`, `regen_fig2_stitch.py` — panels + stitching

**Figure 3 — site dominant-feature signatures**
- `regen_dominant_feature_map.py` — per-building dominant-feature maps
- `regen_fig3_final.py`, `regen_fig3_composite.py`, `regen_fig3_merged.py` — assembled Fig 3 (contains the `GROUPS` feature→GBASE-dimension mapping)

**Figure 4 — boundary spillover (SHAP vs signed distance)**
- `regen_fig4_spillover_metrics.py`, `regen_fig4_spillover_polygon.py` — spillover curves + transition metrics (x₀, w, ΔS)
- `spillover_metric_explainer.py` — schematic explainer of the transition metrics

*(Figure 5, the 2D deep-feature "Favela DNA" plot, is produced by the deep-feature
analysis pipeline outside this folder.)*

**Supplementary figures**
- `regen_persite_importance_favela.py` — per-site feature importance
- `regen_shap_dependence.py`, `regen_dependence_heterogeneity_scan.py`,
  `regen_feature_dependence_panels.py`, `regen_interbuild_dependence.py`,
  `regen_interbuild_shapmap.py` — SHAP dependence / heterogeneity panels
- `regen_casestudies.py` — building-level case studies
- `regen_failure_fig.py` — misclassification (failure) analysis
- `regen_boundary.py` — misclassifications by distance-to-boundary (stacked bars)
- `regen_bettencourt_dist.py` — block-complexity index *k* distribution (4 panels)
- `regen_slope_gap.py` — slope-gap vs per-site accuracy
- `regen_terrain_separability.py` — terrain-only separability vs accuracy

## manuscript_prep/ — submission-preparation scripts

One-off scripts run once against `../01_main_naturecities.tex` /
`../03_supplement_naturecities.tex` (they hardcode the manuscript paths):

- `safe_edits.py` — six "safe" abstract/intro edits (fix finding-1/3 self-contradiction; reframe the first gap; de-jargon; remove redundancy).
- `framing_edits.py` — four framing edits (discovery-first "Here we" pivot; abstract broad-significance close; de-grandiosed opening question; forward-lede in intro ¶1).
- `del_s1.py` — deletes old Supplementary Figure S1 (the 25-indicator illustration) and decrements all downstream hardcoded `Figure SX` references in the main text (single-pass, collision-safe).
- `build_appendix.py` — assembles the self-contained `../Appendix/` bundle (supplement `.tex` with a bibliography added, `02_references.bib`, and all referenced figures + the `\input` table).
- `wordcount.py` — approximate main-text word count (strips LaTeX commands, captions, math, citations), reported per section.
- `make_coverletter.py` — generates `../cover_letter_NatureCities.docx` (python-docx).
