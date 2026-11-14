# main/04_boundary_profiles: boundary profiles of the morphological signature

Assigns every building its signed distance to the official 2013 favela
boundary, averages each indicator's Tree-SHAP contribution in 2.5 m distance
bins (per site, then across the ten sites) and summarises each profile by its
strength |ΔS|, sign-flip distance x₀ and transition width w (Methods 4.8).

## Manuscript items

| Item | Script |
|---|---|
| Fig. 4: cross-site mean Tree-SHAP profile (line) ± 1 s.d. across the ten sites (band) against signed distance, 25 panels grouped by morphometric dimension; dot at x₀ where \|ΔS\| ≥ 0.1 | 01 (data), 02 (drawing) |
| SI Table S5: \|ΔS\|, x₀ and w of the 25 indicators | 01 (values), 03 (LaTeX) |
| SI Section S6: x₀ of the leading indicators (mean interbuilding distance +4.7 m, adjacency +4.2 m, perimeter wall +4.9 m, squareness +3.0 m), widths of about 16-18 m, average weighted distance changing sign about 6 m inside | 03 (printed) |
| Results 2.3: the signature shifts gradually across the boundary; the transition is led by the spatial-distribution indicators, which change sign a few metres outside the polygon | 01, 02 |

Related items computed elsewhere: misclassification drivers (SI Fig. S3) in
`appendix/S05_boundary_errors`; classification-probability profile (SI Fig. S6)
and displacement on both sides of the line (SI Section S6.1, SI Fig. S9) in
`appendix/S06_attribution_analyses`; shifted boundaries and sensitivity of x₀
and w (Extended Data Fig. 5, SI Section S13, SI Table S17) in
`appendix/S13_boundary_robustness`, with the same profile functions
(`favela_xai.boundary`).

## Run order

From the repository root, after `main/02_classification`:

```bash
python main/04_boundary_profiles/01_boundary_profiles.py   # signed distance, per-site and cross-site profiles, |ΔS|, x0, w
python main/04_boundary_profiles/02_figure4.py             # Fig. 4
python main/04_boundary_profiles/03_table_S5.py            # SI Table S5 (LaTeX) and the SI Section S6 values
```

## Inputs

- `outputs/main/02_classification/<site>_buildings_shap.gpkg`: `uID`, `label`
  and the 25 `shap_<indicator>` columns (positive = toward favela).
- `outputs/main/01_features/signed_distance.parquet` (`products.signed_distance`):
  signed distance of every building, shared with the misclassification-rate
  analysis of Methods 4.7; computed from the `main/01_features` building tables
  and the official 2013 favela polygons (`data/raw/Limite_Favelas_2013/`, see
  `data/README.md`) if it does not exist yet.

## Outputs (`outputs/main/04_boundary_profiles/`)

| File | Content | Script |
|---|---|---|
| `profiles_persite.csv` | site × indicator × 2.5 m bin: buildings in the bin and mean SHAP (empty below 20 buildings) | 01 |
| `profiles_crosssite.csv` | indicator × bin: cross-site mean and s.d., number of sites | 01 |
| `transition_metrics.csv` | indicator: Fig. 4 panel order, pooled mean \|SHAP\|, ΔS, \|ΔS\|, x₀ (m), w (m), resolved (\|ΔS\| ≥ 0.1) | 01 |
| `figure4.pdf`, `figure4.svg` | Fig. 4 (the SVG keeps the text editable) | 02 |
| `table_S5_transition_metrics.tex` | SI Table S5 | 03 |

## Definitions (`config.yaml`: `signed_distance`, `boundary_profiles`, `main_04_boundary_profiles`)

- Signed distance: Euclidean distance (EPSG:31983) from the footprint centroid to the nearest edge of the official 2013 favela polygons within 300 m of the site's building extent, negative for favela-labelled buildings. `signed_distance.dissolve_polygons`: `false` uses the edges of each polygon, `true` the boundary of their union.
- Profiles: mean SHAP per 2.5 m bin between −40 and +40 m, bins with fewer than 20 buildings of a site left empty; cross-site mean and s.d. (ddof 0) over the sites that have the bin.
- |ΔS|: mean of the cross-site profile over bins centred at ≤ −25 m minus the mean over bins centred at ≥ +25 m.
- x₀: first sign change from favela to formal evidence (inside to outside) of the profile after Gaussian smoothing (`smoothing_sigma_bins`, in bins; 0 disables smoothing), linearly interpolated between bin centres.
- w = √(Σ\|S′\|(x − x̄)² / Σ\|S′\|) on the smoothed profile, x̄ the \|S′\|-weighted mean distance.
- x₀ and w are computed for profiles with at least 6 valid bins and reported where \|ΔS\| ≥ 0.1; Fig. 4 dot opacity = clip(\|ΔS\| / 0.6, 0.42, 1).
- Fig. 4 panel order: dimensions by the mean of their indicators' pooled mean \|SHAP\|, indicators within a dimension by their own pooled mean \|SHAP\|. SI Table S5 rows: decreasing \|ΔS\|.
