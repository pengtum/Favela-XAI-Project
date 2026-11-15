# appendix/S05_boundary_errors: misclassification at favela boundaries

Where the classifier errs and why (SI Section S5). Errors concentrate at the
official 2013 favela boundary: 26.8% of buildings within 6 m of the boundary
are misclassified against 3.7% at 90 m or more (ratio 7.3), and the near rate
exceeds the far rate in all ten sites. The Tree-SHAP drivers of the false
positives are site-dependent.

The out-of-fold predictions and error-rate tables are computed in
`main/02_classification` (steps 03-05), which also writes the SI S5 error-rate
numbers (`boundary_error_summary.txt`, `blocked_ratios.csv`). This folder lays
out those tables and draws the maps and attribution figures.

## Manuscript items

| Item | Script |
|---|---|
| Extended Data Fig. 1: misclassification rate by signed distance to the 2013 favela boundary (a: rate per 6 m bin with Wilson and site-bootstrap 95% intervals; b: buildings and errors per bin; c: rate by distance band and side, near/far ratio) | 01 |
| Supplementary Table S4: misclassification rate by distance band and side | 02 |
| Supplementary Fig. S2: building-level classification outcomes in the ten sites | 03 |
| Supplementary Fig. S3 and the SI S5 sentence on the false-positive drivers | 04 |
| Supplementary Figs S4 and S5: four building clusters read building by building | 05 |

## Inputs

- `boundary_error_bins.csv`, `boundary_error_bands.csv`: pooled 6 m bins and distance bands by side, from the stratified five-fold out-of-fold predictions (`main/02_classification` steps 03-04; names in `appendix_S05_boundary_errors.inputs_from_main_02`).
- Per-site building table of the main classifier (`products.shap_gpkg`, `main/02_classification` step 01): label, predicted class, test flag, the 25 indicators and their Tree-SHAP values.
- Per-site test accuracies (`products.site_metrics`, `main/02_classification` step 02) for the column order of Fig. S3.
- Official 2013 favela boundaries (Data.Rio; `paths.raw.favelas_2013`).

## Run order

From the repository root, after `main/02_classification` (steps 01-04):

```bash
python appendix/S05_boundary_errors/01_figure_boundary_exposure.py   # Extended Data Fig. 1
python appendix/S05_boundary_errors/02_table_S4.py                    # Supplementary Table S4
python appendix/S05_boundary_errors/03_figure_S2_spatial_mapping.py   # Supplementary Fig. S2 (+ one map per site)
python appendix/S05_boundary_errors/04_failure_drivers.py             # Supplementary Fig. S3, driver numbers
python appendix/S05_boundary_errors/05_case_studies.py                # Supplementary Figs S4 and S5
```

The scripts are independent of each other; `s05_common.py` holds shared helpers.

## Outputs (`outputs/appendix/S05_boundary_errors/`)

- `Figure_R1_boundary_exposure.pdf`: Extended Data Fig. 1 (01).
- `table_R1_exposure_bands.tex`: Supplementary Table S4 (02).
- `Figure_R1_spatial_mapping.pdf`, `spatial_mapping_sites/<site>.pdf`, `spatial_mapping_counts.csv`: Supplementary Fig. S2 (every building, training and held-out test, coloured by its outcome), one map per site, and the numbers of held-out test buildings by outcome (03).
- `Figure_R1_failure_shap.pdf`, `failure_drivers.csv`, `failure_drivers_numbers.txt`: Supplementary Fig. S3 and the driver numbers (04).
- `Figure_R1_casestudies_class.pdf`, `Figure_R1_casestudies_driver.pdf`, `case_study_representatives.csv`: Supplementary Figs S4 and S5 (05).

## Definitions and settings (config.yaml)

- Bins and bands (top-level section `boundary_errors`): 6 m bins on ±90 m with open end bins; bands < 6, 6-30, 30-90 and ≥ 90 m; Wilson 95% interval; site bootstrap with 2,000 resamples.
- False positive: formal building classified favela; false negative: favela building classified formal (main per-site classifier, Methods 4.4).
- `spatial_mapping_figure.buildings` (test_only | all_buildings): buildings coloured by outcome in Fig. S2 (every building, as published | held-out test buildings only, training buildings grey).
- `failure_drivers_figure.population` (all_buildings | test_split): buildings from which the false positives of Fig. S3 are taken.
- `failure_drivers_figure.top_per_site`: features drawn in Fig. S3 are the union of each site's largest mean false-positive SHAP values; the dominant driver is the largest.
- `failure_drivers_figure.site_order` (by_test_accuracy | fixed): column order of Fig. S3.
- `case_studies.windows`, `case_studies.drivers`: centre and half-width of the four clusters (two in Rocinha, two in Grotão de Costa Barros); the four leading features coloured in Fig. S5.

The other keys are in the `appendix_S05_boundary_errors` section.
