# appendix/S13_boundary_robustness: robustness of the boundary analyses

Three checks of the boundary profiles of main-text Figure 4 (Supplementary
Section S13): whether the transition moves with the official 2013 boundary
(twenty randomly shifted boundaries, label-step and within-class controls),
whether it is visible in the raw indicators as well as in the attributions, and
whether it depends on buildings that sit on the line or on the exact position
of the polygons. Buildings, labels, indicators and the attributions of the main
per-site model are held fixed; nothing is refitted.

## Manuscript items

* Extended Data Fig. 5, shifted boundaries and label controls: 05 (data), 07 (figure).
* Supplementary Fig. S22, raw-indicator and SHAP boundary profiles: 03 (data), 04 (figure).
* Supplementary Table S17, sensitivity of the sign-change distance x0: 01, 02, 06 (data), 08 (table).
* Values in SI Section S13: printed at the end of 01, 03, 05 and 08.

## Main results

* The contrast travels with the real boundary: plateau contrast |ΔS| of 2.39
  (mean interbuilding distance) and 1.71 (adjacency) on the real line, against
  at most 1.29 and 0.31 on the twenty shifted boundaries; for mean
  interbuilding distance, adjacency and average weighted distance the
  within-class gradient of the favela-labelled buildings (within 10 m of the
  line against 25-40 m inside it) on the real line exceeds that of all twenty
  shifted lines, in the attributions and in the raw indicators (05).
* The transition is present in the raw indicators: e.g. squareness +0.91 IQR
  units (95% site-bootstrap interval +0.50 to +1.51), average weighted distance
  −0.47 (−0.75 to −0.20); the SHAP contrast is positive in all nine evaluable
  sites for the five leading indicators (03).
* Buildings on the line (4,110 straddling, a further 6,742 near-line): with
  these buildings excluded or the polygons shifted by 2.5 m, x0 stays within
  +0.8 to +6.6 m (mean interbuilding distance, adjacency, perimeter wall),
  +0.4 to +1.2 m (squareness) and −6.5 to −5.4 m (average weighted distance);
  polygon shifts of 2.5 to 10 m change |ΔS| by at most 0.11 (02, 06, 08).

## Run order

From the repository root, after `main/02_classification` (and
`main/04_boundary_profiles`, used only for consistency checks in 05 and 08):

```bash
python appendix/S13_boundary_robustness/01_prepare_inputs.py               # per-site inputs; straddling / near-line buildings
python appendix/S13_boundary_robustness/02_straddler_exclusions.py         # x0, w, dS without straddling / near-line buildings
python appendix/S13_boundary_robustness/03_raw_profiles.py                 # raw vs SHAP profiles, site bootstrap
python appendix/S13_boundary_robustness/04_figure_S22_raw_vs_shap.py       # Supplementary Fig. S22
python appendix/S13_boundary_robustness/05_shifted_boundaries.py           # real line vs 20 shifted boundaries, label step, G
python appendix/S13_boundary_robustness/06_polygon_shifts.py               # rigid and per-polygon shifts of 2.5, 5, 10 m
python appendix/S13_boundary_robustness/07_figure_ED5_shifted_boundaries.py  # Extended Data Fig. 5
python appendix/S13_boundary_robustness/08_table_S17_x0_sensitivity.py     # Supplementary Table S17
```

`robustness_common.py` holds the shared helpers (profile conventions, raw
standardisation, within-class gradient, distances to moved polygons, figure
scaling to 183 mm with PyMuPDF); it is not run on its own.

## Inputs and outputs

* Inputs: per-site building tables with label, the 25 indicators and the Tree
  SHAP values of the main model (`products.shap_gpkg`, from
  `main/02_classification`); the official 2013 favela polygons
  (`paths.raw.favelas_2013`, see `data/README.md`).
* Outputs (`outputs/appendix/S13_boundary_robustness/`):
  `Figure_R1_placebo_boundaries.pdf` (Extended Data Fig. 5),
  `Figure_R1_raw_vs_shap.pdf` (Supplementary Fig. S22),
  `table_R1_x0_sensitivity.tex` (Supplementary Table S17), and the CSV/NPZ
  tables listed in each script's docstring.
* Settings: `config.yaml` sections `boundary_profiles`, `signed_distance`,
  `boundary_robustness` and `appendix_S13_boundary_robustness`.
