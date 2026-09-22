# appendix/S02_site_selection: site selection, analysis windows and site context

Places the ten study sites in the city. The favela polygons covered by the ten
windows are compared with all 1,076 polygons of the official 2013 favela
boundaries, and the context of every site is tabulated. The 107 study polygons
(9.9% of all polygons, 17.5% of the mapped favela area) resemble the other
favelas in area and shape but are denser (median 122.8 versus 96.7 footprints
per hectare) and closer to the centre (median 9.8 versus 20.3 km); the sites lie
3.4-42.9 km from Praça XV in all five planning areas, six on hillsides and four
on flat ground.

## Manuscript items

| Item | Script (output) |
|---|---|
| Supplementary Fig. S1: study favelas compared with all 2013 favelas (a: map; b-e: cumulative distributions of polygon area, compactness 4πA/P², distance to Praça XV and footprints per hectare, with two-sample Kolmogorov-Smirnov D and p) | 01-05 |
| Supplementary Table S1: context of the ten study sites | 06 |
| SI Section S2.1: coverage of the study favelas, footprint density and distance to the centre (study versus other favelas) | 04 (`key_numbers_S2.txt`) |
| SI Section S2.1 and Methods 'Study area and building footprints': site distances, settings, planning areas, windows, favela share | 06 (`site_context_numbers.txt`) |

## Inputs

- Building footprints 2013, all `Tipo` classes (Data.Rio, *Edificações 2013*; `paths.raw.buildings_2013`).
- Official favela boundaries 2013 (Data.Rio, *Limite Favelas 2013*; `paths.raw.favelas_2013`).
- Municipal boundary, planning areas AP1-AP5 and bairros (Data.Rio, CC BY 4.0; `paths.raw.municipality`, `paths.raw.planning_areas`, `paths.raw.bairros`); `01_context_layers.py --download` fetches them.
- Analysed buildings of each site (`main/01_features`, `products.features_gpkg`).
- Per-building slope for the Table S1 slope and setting columns (`appendix/S11_terrain` step 01; needs the 5 m LiDAR terrain model, available from the Instituto Municipal de Urbanismo Pereira Passos on request, `paths.raw.elevation_raster`). Without it these columns are left empty.

## Run order

From the repository root, after `main/01_features` (and step 01 of `appendix/S11_terrain` for the slope columns):

```bash
python appendix/S02_site_selection/01_context_layers.py [--download]   # municipality, planning areas, bairros
python appendix/S02_site_selection/02_footprints_2013.py                # centroid, area, type of every 2013 footprint
python appendix/S02_site_selection/03_polygons_and_membership.py        # polygon descriptors, study favelas
python appendix/S02_site_selection/04_compare_study_vs_city.py          # study vs other favelas, SI S2.1 numbers
python appendix/S02_site_selection/05_figure_S1.py                      # Supplementary Fig. S1
python appendix/S02_site_selection/06_site_context_table.py             # Supplementary Table S1
```

Step 04 needs 02 and 03; 05 needs 01, 03 and 04; 06 needs 01. `s02_lib.py` holds shared helpers.

## Outputs (`outputs/appendix/S02_site_selection/`)

- `Figure_R1_site_selection.pdf`: Supplementary Fig. S1 (05).
- `table_R1_site_context.tex`, `site_context_table.csv`: Supplementary Table S1 (06).
- `key_numbers_S2.txt`, `study_vs_city.csv`: SI S2.1 numbers, distributions and tests (04).
- `site_context_numbers.txt`: site numbers of SI S2.1 and Methods (06).
- Intermediate files: `context_layers/` (01), `footprints_2013_centroids.parquet` (02), `polygons_base.parquet`, `study_membership.csv`, `site_hulls.parquet` (03), `city_polygons.csv` (04).

## Definitions and settings (config.yaml)

- City centre: Praça XV (`site_context.city_centre_lonlat`).
- Study favela: a 2013 polygon containing the centroid of at least one favela-labelled building of a site (`site_context.study_favela_rule`).
- Footprint density: footprints of all types per hectare, centroid inside the polygon (`site_context.density_all_tipo`).
- Area-weighted distance statistic: `appendix_S02_site_selection.area_weighted_statistic` (median | mean); both are written to `study_vs_city.csv`.
- Site location: mean position of the site's favela-labelled buildings; window: extent of its analysed building centroids.
- Setting: hillside if the favela mean slope is at least `terrain.hillside_min_favela_slope_deg`, flat if below `terrain.flat_max_favela_slope_deg`.
- Map generalisation of Fig. S1a: `site_context.context_simplify_m`, `site_context.map_min_part_area_ha`.
