# main/01_features: footprints, tessellation and the 25 morphological indicators

Builds the per-building table of every site from the 2013 building footprints
of Rio de Janeiro: footprints are cleaned, cut to the ten site windows,
labelled favela or formal settlement with the official 2013 favela polygons,
tessellated and described by 25 morphological indicators. The folder also
computes the signed distance of every building to the 2013 favela boundary,
writes SI Table S2 and draws the element maps of Fig. 1.

## Manuscript items

| Item | Script |
|---|---|
| Methods 4.1: ten sites; building counts before the tessellation stage (printed by 04 next to the totals stated in Methods 4.1) and after it | 01, 04 |
| Methods 4.2, SI Section S1: footprint pre-processing; footprints discarded at the tessellation stage (printed by 04 next to the figure stated in Methods 4.2) | 01, 02, 04 |
| Methods 4.3, SI Section S3: 25 indicators in five dimensions (dimension 3, shape 13, spatial distribution 4, configuration 3, connectivity 2), computed with momepy or directly from the geometries (areas, perimeters, covered area ratio); average weighted distance from a Delaunay triangulation of the centroids | 03 |
| SI Table S2: indicator names, figure labels and dimensions | 06 |
| Methods 4.7-4.8: signed distance to the official 2013 favela boundary (input of all boundary analyses) | 05 |
| Fig. 1a: study-area map with per-class building counts; Rocinha input map | 04, 07, 08, 09 |
| Fig. 1c: one representative indicator per dimension mapped across Rocinha | 09 |

Fig. 1 is assembled by hand from these element maps; panel b (pipeline
schematic) is artwork. The pre-processing steps that Methods 4.2 and SI Section S1
describe as QGIS tools are applied here in Python
(`favela_xai.features.preprocess_footprints`; *Fix Geometries* with the structure
method is `shapely.make_valid(method="structure")`).

## Run order

From the repository root, in the environment of `environment.yml`:

```bash
python main/01_features/01_preprocess_footprints.py   # clean the footprints, cut the windows, label
python main/01_features/02_tessellation.py            # morphological tessellation and exclusions
python main/01_features/03_indicators.py              # the 25 indicators -> <site>_buildings_features.gpkg
python main/01_features/04_site_summary.py            # building counts and pre-processing report
python main/01_features/05_signed_distance.py         # signed distance to the 2013 boundary
python main/01_features/06_table_S2.py                # SI Table S2 and the indicator registry
python main/01_features/07_fig1_basemap.py            # optional, needs internet: basemap tiles for Fig. 1a
python main/01_features/08_figure1_study_area.py      # Fig. 1a study-area map
python main/01_features/09_figure1_example_maps.py    # Fig. 1a input map and Fig. 1c indicator maps
```

Scripts 01-03 accept `--sites <key> ...`; 04 and 05 need all sites. 06 needs
no input data. Without the tile cache of 07, 08 draws the municipal outline
only. `figure1_common.py` holds helpers of 07-09.

## Inputs

- Building footprints 2013 (*Edificações 2013*, Data.Rio): `paths.raw.buildings_2013`
- Official favela polygons 2013 (*Limite Favelas 2013*, Data.Rio): `paths.raw.favelas_2013`
- Municipal boundary of Rio de Janeiro (Fig. 1a only, Data.Rio): `paths.raw.municipality`
- Site windows and settings: `config.yaml`, sections `sites`, `preprocessing`,
  `tessellation`, `indicators`, `labels`, `signed_distance`, `main_01_features`

Data are not distributed with the code; `data/README.md` lists the sources.

## Outputs (`outputs/main/01_features/`)

| File | Content | Script |
|---|---|---|
| `<site>_footprints_clean.gpkg` | cleaned footprints of the window with `label` (`favela` / `not-favela`) | 01 |
| `<site>_buildings_tessellated.gpkg`, `<site>_tessellation.gpkg`, `<site>_excluded_buildings.gpkg` | kept buildings, their cells (joined by `uID`) and excluded buildings with the reason | 02 |
| `<site>_buildings_features.gpkg` | per-site building table: `uID`, the attributes of `main_01_features.keep_source_columns` that the footprint layer has (`cod_unico` record identifier, `Tipo` cadastre class, `ALTURA` recorded height in m), `label`, the 25 indicators, geometry (EPSG:31983); input of every later folder (row order: `main/02_classification/README.md`, 'Row order'). The per-site tables of the analysis carry `uID`, `label`, `Tipo`, `ALTURA`, the 25 indicators and the geometry, without `cod_unico`, which no later step reads | 03 |
| `preprocessing_report.csv`, `site_counts.csv`, `indicator_missing_values.csv` | counts per cleaning and exclusion step, per-class building counts and positions, missing indicator values | 04 |
| `signed_distance.parquet` | site, `uID`, `is_favela`, `signed_dist` (m, negative on the favela side) | 05 |
| `table_S2_indicators.tex`, `indicator_registry.csv` | SI Table S2 and the full indicator registry | 06 |
| `fig1_map_sites.*`, `fig1_map_input.*`, `fig1_map_feat_<indicator>.*` | Fig. 1 element maps (PDF and SVG) | 08, 09 |

The covered area ratio (`CAR` = `area` / `tes_area`) is not bounded by 1:
where footprints touch or overlap, the boundary between their cells, drawn
from the footprints shrunk by `tessellation.shrink` and discretised at
`tessellation.segment`, can cut into a footprint, and 872 of the 188,292
buildings of the per-site tables of the analysis have `CAR` > 1.

## Labels of the analysed buildings

The favela labels analysed in the paper (81,834 of the 188,292 buildings) were
assigned when the study data were prepared, and the per-building tables of the
analysis carry them; every later folder reads the label from the table. The
overlay rule of this folder (`labels` in `config.yaml`: footprint intersects
the dissolved 2013 polygons) reproduces them for all but 15 buildings (SI
Table S8; `appendix/S10_boundaries_2013_2019`).

## Settings with alternatives

- `indicators.adjacency_contiguity` (tessellation | buildings): queen contiguity
  of the tessellation cells (as analysed) or of the footprints, used by the adjacency
  ratio; `appendix/S15_adjacency_sensitivity` compares the two (SI Section S15).
- `signed_distance.dissolve_polygons` (false | true): distance to the edges of
  the individual 2013 polygons or to the boundary of their dissolved union.
