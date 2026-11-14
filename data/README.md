# Data

This repository contains code only. The analysis uses open municipal data of the
City of Rio de Janeiro, produced by the Instituto Municipal de Urbanismo Pereira
Passos (IPP) and published on the Data.Rio portal, plus OpenStreetMap streets and a terrain model
available from IPP on request.
Download the inputs from the sources below and place them at the paths given.
These are the defaults of `paths.raw` in `config.yaml`; edit those entries if the
files are stored elsewhere.

## Raw inputs

| Input | Used for | Source | Place at |
|---|---|---|---|
| Building footprints 2013 (*Edificações 2013*), all `Tipo` classes | all analyses | Data.Rio: <https://www.data.rio/datasets/124f1607b57d4292af5c0dfc2ff8572c_0/about> | `data/raw/Edificacoes_2013/Edificacoes_2013.shp` (with `.shx`, `.dbf`, `.prj`, `.cpg`) |
| Official favela boundaries 2013 (*Limite Favelas 2013*) | favela labels, signed distance to the boundary, all boundary analyses | Data.Rio: <https://www.data.rio/datasets/PCRJ::limite-favelas-2013/about> | `data/raw/Limite_Favelas_2013/Limite_Favelas_2013.shp` |
| Official favela boundaries 2019 (*Limite Favelas 2019*) | 2013/2019 boundary overlay (SI S10) | Data.Rio: <https://www.data.rio/datasets/PCRJ::limite-favelas-2019/about> | `data/raw/Limite_Favelas_2019/Limite_Favelas_2019.shp` |
| Municipal boundary, planning areas (AP1-AP5) and bairros | site context (SI S2, Table S1, Fig. S1); city outline in Fig. 1a | Data.Rio administrative boundaries service, <https://pgeo3.rio.rj.gov.br/arcgis/rest/services/Cartografia/Limites_administrativos/FeatureServer> (layer 0 *Limite do Município*, layer 1 *Limite Áreas de Planejamento (AP)*, layer 4 *Limite de Bairros*). `python appendix/S02_site_selection/01_context_layers.py --download` exports the three layers as GeoJSON. | `data/raw/context/Limite_Municipio.geojson`, `data/raw/context/Limite_Areas_Planejamento.geojson`, `data/raw/context/Limite_Bairros.geojson` |
| OpenStreetMap street network | block-complexity index *k* (SI S9) | OpenStreetMap via `osmnx` (Overpass API), downloaded by `appendix/S09_block_complexity_k/01_download_osm_streets.py`; © OpenStreetMap contributors, ODbL | `data/raw/osm_streets/<site>.gpkg` (one file per site) |
| 5 m LiDAR digital terrain model | terrain ablation (SI S11); slope column of SI Table S1 | Available from IPP on request: <https://www.rio.rj.gov.br/web/ipp> | any location; set `paths.raw.elevation_raster` in `config.yaml` to the GeoTIFF (EPSG:31983) |
| Basemap tiles (optional) | background of the study-area map, Fig. 1a | CARTO Positron (no labels) via `contextily`, downloaded by `main/01_features/07_fig1_basemap.py`; © OpenStreetMap contributors, © CARTO | `data/raw/context/basemap_cache/` |

## Notes

* All layers are used in SIRGAS 2000 / UTM zone 23S (EPSG:31983). Vector
  layers stored in another CRS are reprojected after reading; the 2013
  footprint layer, which is read by bounding box, and the terrain model must
  be supplied in EPSG:31983.
* The ten study sites are rectangular windows of the 2013 footprint layer; their
  coordinates and names are listed under `sites` in `config.yaml`.
* The 2019 favela polygons were accessed in September 2024, the administrative
  boundaries in September 2026 and the OpenStreetMap streets on 8 May 2026.
  Later releases may differ.
* Data.Rio layers are licensed CC BY 4.0 (source: Instituto Municipal de
  Urbanismo Pereira Passos, Prefeitura da Cidade do Rio de Janeiro);
  OpenStreetMap data are licensed ODbL.

## Derived data

All intermediate and final results (per-building indicators, signed distances,
fitted classifiers, Tree-SHAP attributions, out-of-fold predictions and the
tables and figures) are written by the scripts to `outputs/`, which is not part
of the repository. Derived per-building data are available from the
corresponding author upon reasonable request.
