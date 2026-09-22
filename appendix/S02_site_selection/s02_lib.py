"""Helpers shared by the numbered scripts of appendix/S02_site_selection (not run on its own).

Used for SI Section S2.1, Supplementary Fig. S1 and Supplementary Table S1.
Contents: names of the output files (outputs/appendix/S02_site_selection/), the
city centre (Praca XV) in the analysis CRS, the analysed buildings of a site
(centroid and label, from main/01_features), and readers for the 2013 favela
polygons and their descriptors. Settings: ``site_context`` and
``appendix_S02_site_selection`` in config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from favela_xai.config import ConfigError
from favela_xai.io import read_site_features
from favela_xai.labels import label_to_int

# -----------------------------------------------------------------------------
# Files written by this folder (outputs/appendix/S02_site_selection/)
# -----------------------------------------------------------------------------
CONTEXT_DIR = "context_layers"                                   # 01
PLANNING_AREAS_GPKG = "context_layers/rio_planning_areas.gpkg"   # 01
MUNICIPALITY_GPKG = "context_layers/rio_municipality.gpkg"       # 01
BAIRROS_GPKG = "context_layers/rio_bairros.gpkg"                 # 01
FOOTPRINTS_PARQUET = "footprints_2013_centroids.parquet"         # 02
POLYGONS_PARQUET = "polygons_base.parquet"                       # 03
MEMBERSHIP_CSV = "study_membership.csv"                          # 03
SITE_HULLS_PARQUET = "site_hulls.parquet"                        # 03
CITY_POLYGONS_CSV = "city_polygons.csv"                          # 04
FIGURE_S1 = "Figure_R1_site_selection.pdf"                       # 05
TABLE_S1_CSV = "site_context_table.csv"                          # 06
TABLE_S1_TEX = "table_R1_site_context.tex"                       # 06


def require(path: Path, step: str) -> Path:
    """Stop with a clear message when an input written by an earlier step is missing."""
    if not Path(path).exists():
        raise ConfigError(f"{path} does not exist; run {step} first.")
    return Path(path)


def city_centre_xy(cfg) -> tuple[float, float]:
    """Praca XV (historic centre; SI Fig. S1) in the analysis CRS, from ``site_context.city_centre_lonlat``."""
    lon, lat = cfg.get("site_context.city_centre_lonlat")
    tr = Transformer.from_crs("EPSG:4326", cfg.get("crs"), always_xy=True)
    return tr.transform(float(lon), float(lat))


def site_buildings(cfg, site: str) -> pd.DataFrame:
    """Analysed buildings of a site: centroid x, y (EPSG:31983) and y_true (1 = favela).

    Rows follow the per-site building table of main/01_features
    (``products.features_gpkg``), the building set of the whole analysis.
    """
    g = read_site_features(cfg, site, columns=["label"])
    cen = shapely.centroid(np.asarray(g.geometry.values, dtype=object))
    return pd.DataFrame({"x": shapely.get_x(cen), "y": shapely.get_y(cen),
                         "y_true": label_to_int(g["label"])})


def read_polygons_2013(cfg) -> gpd.GeoDataFrame:
    """The official 2013 favela polygons with the polygon code in ``cod_favela`` and the name in ``nome``.

    The attribute names of the Data.Rio layer are configurable
    (``appendix_S02_site_selection.polygon_code_field`` / ``polygon_name_field``);
    they are copied to the two canonical column names used by this folder.
    """
    from favela_xai.io import read_favela_polygons

    s = cfg.section("appendix_S02_site_selection")
    pol = read_favela_polygons(cfg, 2013).reset_index(drop=True)
    for field, canon in ((s["polygon_code_field"], "cod_favela"), (s["polygon_name_field"], "nome")):
        if field not in pol.columns:
            raise ConfigError(f"the 2013 favela layer has no attribute '{field}'; attributes: {list(pol.columns)}")
        if field != canon:
            pol[canon] = pol[field]
    pol["nome"] = pol["nome"].astype(str).str.strip()
    return pol


def polygon_descriptors(polygons: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    """Area, perimeter, compactness 4 pi A / P^2, parts, centroid and distance to Praca XV.

    SI Fig. S1 panels b-d (area, compactness, distance to Praca XV).
    """
    pol = polygons.reset_index(drop=True).copy()
    area = pol.geometry.area
    pol["area_ha"] = area / 1e4
    pol["perimeter_m"] = pol.geometry.length
    pol["compactness"] = 4 * np.pi * area / pol.geometry.length ** 2
    pol["n_parts"] = shapely.get_num_geometries(np.asarray(pol.geometry.values, dtype=object))
    cx0, cy0 = city_centre_xy(cfg)
    cen = pol.geometry.centroid
    pol["cx"], pol["cy"] = cen.x, cen.y
    pol["dist_centre_km"] = np.hypot(cen.x - cx0, cen.y - cy0) / 1e3
    return pol


def read_polygons(out_dir: Path, cfg) -> gpd.GeoDataFrame:
    """The 2013 favela polygons with their descriptors (written by 03_polygons_and_membership.py).

    The index is ``poly_index``, the row position in the official 2013 layer.
    """
    path = require(Path(out_dir) / POLYGONS_PARQUET, "03_polygons_and_membership.py")
    base = pd.read_parquet(path)
    geom = shapely.from_wkb(base.pop("wkb").values)
    pol = gpd.GeoDataFrame(base, geometry=geom, crs=cfg.get("crs"))
    return pol.set_index("poly_index", drop=False).rename_axis(None).sort_index()


def site_display_names(cfg) -> dict[str, str]:
    """Display name of every processed site (config ``sites``)."""
    return {s: cfg.site_name(s) for s in cfg.sites}
