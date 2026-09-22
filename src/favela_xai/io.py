"""Reading and writing of inputs and shared products.

Manuscript items: Main Methods 4.1 'Study area and building footprints'
(2013 footprints, 2013 and 2019 favela polygons, site windows) and the data
availability statement. All locations come from config.yaml.

Raw inputs
* ``read_buildings_2013``: municipal 2013 footprint layer (all Tipo classes),
  optionally only inside a bounding box.
* ``read_favela_polygons``: official favela polygons of 2013 or 2019.
* ``read_site_footprints``: raw footprints inside a site window (padded read).
Shared products (paths from ``products`` in config.yaml)
* ``read_site_features`` / ``read_site_shap``: per-site building tables written
  by main/01_features and main/02_classification.
* ``write_table`` / ``write_gpkg``: writers that create the parent folder.
Every layer is returned in the analysis CRS (``crs`` in config.yaml, EPSG:31983).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# pyogrio supports column selection and bounding-box reads (environment.yml).
gpd.options.io_engine = "pyogrio"


def _to_crs(gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("input layer has no CRS; expected SIRGAS 2000 / UTM 23S (EPSG:31983)")
    return gdf if gdf.crs.equals(crs) else gdf.to_crs(crs)


def read_vector(path: str | Path, cfg, bbox=None, columns=None) -> gpd.GeoDataFrame:
    """Read a vector file and return it in the analysis CRS.

    ``bbox`` (xmin, ymin, xmax, ymax) is given in the analysis CRS; the layers of
    this study are stored in SIRGAS 2000 / UTM 23S, so no reprojection of the box
    is needed.
    """
    kwargs = {}
    if bbox is not None:
        kwargs["bbox"] = tuple(float(v) for v in bbox)
    if columns is not None:
        kwargs["columns"] = list(columns)
    gdf = gpd.read_file(path, **kwargs)
    return _to_crs(gdf, cfg.get("crs"))


def read_buildings_2013(cfg, bbox=None) -> gpd.GeoDataFrame:
    """Municipal 2013 building-footprint layer (Edificacoes 2013, all Tipo classes)."""
    return read_vector(cfg.raw_path("buildings_2013"), cfg, bbox=bbox)


def read_favela_polygons(cfg, year: int = 2013) -> gpd.GeoDataFrame:
    """Official favela polygons (Limite Favelas 2013 or 2019)."""
    if year not in (2013, 2019):
        raise ValueError("year must be 2013 or 2019")
    gdf = read_vector(cfg.raw_path(f"favelas_{year}"), cfg)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)


def padded(window, pad: float):
    xmin, ymin, xmax, ymax = window
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def read_site_footprints(cfg, site: str) -> gpd.GeoDataFrame:
    """Raw 2013 footprints around a site window.

    The layer is read inside the window padded by ``preprocessing.read_pad_m``;
    the window predicate itself is applied after cleaning
    (favela_xai.features.select_window).
    """
    window = cfg.site_window(site)
    return read_buildings_2013(cfg, bbox=padded(window, float(cfg.get("preprocessing.read_pad_m"))))


def read_site_features(cfg, site: str, columns=None) -> gpd.GeoDataFrame:
    """Per-site building table of main/01_features (uID, label, 25 indicators, geometry)."""
    path = cfg.require_product("features_gpkg", site=site)
    gdf = gpd.read_file(path, columns=columns) if columns is not None else gpd.read_file(path)
    return _to_crs(gdf, cfg.get("crs"))


def read_site_shap(cfg, site: str, columns=None) -> gpd.GeoDataFrame:
    """Per-site table of main/02_classification (label, split, prediction, 25 indicators,
    25 ``shap_`` columns with positive values toward favela, geometry)."""
    path = cfg.require_product("shap_gpkg", site=site)
    gdf = gpd.read_file(path, columns=columns) if columns is not None else gpd.read_file(path)
    return _to_crs(gdf, cfg.get("crs"))


def write_gpkg(gdf: gpd.GeoDataFrame, path: str | Path, layer: str | None = None) -> Path:
    """Write a GeoPackage (one layer), creating the folder; overwrites an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG", layer=layer or path.stem)
    return path


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a CSV or Parquet table (by suffix), creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"unsupported table format: {path.suffix}")
    return path


def write_text(text: str, path: str | Path) -> Path:
    """Write a text file (LaTeX table bodies, logs), creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
