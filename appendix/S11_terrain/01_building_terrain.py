"""Step 1 of the terrain ablation: slope raster and per-building terrain features.

Manuscript items
* SI Section S11 'Data' and Methods 4.10: slope (degrees) from finite-difference
  gradients of the 5 m digital terrain model (DTM) and nine zonal statistics
  per building footprint (rasterstats.zonal_stats), appended to the per-site
  building set of the main analysis.
* These features are the input of SI Table S9 and SI Figs S14-S18 (scripts
  02-06 of this folder).

Inputs
* ``paths.raw.elevation_raster`` in config.yaml: the 5 m LiDAR digital terrain
  model (GeoTIFF, EPSG:31983), available from the Instituto Municipal de
  Urbanismo Pereira Passos (IPP) on request. The script stops with an
  explanatory message while the entry is empty.
* ``products.features_gpkg`` (main/01_features): the per-site building tables
  of the main analysis (uID, label, 25 indicators, geometry).

Outputs (outputs/appendix/S11_terrain/)
* ``slope_5m.tif``: slope in degrees on the DTM grid (nodata -9999).
* ``<site>_buildings_with_terrain.gpkg``: the building table of the site with
  the nine terrain columns appended (elev_mean, elev_min, elev_max,
  elev_range, elev_std, slope_mean, slope_max, slope_median, slope_std).

Settings: ``terrain`` (SI S11 'Data') and ``appendix_S11_terrain`` in config.yaml.

Run from the repository root:
    python appendix/S11_terrain/01_building_terrain.py            # reuse an existing slope raster
    python appendix/S11_terrain/01_building_terrain.py --overwrite  # recompute it
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterstats import zonal_stats

from favela_xai.config import ConfigError, load_config
from favela_xai.io import read_site_features, write_gpkg

import terrain_common as tc


def check_raster(src, cfg) -> None:
    """The DTM must be in the analysis CRS (SI S11 'Data': EPSG:31983) at 5 m."""
    if src.crs is None or not CRS.from_user_input(src.crs).equals(CRS.from_user_input(cfg.get("crs"))):
        raise ConfigError(f"the elevation raster must be in {cfg.get('crs')} (found {src.crs}); "
                          "reproject it before running this script")
    res = float(cfg.get("terrain.resolution_m"))
    cell_x, cell_y = abs(src.transform[0]), abs(src.transform[4])
    if not (np.isclose(cell_x, res) and np.isclose(cell_y, res)):
        print(f"  note: DTM cell size {cell_x} x {cell_y} m differs from terrain.resolution_m = {res} m")


def compute_slope(dtm_path, slope_path, nodata_out: float, cfg) -> None:
    """Slope in degrees from finite-difference gradients of the DTM (SI S11 'Data').

    numpy.gradient uses central differences in the interior and one-sided
    differences at the raster edges; cells next to a nodata cell become nodata.
    """
    with rasterio.open(dtm_path) as src:
        check_raster(src, cfg)
        dem = src.read(1).astype(np.float64)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        cell_x = abs(src.transform[0])
        cell_y = abs(src.transform[4])
        print(f"  DTM {dem.shape[1]} x {dem.shape[0]} cells of {cell_x} x {cell_y} m")
        meta = src.meta.copy()

    dz_dy, dz_dx = np.gradient(dem, cell_y, cell_x)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    out = slope.astype(np.float32)
    out[np.isnan(out)] = nodata_out
    meta.update(dtype="float32", nodata=nodata_out, count=1)
    with rasterio.open(slope_path, "w", **meta) as dst:
        dst.write(out, 1)
    print(f"  slope raster written: {slope_path}")


def building_terrain(geoms, dtm_path, slope_path, dtm_nodata: float, slope_nodata: float,
                     all_touched: bool) -> pd.DataFrame:
    """Nine zonal statistics per building footprint (SI S11 'Data').

    Elevation: mean, min, max, std and range (= max - min) of the DTM cells;
    slope: mean, max, median and std of the slope cells. ``all_touched``
    (terrain.zonal_all_touched) counts every cell touched by the footprint.
    Buildings outside the raster get missing values.
    """
    elev = zonal_stats(geoms, str(dtm_path), stats=["mean", "min", "max", "std"],
                       nodata=dtm_nodata, all_touched=all_touched)
    slope = zonal_stats(geoms, str(slope_path), stats=["mean", "max", "std", "median"],
                        nodata=slope_nodata, all_touched=all_touched)
    df = pd.DataFrame({
        "elev_mean": [s["mean"] for s in elev],
        "elev_min": [s["min"] for s in elev],
        "elev_max": [s["max"] for s in elev],
        "elev_std": [s["std"] for s in elev],
        "slope_mean": [s["mean"] for s in slope],
        "slope_max": [s["max"] for s in slope],
        "slope_median": [s["median"] for s in slope],
        "slope_std": [s["std"] for s in slope],
    }).astype(float)
    df["elev_range"] = df["elev_max"] - df["elev_min"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true", help="recompute the slope raster if it exists")
    args = parser.parse_args()

    cfg = load_config()
    out = cfg.output_dir(__file__)
    if cfg.get("terrain.building_set") != "main_analysis":
        raise ConfigError("terrain.building_set must be main_analysis (SI S11 'Data')")
    if cfg.get("terrain.slope_method") != "finite_difference":
        raise ConfigError("terrain.slope_method: only finite_difference is implemented (SI S11 'Data')")
    dtm_path = cfg.elevation_raster()                  # user-supplied; explanatory error if not set
    slope_path = out / tc.SLOPE_RASTER
    slope_nodata = float(cfg.get("appendix_S11_terrain.slope_nodata"))
    dtm_nodata_fallback = float(cfg.get("appendix_S11_terrain.dtm_nodata_fallback"))
    all_touched = bool(cfg.get("terrain.zonal_all_touched"))
    features = tc.terrain_features(cfg)

    print("[1/2] Slope raster from the 5 m DTM")
    if slope_path.exists() and not args.overwrite:
        print(f"  using existing {slope_path} (pass --overwrite to recompute)")
    else:
        compute_slope(dtm_path, slope_path, slope_nodata, cfg)

    with rasterio.open(dtm_path) as src:
        dtm_nodata = src.nodata if src.nodata is not None else dtm_nodata_fallback

    print("[2/2] Zonal statistics per building (main-analysis building set)")
    for site in cfg.sites:
        gdf = read_site_features(cfg, site)            # products.features_gpkg, EPSG:31983
        terr = building_terrain(list(gdf.geometry), dtm_path, slope_path, dtm_nodata, slope_nodata, all_touched)
        for col in features:
            gdf[col] = terr[col].to_numpy()
        n_missing = int(gdf["elev_mean"].isna().sum())
        path = write_gpkg(gdf, out / tc.BUILDINGS_GPKG.format(site=site))
        print(f"  {site}: {len(gdf):,} buildings"
              + (f", {n_missing:,} without DTM coverage" if n_missing else "") + f" -> {path.name}")
        # Summary of the per-building terrain values (printed only).
        print(f"      median per building: elevation range {gdf['elev_range'].median():.2f} m, "
              f"mean slope {gdf['slope_mean'].median():.1f} deg")


if __name__ == "__main__":
    main()
