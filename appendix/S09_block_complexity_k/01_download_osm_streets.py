"""Download the OpenStreetMap street network of each site (input of the block-complexity index k).

Manuscript items: Methods 4.9 and SI S9 'Implementation summary' (street
network within the building bounding box padded by 200 m; all highway-tagged
segments except abandoned, construction, planned, proposed, raceway and razed).

Run this script in the download environment (environment-download.yml; osmnx
2.1, Python >= 3.11). It is the only step of the repository that needs network
access. OpenStreetMap changes over time; the paper used extracts downloaded on
8 May 2026. Existing files are kept unless --force is given. Record the date of
a new download in ``block_complexity.osm_access_date`` of config.yaml.

Settings (config.yaml)
* block_complexity: osm_bbox_pad_m (200 m), osm_excluded_highway,
  osm_exclude_area_yes (also leave out ways tagged area=yes);
* appendix_S09_block_complexity_k: osm_network_type, osm_simplify,
  osm_retain_all (osmnx graph options).

Inputs
* products.features_gpkg (main/01_features): the analysed buildings of each
  site; their bounding box defines the download extent.

Outputs
* <paths.raw.osm_streets_dir>/<site>.gpkg   street segments (LineStrings,
  EPSG:31983); (c) OpenStreetMap contributors, ODbL
* <paths.raw.osm_streets_dir>/download_log.csv   site, UTC time, osmnx version,
  Overpass endpoint, bounding box and number of segments of every download

Run (from the repository root, download environment):
    python appendix/S09_block_complexity_k/01_download_osm_streets.py [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt

import geopandas as gpd
import osmnx as ox
import pandas as pd
import shapely

from favela_xai.config import load_config
from favela_xai.io import read_site_features


def highway_filter(cfg) -> str:
    """Overpass filter: every highway-tagged way except the excluded values."""
    bc = cfg.section("block_complexity")
    flt = '["highway"]["highway"!~"' + "|".join(bc["osm_excluded_highway"]) + '"]'
    if bc.get("osm_exclude_area_yes", True):
        flt += '["area"!~"yes"]'
    return flt


def download_site(cfg, site: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Street segments inside the site's building bounding box padded by osm_bbox_pad_m."""
    bc = cfg.section("block_complexity")
    s9 = cfg.section("appendix_S09_block_complexity_k")
    buildings = read_site_features(cfg, site, columns=["label"])
    if s9.get("drop_invalid_geometries", True):
        buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
    pad = float(bc["osm_bbox_pad_m"])
    minx, miny, maxx, maxy = buildings.total_bounds
    box = shapely.box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    left, bottom, right, top = gpd.GeoSeries([box], crs=buildings.crs).to_crs(4326).iloc[0].bounds
    # osmnx 2.x bounding-box order: (left, bottom, right, top)
    graph = ox.graph_from_bbox(bbox=(left, bottom, right, top),
                               network_type=s9["osm_network_type"],
                               custom_filter=highway_filter(cfg),
                               retain_all=bool(s9["osm_retain_all"]),
                               simplify=bool(s9["osm_simplify"]))
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    streets = gpd.GeoDataFrame(geometry=edges.reset_index().geometry.values, crs=edges.crs).to_crs(buildings.crs)
    info = dict(site=site, download_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                osmnx_version=ox.__version__, overpass_url=ox.settings.overpass_url,
                bbox_lon_lat=f"{left:.6f},{bottom:.6f},{right:.6f},{top:.6f}", n_segments=len(streets))
    return streets, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="download again even if a site file exists")
    args = ap.parse_args()

    cfg = load_config()
    folder = cfg.raw_path("osm_streets_dir", required=False)
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "download_log.csv"
    log = []
    for site in cfg.sites:
        path = folder / f"{site}.gpkg"
        if path.exists() and not args.force:
            print(f"{site}: {path.name} exists, kept (use --force to download again)")
            continue
        streets, info = download_site(cfg, site)
        if path.exists():
            path.unlink()
        streets.to_file(path, driver="GPKG", layer=site)
        log.append(info)
        print(f"{site}: {len(streets):,} street segments -> {path}")
    if log:
        new = pd.DataFrame(log)
        if log_path.exists():
            new = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
        new.to_csv(log_path, index=False)
        print(f"Record the download date in block_complexity.osm_access_date of config.yaml "
              f"(this run: {log[-1]['download_utc'][:10]}).")


if __name__ == "__main__":
    main()
