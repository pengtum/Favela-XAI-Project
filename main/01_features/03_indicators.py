"""Step 3: the 25 morphological indicators of every analysed building.

Manuscript items
  * Methods 4.3, SI Section S3 and SI Table S2: 25 indicators in five
    dimensions (dimension 3, shape 13, spatial distribution 4, configuration 3,
    connectivity 2), computed with momepy or, for areas, perimeters and the
    covered area ratio, directly from the geometries; average weighted distance
    from a Delaunay triangulation of the building centroids (scipy).
  * Methods 4.5: the spatial indicators use a 200 m neighbourhood.
  * The per-site building tables written here (products.features_gpkg) are the
    input of main/02_classification and of every later analysis.

Method (favela_xai.features.compute_indicators)
  momepy functional API with libpysal graphs: queen contiguity of the
  tessellation cells for the sets T_i (items 17-19 and 23), a 200 m distance
  band on building centroids for the neighbourhoods B_i (items 18 and 19);
  covered area ratio = footprint area / cell
  area; average weighted distance = mean length of the Delaunay edges at each
  centroid; equivalent rectangular index, rectangularity, elongation and cell
  alignment (items 11, 12, 15 and 22) evaluated on the minimum-area bounding
  rectangle of SI S3, computed in the same way for every GEOS version
  (favela_xai.features.minimum_area_rectangle).
  indicators.adjacency_contiguity (tessellation | buildings): queen
  contiguity of the tessellation cells (as analysed) or of the footprints, used
  by the adjacency ratio (the two are compared in SI Section S15). The
  indicator registry (names, labels, dimensions, SI S3 formulas) is
  favela_xai.features.FEATURES.

Inputs (outputs/main/01_features/, written by 02_tessellation.py)
  <site>_buildings_tessellated.gpkg, <site>_tessellation.gpkg; config.yaml: indicators.*

Outputs (outputs/main/01_features/)
  <site>_buildings_features.gpkg      products.features_gpkg: uID, cod_unico, Tipo, ALTURA
                                      (recorded height), label, the 25 indicators, geometry
                                      (EPSG:31983)
  steps/<site>_indicator_missing.csv  missing values per indicator (collected by 04_site_summary.py)

Run from the repository root (all sites, or a subset):
  python main/01_features/03_indicators.py [--sites palmeras]
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import pandas as pd

from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER, check_registry, compute_indicators
from favela_xai.io import write_gpkg, write_table


def parse_sites(cfg, description: str) -> list[str]:
    """Sites given with --sites (default: every site in config.yaml)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sites", nargs="+", help="site keys to process (default: all)")
    args = parser.parse_args()
    sites = args.sites or cfg.sites
    unknown = sorted(set(sites) - set(cfg.all_sites))
    if unknown:
        raise SystemExit(f"unknown site key(s): {unknown}; valid keys: {cfg.all_sites}")
    return sites


def read_step2(out_dir, site: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Buildings and cells of step 2, both indexed by uID and in the same order."""
    b_path = out_dir / f"{site}_buildings_tessellated.gpkg"
    c_path = out_dir / f"{site}_tessellation.gpkg"
    for p in (b_path, c_path):
        if not p.exists():
            raise SystemExit(f"{p} not found; run main/01_features/02_tessellation.py first")
    buildings = gpd.read_file(b_path)
    cells = gpd.read_file(c_path)
    buildings.index = pd.Index(buildings["uID"].astype(int).to_numpy())
    cells.index = pd.Index(cells["uID"].astype(int).to_numpy())
    cells = cells.drop(columns="uID")
    if not buildings.index.is_unique or set(buildings.index) != set(cells.index):
        raise SystemExit(f"{site}: buildings and tessellation cells do not share the same uIDs")
    cells = cells.loc[buildings.index]
    return buildings, cells


def main() -> None:
    cfg = load_config()
    sites = parse_sites(cfg, "Compute the 25 morphological indicators of each site.")
    out_dir = cfg.output_dir(__file__)
    keep_cols = list(cfg.get("main_01_features.keep_source_columns", []))

    # Registry check: 25 indicators, 3/13/4/3/2 per dimension (Methods 4.3).
    check_registry(cfg.get("indicators.dimension_sizes"))

    for site in sites:
        buildings, cells = read_step2(out_dir, site)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # momepy / libpysal notes on islands and deprecations
            indicators = compute_indicators(buildings, cells, cfg)

        # One row per building: uID, the attributes kept from the 2013 layer
        # (main_01_features.keep_source_columns), label, the 25 indicators, footprint.
        front = ["uID"] + [c for c in keep_cols if c in buildings.columns] + ["label"]
        table = pd.concat([buildings[front], indicators[MODEL_FEATURE_ORDER]], axis=1)
        gdf = gpd.GeoDataFrame(table, geometry=buildings.geometry, crs=buildings.crs).reset_index(drop=True)
        write_gpkg(gdf, cfg.product("features_gpkg", site=site))

        missing = indicators[MODEL_FEATURE_ORDER].isna().sum()
        write_table(pd.DataFrame({"site": site, "indicator": missing.index, "n_missing": missing.to_numpy()}),
                    out_dir / "steps" / f"{site}_indicator_missing.csv")
        with_missing = ", ".join(f"{k} {v}" for k, v in missing.items() if v) or "none"
        print(f"{site:24s} buildings {len(gdf):6,d}  missing values: {with_missing}")


if __name__ == "__main__":
    main()
