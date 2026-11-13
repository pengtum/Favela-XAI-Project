"""Step 2: morphological tessellation with one cell per building.

Manuscript items
  * Methods 4.2 and SI Section S1: geometries failing at the tessellation stage
    (tessellation failure, too small, invalid) are discarded, 443 of the 188,735
    footprints (0.23%) (counts and shares per site written by 04_site_summary.py).
  * The cells are the basis of the tessellation indicators of Methods 4.3 and
    SI Section S3 (cell area, compactness-weighted axis, number of neighbours,
    covered area ratio, cell alignment) and of the contiguity graph of the
    spatial indicators.

Method (favela_xai.features.tessellate)
  Buildings that are null, invalid, empty, not a single Polygon or smaller than
  tessellation.min_area_m2 are excluded; momepy.morphological_tessellation runs
  within momepy.buffered_limit (tessellation.limit_buffer_m); buildings without
  a cell are excluded. Kept buildings and cells share the id uID.

Inputs
  outputs/main/01_features/<site>_footprints_clean.gpkg   (01_preprocess_footprints.py)
  config.yaml: tessellation.*

Outputs (outputs/main/01_features/)
  <site>_buildings_tessellated.gpkg   kept buildings: uID, source columns, label, geometry
  <site>_tessellation.gpkg            tessellation cells: uID, geometry
  <site>_excluded_buildings.gpkg      excluded buildings with exclusion_reason (only if any)
  steps/<site>_tessellation.csv       counts per exclusion reason

Run from the repository root:
  python main/01_features/02_tessellation.py [--sites palmeras rocinha]
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import pandas as pd

from favela_xai.config import load_config
from favela_xai.features import tessellate
from favela_xai.io import write_gpkg, write_table
from favela_xai.labels import FAVELA

REASONS = ["null_geometry", "invalid_geometry", "empty_geometry", "not_polygon", "too_small",
           "tessellation_failure"]


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


def main() -> None:
    cfg = load_config()
    sites = parse_sites(cfg, "Morphological tessellation of each site.")
    out_dir = cfg.output_dir(__file__)

    for site in sites:
        footprints = gpd.read_file(cfg.require_product("footprints_clean", site=site))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)   # momepy/geopandas CRS and deprecation notes
            kept, cells, excluded, report = tessellate(footprints, cfg)

        # Buildings and cells are written with their common uID column.
        write_gpkg(kept.reset_index(drop=True), out_dir / f"{site}_buildings_tessellated.gpkg")
        cells_out = cells.copy()
        cells_out.insert(0, "uID", cells.index.to_numpy())
        write_gpkg(cells_out.reset_index(drop=True), out_dir / f"{site}_tessellation.gpkg")
        excluded_path = out_dir / f"{site}_excluded_buildings.gpkg"
        if len(excluded):
            write_gpkg(gpd.GeoDataFrame(excluded.reset_index(drop=True), geometry=footprints.geometry.name,
                                        crs=footprints.crs), excluded_path)
        elif excluded_path.exists():
            excluded_path.unlink()

        row = {"site": site, "tessellation_input": report["tessellation_input"],
               "tessellation_kept": report["tessellation_kept"]}
        for reason in REASONS:
            row[f"excluded_{reason}"] = report.get(f"excluded_{reason}", 0)
        row["excluded_total"] = int(len(excluded))
        row["favela_excluded"] = int((excluded["label"] == FAVELA).sum()) if len(excluded) else 0
        row["favela_kept"] = int((kept["label"] == FAVELA).sum())
        write_table(pd.DataFrame([row]), out_dir / "steps" / f"{site}_tessellation.csv")
        share = 100 * row["excluded_total"] / max(row["tessellation_input"], 1)
        print(f"{site:24s} input {row['tessellation_input']:6,d}  kept {row['tessellation_kept']:6,d}  "
              f"excluded {row['excluded_total']:4d} ({share:.2f}%)")


if __name__ == "__main__":
    main()
