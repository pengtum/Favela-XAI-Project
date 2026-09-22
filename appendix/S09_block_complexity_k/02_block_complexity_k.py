"""Per-building block-complexity index k (re-implementation of Bettencourt et al.).

Manuscript items
* Methods 4.9 and SI S9 'Implementation summary': blocks, tessellation per
  block, cell adjacency and multi-source breadth-first search.
* Methods 4.9 and SI S9: share of buildings outside every polygonised block
  (block_coverage.csv).
The per-building k is the input of scripts 03-06 (SI Tables S6-S7, SI S9.2,
SI Fig. S13).

Algorithm, per site (settings in config.yaml ``block_complexity`` and
``appendix_S09_block_complexity_k``)
0. Buildings: the analysed building set of the site (main/01_features);
   invalid or empty geometries are left out.
1. Blocks: the union of the street LineStrings is polygonised with
   shapely.ops.polygonize; polygons are kept if valid, larger than 100 m2
   (min_block_area_m2) and intersecting the study area (union of the
   buildings buffered by study_area_buffer_m = 50 m).
2. Each building's centroid is assigned to the block that contains it (spatial
   join, first match); buildings outside every block get block_id -1 and no k.
3. Within each block the buildings are tessellated with momepy.Tessellation
   clipped to the block polygon (shrink 0.4, segment 0.5). If the tessellation
   of a block fails, each building buffered by 0.5 m serves as its cell
   (tessellation_fallback_buffer_m; the number of such blocks is reported).
4. Two cells are adjacent when they share more than 1 cm of boundary
   (min_shared_boundary_m).
5. Cells that intersect the block boundary buffered by 0.5 m
   (frontier_buffer_m) get k = 1; every other cell gets k = its multi-source
   breadth-first-search depth from that frontier + 1. A block with a single
   building has k = 1. Cells not connected to the frontier get k = -1;
   buildings without a cell have no k. Only valid k (k >= 1) enters the
   comparisons.
Per-block aggregates (number of buildings, median footprint area) are written
as well; they are not used as classifiers (Methods 4.9).

Inputs
* products.features_gpkg (main/01_features): footprints, uID and label.
* <paths.raw.osm_streets_dir>/<site>.gpkg (01_download_osm_streets.py).

Outputs (outputs/appendix/S09_block_complexity_k/)
* block_complexity_k.csv   per building: site, uID, label, block_id,
                           block_n_bldgs, block_median_area, k
* block_coverage.csv       per site: buildings, buildings in a block, share
                           outside every block, blocks, street segments,
                           buildings with a valid k, unreachable cells,
                           buildings without a cell, blocks with fallback cells

Run (from the repository root, analysis environment):
    python appendix/S09_block_complexity_k/02_block_complexity_k.py
"""
from __future__ import annotations

import warnings
from collections import deque

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from shapely.ops import polygonize, unary_union

from favela_xai.config import ConfigError, load_config
from favela_xai.io import read_site_features, write_table

warnings.filterwarnings("ignore")      # momepy legacy-API and tessellation warnings


def read_streets(cfg, site: str, crs) -> gpd.GeoDataFrame:
    """Street segments of one site (written by 01_download_osm_streets.py)."""
    folder = cfg.raw_path("osm_streets_dir", required=False)
    path = folder / f"{site}.gpkg"
    if not path.exists():
        raise ConfigError(f"{path} not found: run 01_download_osm_streets.py (download environment) first")
    streets = gpd.read_file(path)
    return streets if streets.crs.equals(crs) else streets.to_crs(crs)


def polygonise_blocks(streets: gpd.GeoDataFrame, study_area, min_area: float) -> gpd.GeoDataFrame:
    """Faces of the planar street graph kept as blocks (valid, > min_area, intersecting the study area)."""
    faces = list(polygonize(unary_union(streets.geometry.values)))
    blocks = gpd.GeoDataFrame(geometry=faces, crs=streets.crs)
    blocks = blocks[blocks.geometry.is_valid & (blocks.area > min_area)]
    blocks = blocks[blocks.intersects(study_area)].reset_index(drop=True)
    blocks["block_id"] = np.arange(len(blocks))
    return blocks


def assign_blocks(buildings: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> np.ndarray:
    """Block of each building's centroid (first match), -1 outside every block."""
    points = gpd.GeoDataFrame({"_row": np.arange(len(buildings))}, geometry=buildings.geometry.centroid.values,
                              crs=buildings.crs)
    joined = gpd.sjoin(points, blocks[["block_id", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset=["_row"], keep="first").set_index("_row")
    return joined["block_id"].reindex(np.arange(len(buildings))).fillna(-1).astype(int).to_numpy()


def block_k(block_buildings: gpd.GeoDataFrame, block_polygon, p: dict) -> tuple[pd.Series, bool]:
    """k of the buildings of one block (indexed by uID) and whether the fallback cells were used."""
    n = len(block_buildings)
    if n == 1:
        # a single building is adjacent to the block boundary
        return pd.Series([1], index=block_buildings["uID"].to_numpy(), dtype=int), False

    tmp = block_buildings[["uID", "geometry"]].reset_index(drop=True).copy()
    tmp["uID"] = tmp["uID"].astype(int)
    fallback = False
    try:
        cells = momepy.Tessellation(tmp, unique_id="uID", limit=block_polygon,
                                    shrink=p["shrink"], segment=p["segment"], verbose=False).tessellation
    except Exception:
        cells = tmp.copy()
        cells["geometry"] = cells.geometry.buffer(p["fallback_buffer"])
        fallback = True
    cells = cells.reset_index(drop=True)

    # adjacency: cells sharing more than min_shared_boundary_m of boundary
    graph = nx.Graph()
    ids = cells["uID"].astype(int).to_numpy()
    graph.add_nodes_from(ids.tolist())
    geoms = cells.geometry.values
    sindex = cells.sindex
    for i, geom in enumerate(geoms):
        for j in sindex.intersection(geom.bounds):
            if j <= i:
                continue
            if geom.intersects(geoms[j]) and geom.intersection(geoms[j]).length > p["min_shared"]:
                graph.add_edge(int(ids[i]), int(ids[j]))

    # frontier: cells touching the block boundary (buffered); multi-source BFS
    frontier_zone = block_polygon.boundary.buffer(p["frontier_buffer"])
    frontier = ids[cells.geometry.intersects(frontier_zone).to_numpy()]
    k = {int(c): 1 for c in frontier}
    queue = deque(k)
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v not in k:
                k[v] = k[u] + 1
                queue.append(v)
    for c in ids:                       # not connected to the frontier
        k.setdefault(int(c), -1)
    return pd.Series(k, name="k", dtype=int), fallback


def process_site(cfg, site: str) -> tuple[pd.DataFrame, dict]:
    bc = cfg.section("block_complexity")
    s9 = cfg.section("appendix_S09_block_complexity_k")
    p = dict(shrink=float(s9["tessellation_shrink"]), segment=float(s9["tessellation_segment"]),
             fallback_buffer=float(bc["tessellation_fallback_buffer_m"]),
             min_shared=float(bc["min_shared_boundary_m"]), frontier_buffer=float(bc["frontier_buffer_m"]))

    b = read_site_features(cfg, site, columns=["uID", "label"])
    n_all = len(b)
    if s9.get("drop_invalid_geometries", True):
        b = b[b.geometry.is_valid & ~b.geometry.is_empty]
    b = b.reset_index(drop=True)
    streets = read_streets(cfg, site, b.crs)
    study_area = unary_union(b.buffer(float(bc["study_area_buffer_m"])).geometry.values)
    blocks = polygonise_blocks(streets, study_area, float(bc["min_block_area_m2"]))
    b["block_id"] = assign_blocks(b, blocks)
    print(f"{site}: {len(b):,} buildings, {len(streets):,} street segments, {len(blocks):,} blocks, "
          f"{(b.block_id >= 0).mean():.1%} of buildings in a block")

    polygon_of = dict(zip(blocks["block_id"], blocks.geometry))
    parts, n_fallback = [], 0
    for bid, group in b[b.block_id >= 0].groupby("block_id"):
        ks, fb = block_k(group, polygon_of[bid], p)
        n_fallback += int(fb)
        parts.append(ks)
    k = pd.concat(parts) if parts else pd.Series(dtype=float, name="k")

    b["k"] = k.reindex(b["uID"].to_numpy()).to_numpy()
    b.loc[b.block_id < 0, "k"] = np.nan
    in_block = b.block_id >= 0
    b["block_n_bldgs"] = b.groupby("block_id")["uID"].transform("size").where(in_block)
    b["block_median_area"] = b.geometry.area.groupby(b["block_id"]).transform("median").where(in_block)
    b["site"] = site

    coverage = dict(site=site, n_buildings=len(b), n_invalid_geometry_dropped=n_all - len(b),
                    n_in_block=int(in_block.sum()), share_outside_blocks=float(1 - in_block.mean()),
                    n_blocks=len(blocks), n_street_segments=len(streets), n_valid_k=int((b.k >= 1).sum()),
                    n_unreachable=int((b.k == -1).sum()), n_without_cell=int((in_block & b.k.isna()).sum()),
                    n_blocks_fallback_cells=n_fallback)
    cols = ["site", "uID", "label", "block_id", "block_n_bldgs", "block_median_area", "k"]
    return pd.DataFrame(b[cols]), coverage


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    tables, coverage = [], []
    for site in cfg.sites:
        tab, cov = process_site(cfg, site)
        tables.append(tab)
        coverage.append(cov)
    k = pd.concat(tables, ignore_index=True)
    cov = pd.DataFrame(coverage)
    write_table(k, out / "block_complexity_k.csv")
    write_table(cov, out / "block_coverage.csv")
    s = 100 * cov["share_outside_blocks"]
    print(f"buildings outside every polygonised block: mean {s.mean():.0f}% per site "
          f"(range {s.min():.0f}-{s.max():.0f}%)")
    valid = k.k >= 1
    print(f"buildings with a valid k: {valid.sum():,}; k median {k.k[valid].median():.0f}, "
          f"p90 {k.k[valid].quantile(.9):.0f}, max {k.k[valid].max():.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
