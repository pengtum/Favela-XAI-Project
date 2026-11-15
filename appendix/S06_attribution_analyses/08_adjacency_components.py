"""SI Section S3 (item 19) and Section S6: the adjacency ratio recomputed, and the
components of its numerator.

SI Section S3 (item 19) defines the adjacency ratio of building i as
ADJ_i = K_i / N_i: N_i is the number of buildings whose centroids lie within
200 m of the centroid of building i (the set B_i, building i included) and K_i
the number of connected components of the site's cell-contiguity graph (queen
contiguity of the tessellation cells) represented among them. SI Sections S3
and S6 read K_i as one plus the number of components formed by roof-level
structures that the cadastre records as polygons of their own inside the
outline of a larger building, mainly machine rooms (casa de máquina) and
upper-level volumes (projeção), whose cells are contiguous with none of the
surrounding cells. This script
  (i)  recomputes the ratio of every analysed building from the footprints and
       the tessellation cells with the configured graph and neighbourhood, and
       counts the buildings whose recomputed value equals the analysed value;
  (ii) for the buildings with K_i > 1, tabulates the cadastre class (Tipo) of
       the components beyond the main one.

Method
* Cells: the tessellation cells written by main/01_features/02_tessellation.py
  (outputs/main/01_features/<site>_tessellation.gpkg), matched to the analysed
  buildings by uID. Where that file is absent, or with --regenerate_cells, the
  cells are regenerated from the analysed footprints with
  favela_xai.features.tessellate (tessellation.* in config.yaml); column
  'cells' of the site table records which cells were used.
* Graph: queen contiguity of the cells, built as in
  favela_xai.features.compute_indicators (libpysal Graph.build_contiguity;
  indicators.tessellation_contiguity, indicators.adjacency_contiguity =
  tessellation), and its connected components (Graph.component_labels).
* Neighbourhood B_i: the buildings whose centroids lie within
  indicators.neighbourhood_band_m (200 m) of the centroid of building i,
  building i included (the distance band of compute_indicators, to which
  momepy.building_adjacency adds the building itself), found with a k-d tree
  (scipy cKDTree, distance <= 200 m). Recomputed ratio = K_i / N_i in float64.
  A match is exact floating-point equality with the analysed value
  (products.features_gpkg); the largest absolute difference is also given.
  With --momepy the ratio is also computed exactly as compute_indicators does
  (momepy.building_adjacency on the libpysal distance band) and compared in the
  same way; that band graph has about 20 million edges in Catumbi and takes
  about 4 GB of memory to build.
* Main and further components: the cells tile the buffered limit of the
  tessellation (momepy.buffered_limit, tessellation.limit_buffer_m). In each
  connected part of the limit, the component whose cells cover the largest
  area is that part's main component; every other component is a further
  component. A building belongs to the part that contains its footprint.
  K_i = M_i + F_i, with M_i main and F_i further components represented among
  the buildings of B_i.
* Cadastre class: attribute Tipo of the 2013 building layer
  (main_01_features.keep_source_columns), in three groups: machine room (class
  name containing CASA_DE_MAQUINA), upper-level volume (PROJECAO) and other.
  The class of a further component is the set of groups of its buildings. A
  building of a further component lies inside a larger footprint when at least
  99% of its footprint area lies within one other analysed footprint of larger
  area.

Inputs
* products.features_gpkg (main/01_features): uID, Tipo, the analysed adjacency
  ratio and the footprint of every analysed building.
* outputs/main/01_features/<site>_tessellation.gpkg (main/01_features/
  02_tessellation.py): uID and cell of every analysed building (see Method).

Outputs (outputs/appendix/S06_attribution_analyses/)
* adjacency_reproduction_by_site.csv: per site, and over all sites (final row
  'all_sites'): buildings, cells used, buildings whose recomputed ratio equals
  the analysed value, largest absolute difference (with --momepy the same for
  the momepy computation); components, main components, further components and
  the buildings in them; buildings with K_i > 1 and, among them, those with one
  main component in B_i (K_i - 1 = F_i).
* adjacency_further_components_tipo.csv: per site and over all sites, per Tipo
  class: group, buildings in further components, and how many of them lie
  inside a larger footprint.
* adjacency_further_components_class.csv: per site and over all sites, per
  component class: further components, further components represented in B_i
  of a building with K_i > 1, and the pairs (building i with K_i > 1, further
  component represented in B_i).
* adjacency_components_buildings.parquet: per building (site, uID): N_i, K_i,
  M_i, F_i, analysed and recomputed ratio, component, main component (bool).
* adjacency_components_summary.txt: the summary printed at the end.

Run from the repository root (after main/01_features):
    python appendix/S06_attribution_analyses/08_adjacency_components.py [--sites ...]
        [--regenerate_cells] [--momepy]
"""
from __future__ import annotations

import argparse
import time
import warnings
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.spatial import cKDTree

from favela_xai.config import ConfigError, load_config
from favela_xai.features import tessellate
from favela_xai.io import read_site_features, write_table, write_text

INDICATOR = "adjacency"     # the adjacency ratio (SI S3 item 19), as analysed
CLASS_FIELD = "Tipo"        # cadastre class of the 2013 building layer
# Groups of cadastre classes: (group, text contained in the class name). A
# building belongs to the first group whose text its class name contains,
# otherwise to OTHER.
GROUPS = [("machine room", "CASA_DE_MAQUINA"), ("upper-level volume", "PROJECAO")]
OTHER = "other"
GROUP_ORDER = [g for g, _ in GROUPS] + [OTHER]
INSIDE_SHARE = 0.99         # share of the footprint area within a larger footprint
ALL = "all_sites"
SITE_TABLE = "adjacency_reproduction_by_site.csv"
TIPO_TABLE = "adjacency_further_components_tipo.csv"
CLASS_TABLE = "adjacency_further_components_class.csv"
BUILDING_TABLE = "adjacency_components_buildings.parquet"
SUMMARY = "adjacency_components_summary.txt"


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def parse_args(cfg) -> tuple[list[str], argparse.Namespace]:
    """Sites (default: every site in config.yaml, in config order) and options."""
    parser = argparse.ArgumentParser(
        description="Adjacency ratio recomputed from footprints and cells; classes of its further components.")
    parser.add_argument("--sites", nargs="+", help="site keys to process (default: all)")
    parser.add_argument("--regenerate_cells", action="store_true",
                        help="regenerate the cells from the analysed footprints even if "
                             "main/01_features/<site>_tessellation.gpkg exists")
    parser.add_argument("--momepy", action="store_true",
                        help="also compute the ratio with momepy.building_adjacency on the libpysal distance band")
    args = parser.parse_args()
    sites = args.sites or cfg.sites
    unknown = sorted(set(sites) - set(cfg.all_sites))
    if unknown:
        raise SystemExit(f"unknown site key(s): {unknown}; valid keys: {cfg.all_sites}")
    order = {s: i for i, s in enumerate(cfg.all_sites)}
    return sorted(dict.fromkeys(sites), key=order.get), args


def read_buildings(cfg, site: str) -> gpd.GeoDataFrame:
    """uID (index), Tipo, analysed adjacency ratio and footprint of the analysed buildings, in uID order."""
    path = cfg.require_product("features_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    missing = [c for c in ("uID", CLASS_FIELD, INDICATOR) if c not in fields]
    if missing:
        raise ConfigError(f"{path} lacks the columns {missing}; keep {CLASS_FIELD} with "
                          "main_01_features.keep_source_columns and re-run main/01_features.")
    gdf = read_site_features(cfg, site, columns=["uID", CLASS_FIELD, INDICATOR])
    gdf.index = pd.Index(gdf["uID"].astype(int).to_numpy())
    if not gdf.index.is_unique:
        raise SystemExit(f"{site}: uID is not unique in {path}")
    return gdf.sort_index()


def site_cells(cfg, site: str, buildings: gpd.GeoDataFrame, regenerate: bool) -> tuple[gpd.GeoDataFrame, str]:
    """Tessellation cell of every analysed building (index uID, order of ``buildings``) and their source."""
    path = cfg.output_dir("main/01_features", create=False) / f"{site}_tessellation.gpkg"
    if path.exists() and not regenerate:
        cells = gpd.read_file(path)
        cells.index = pd.Index(cells["uID"].astype(int).to_numpy())
        cells = cells[[cells.geometry.name]]
        source = "main/01_features/02_tessellation.py"
    else:
        # favela_xai.features.tessellate numbers the buildings it keeps 0..n-1; the
        # analysed uID travels with each footprint and is restored on the cells.
        src = gpd.GeoDataFrame({"analysed_uID": buildings.index.to_numpy()},
                               geometry=buildings.geometry.to_numpy(), crs=buildings.crs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)   # momepy/geopandas CRS and deprecation notes
            kept, cells, _, _ = tessellate(src, cfg)
        cells = gpd.GeoDataFrame(geometry=cells.geometry.to_numpy(), crs=cells.crs,
                                 index=pd.Index(kept.loc[cells.index, "analysed_uID"].to_numpy()))
        source = "regenerated from products.features_gpkg (favela_xai.features.tessellate)"
    if set(cells.index) != set(buildings.index) or not cells.index.is_unique:
        raise SystemExit(f"{site}: the cells ({source}) do not match the analysed buildings one to one "
                         f"({len(buildings.index.difference(cells.index))} buildings without a cell, "
                         f"{len(cells.index.difference(buildings.index))} cells without an analysed building)")
    return cells.loc[buildings.index], source


# -----------------------------------------------------------------------------
# Graph, components and neighbourhoods
# -----------------------------------------------------------------------------
def contiguity_graph(cfg, cells: gpd.GeoDataFrame):
    """Contiguity graph of the cells, as favela_xai.features.compute_indicators builds it."""
    from libpysal import graph

    ind = cfg.section("indicators")
    if ind.get("adjacency_contiguity", "tessellation") != "tessellation":
        raise SystemExit("indicators.adjacency_contiguity is not 'tessellation'; this script checks the ratio on "
                         "the contiguity of the tessellation cells (SI Section S15 covers footprint contiguity).")
    rook = ind.get("tessellation_contiguity", "queen") == "rook"
    return graph.Graph.build_contiguity(cells[[cells.geometry.name]], rook=rook)


def limit_parts(cfg, buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Connected part of the buffered tessellation limit that contains each footprint."""
    import momepy

    limit = momepy.buffered_limit(buildings, buffer=float(cfg.section("tessellation")["limit_buffer_m"]))
    parts = list(getattr(limit, "geoms", [limit]))
    i, j = shapely.STRtree(parts).query(buildings.geometry.representative_point().to_numpy(), predicate="within")
    part = np.full(len(buildings), -1)
    part[i] = j
    if (part < 0).any():
        raise SystemExit(f"{int((part < 0).sum())} footprints lie outside the buffered limit")
    return part


def main_components(component: np.ndarray, part: np.ndarray, cell_area: np.ndarray) -> np.ndarray:
    """In each part of the limit, the component whose cells cover the largest area."""
    covered = pd.DataFrame({"part": part, "component": component, "area": cell_area}).groupby(
        ["part", "component"])["area"].sum()
    return np.unique(covered.loc[covered.groupby(level="part").idxmax()].index.get_level_values("component"))


def neighbourhood_components(xy: np.ndarray, component: np.ndarray, is_main: np.ndarray, radius: float):
    """N_i, M_i and F_i of every building, and the pairs (building, further component in B_i)."""
    n = len(xy)
    N = cKDTree(xy).query_ball_point(xy, r=radius, return_length=True)
    M = np.zeros(n, dtype=np.int64)
    for c in np.unique(component[is_main]):
        members = np.flatnonzero(component == c)
        M += cKDTree(xy[members]).query_ball_point(xy, r=radius, return_length=True) > 0
    further = np.flatnonzero(~is_main)
    if len(further) == 0:
        return N, M, np.zeros(n, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    lists = cKDTree(xy[further]).query_ball_point(xy, r=radius)
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=n)
    focal = np.repeat(np.arange(n), lengths)
    member = further[np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(lengths.sum()))]
    pairs = np.unique(np.column_stack([focal, component[member]]), axis=0)
    F = np.bincount(pairs[:, 0], minlength=n).astype(np.int64)
    return N, M, F, pairs


def momepy_ratio(buildings: gpd.GeoDataFrame, contiguity, radius: float) -> np.ndarray:
    """The ratio as compute_indicators computes it: momepy.building_adjacency on the libpysal distance band."""
    import momepy
    from libpysal import graph

    band = graph.Graph.build_distance_band(buildings.geometry.centroid, radius)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = momepy.building_adjacency(contiguity_graph=contiguity, neighborhood_graph=band)
    return ratio.reindex(buildings.index).to_numpy(float)


# -----------------------------------------------------------------------------
# Cadastre classes
# -----------------------------------------------------------------------------
def tipo_groups(tipo: pd.Series) -> np.ndarray:
    """Group of every building from its cadastre class."""
    names = tipo.fillna("").astype(str)
    group = np.full(len(names), OTHER, dtype=object)
    free = np.ones(len(names), dtype=bool)
    for key, text in GROUPS:
        hit = free & names.str.contains(text, regex=False).to_numpy()
        group[hit] = key
        free &= ~hit
    return group


def component_class(groups: pd.Series) -> str:
    """Class of a further component: the groups of its buildings."""
    present = set(groups)
    return " + ".join(g for g in GROUP_ORDER if g in present)


def sort_tipo(table: pd.DataFrame) -> pd.DataFrame:
    """Rows by group (GROUP_ORDER), then by number of buildings."""
    rank = table["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    return (table.assign(_rank=rank).sort_values(["_rank", "buildings", "Tipo"], ascending=[True, False, True])
            .drop(columns="_rank").reset_index(drop=True))


def sort_classes(table: pd.DataFrame) -> pd.DataFrame:
    """Rows by number of pairs with buildings with K_i > 1."""
    return table.sort_values(["pairs_K_gt_1", "component_class"], ascending=[False, True]).reset_index(drop=True)


def inside_larger_footprint(geoms: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Whether footprint idx[k] lies (INSIDE_SHARE of its area) within one other, larger footprint."""
    out = np.zeros(len(idx), dtype=bool)
    if len(idx) == 0:
        return out
    area = shapely.area(geoms)
    a, b = shapely.STRtree(geoms).query(geoms[idx], predicate="intersects")
    other = idx[a] != b
    a, b = a[other], b[other]
    shared = shapely.area(shapely.intersection(geoms[idx[a]], geoms[b]))
    hit = (shared >= INSIDE_SHARE * area[idx[a]]) & (area[b] > area[idx[a]])
    out[a[hit]] = True
    return out


# -----------------------------------------------------------------------------
# One site
# -----------------------------------------------------------------------------
def analyse_site(cfg, site: str, regenerate: bool, with_momepy: bool):
    t0 = time.time()
    buildings = read_buildings(cfg, site)
    cells, source = site_cells(cfg, site, buildings, regenerate)
    contiguity = contiguity_graph(cfg, cells)
    component = contiguity.component_labels.reindex(buildings.index).to_numpy()
    part = limit_parts(cfg, buildings)
    is_main = np.isin(component, main_components(component, part, cells.geometry.area.to_numpy()))

    radius = float(cfg.get("indicators.neighbourhood_band_m"))
    cen = buildings.geometry.centroid
    xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])
    N, M, F, pairs = neighbourhood_components(xy, component, is_main, radius)
    K = M + F
    recomputed = K.astype(np.float64) / N.astype(np.float64)
    analysed = buildings[INDICATOR].to_numpy(float)

    n = len(buildings)
    tipo = buildings[CLASS_FIELD].fillna("").astype(str).to_numpy()
    group = tipo_groups(buildings[CLASS_FIELD])
    further = np.flatnonzero(~is_main)
    sizes = pd.Series(component[further]).value_counts()
    row = {"site": site, "site_name": cfg.site_name(site), "buildings": n, "cells": source,
           "exact_matches": int(np.sum(recomputed == analysed)),
           "max_abs_difference": float(np.max(np.abs(recomputed - analysed)))}
    if with_momepy:
        direct = momepy_ratio(buildings, contiguity, radius)
        row["momepy_exact_matches"] = int(np.sum(direct == analysed))
        row["momepy_max_abs_difference"] = float(np.max(np.abs(direct - analysed)))
    k_gt_1 = K > 1
    row.update({
        "components": int(len(np.unique(component))),
        "main_components": int(len(np.unique(component[is_main]))),
        "further_components": int(len(sizes)),
        "buildings_in_further_components": int(len(further)),
        "largest_further_component": int(sizes.max()) if len(sizes) else 0,
        "buildings_K_gt_1": int(k_gt_1.sum()),
        "buildings_K_gt_1_one_main": int((k_gt_1 & (M == 1)).sum()),
        "buildings_without_one_main": int((M != 1).sum()),
    })

    # Cadastre classes of the buildings in further components
    inside = inside_larger_footprint(buildings.geometry.to_numpy(), further)
    fb = pd.DataFrame({"component": component[further], "Tipo": tipo[further], "group": group[further],
                       "inside": inside})
    tipo_rows = sort_tipo(fb.groupby(["Tipo", "group"], sort=False)
                          .agg(buildings=("inside", "size"), inside_larger_footprint=("inside", "sum")).reset_index())
    tipo_rows.insert(0, "site", site)

    # Classes of the further components, and their pairs with buildings with K_i > 1
    classes = fb.groupby("component")["group"].agg(component_class)
    pr = pd.DataFrame(pairs, columns=["building", "component"])
    pr = pr[k_gt_1[pr["building"].to_numpy()]]
    pr["class"] = classes.reindex(pr["component"].to_numpy()).to_numpy()
    class_rows = pd.DataFrame({
        "further_components": classes.value_counts(),
        "in_neighbourhoods_K_gt_1": pr.drop_duplicates("component")["class"].value_counts(),
        "pairs_K_gt_1": pr["class"].value_counts()}).fillna(0).astype(int)
    class_rows = sort_classes(class_rows.rename_axis("component_class").reset_index())
    class_rows.insert(0, "site", site)

    per_building = pd.DataFrame({"site": site, "uID": buildings.index.to_numpy(), "N": N, "K": K, "M": M, "F": F,
                                 "adjacency_analysed": analysed, "adjacency_recomputed": recomputed,
                                 "component": component, "main_component": is_main})
    row["seconds"] = round(time.time() - t0, 1)
    print(f"{site:24s} buildings {n:6,d}  exact {row['exact_matches']:6,d}  further components "
          f"{row['further_components']:4d} ({row['buildings_in_further_components']:5,d} buildings)  "
          f"K>1 {row['buildings_K_gt_1']:6,d}  [{row['seconds']:.0f} s; cells: {source.split(' ')[0]}]", flush=True)
    return row, tipo_rows, class_rows, per_building


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def pooled_site_row(site_table: pd.DataFrame) -> dict:
    row = {"site": ALL, "site_name": "All sites (pooled)", "cells": "; ".join(dict.fromkeys(site_table["cells"]))}
    for c in site_table.columns:
        if c in row or c == "seconds":
            continue
        row[c] = site_table[c].max() if c.startswith(("max_", "momepy_max_", "largest_")) else site_table[c].sum()
    row["seconds"] = site_table["seconds"].sum()
    return row


def pooled_rows(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = table.drop(columns="site").groupby(keys, sort=False).sum().reset_index()
    out.insert(0, "site", ALL)
    return out


def summary_lines(site_row: dict, tipo: pd.DataFrame, classes: pd.DataFrame, n_sites: int, with_momepy: bool):
    pooled_tipo = tipo[tipo["site"] == ALL]
    by_group = pooled_tipo.groupby("group")["buildings"].sum().reindex(GROUP_ORDER, fill_value=0)
    pooled_cls = classes[classes["site"] == ALL].sort_values("pairs_K_gt_1", ascending=False)
    n_further = int(site_row["buildings_in_further_components"])
    lines = [
        f"Adjacency ratio (SI Section S3, item 19) recomputed for {site_row['buildings']:,} analysed buildings "
        f"in {n_sites} site(s); cells: {site_row['cells']}",
        f"  recomputed ratio equal to the analysed value: {site_row['exact_matches']:,} of {site_row['buildings']:,} "
        f"buildings (largest absolute difference {site_row['max_abs_difference']:.3g})",
    ]
    if with_momepy:
        lines.append(f"  momepy.building_adjacency on the libpysal distance band: {site_row['momepy_exact_matches']:,} "
                     f"of {site_row['buildings']:,} (largest absolute difference "
                     f"{site_row['momepy_max_abs_difference']:.3g})")
    lines += [
        f"Components of the cell-contiguity graph: {site_row['components']:,}, of which "
        f"{site_row['main_components']} main components (one per part of the buffered limit) and "
        f"{site_row['further_components']:,} further components with {n_further:,} buildings "
        f"(at most {site_row['largest_further_component']} buildings in one)",
        f"  buildings of further components by cadastre class: "
        + ", ".join(f"{g} {int(by_group[g]):,}" for g in GROUP_ORDER),
        f"  (SI Section S3, item 19: mainly machine rooms, casa de máquina, and upper-level volumes, projeção)",
        f"  inside a larger footprint (>= {100 * INSIDE_SHARE:g}% of the area): "
        f"{int(pooled_tipo['inside_larger_footprint'].sum()):,} of {n_further:,}",
        "  other classes: " + (", ".join(f"{r.Tipo} {int(r.buildings)}" for r in
                                          pooled_tipo[pooled_tipo['group'] == OTHER].itertuples()) or "none"),
        f"Buildings with K_i > 1: {site_row['buildings_K_gt_1']:,} of {site_row['buildings']:,}; with exactly one main "
        f"component within the neighbourhood (K_i - 1 = number of further components): "
        f"{site_row['buildings_K_gt_1_one_main']:,}",
        "  further components in their neighbourhoods, by class (components / building-component pairs):",
        *[f"    {r.component_class:40s} {int(r.in_neighbourhoods_K_gt_1):6,d} / {int(r.pairs_K_gt_1):8,d}"
          for r in pooled_cls.itertuples()],
    ]
    return lines


def main() -> None:
    cfg = load_config()
    sites, args = parse_args(cfg)
    out = cfg.output_dir(__file__)

    rows, tipo_parts, class_parts, buildings = [], [], [], []
    for site in sites:
        row, tipo_rows, class_rows, per_building = analyse_site(cfg, site, args.regenerate_cells, args.momepy)
        rows.append(row)
        tipo_parts.append(tipo_rows)
        class_parts.append(class_rows)
        buildings.append(per_building)

    site_table = pd.DataFrame(rows)
    site_table = pd.concat([site_table, pd.DataFrame([pooled_site_row(site_table)])], ignore_index=True)
    tipo = pd.concat(tipo_parts, ignore_index=True)
    tipo = pd.concat([tipo, sort_tipo(pooled_rows(tipo, ["Tipo", "group"]))], ignore_index=True)
    classes = pd.concat(class_parts, ignore_index=True)
    classes = pd.concat([classes, sort_classes(pooled_rows(classes, ["component_class"]))], ignore_index=True)

    write_table(site_table, out / SITE_TABLE)
    write_table(tipo, out / TIPO_TABLE)
    write_table(classes, out / CLASS_TABLE)
    write_table(pd.concat(buildings, ignore_index=True), out / BUILDING_TABLE)

    lines = summary_lines(site_table.iloc[-1].to_dict(), tipo, classes, len(sites), args.momepy)
    write_text("\n".join(lines) + "\n", out / SUMMARY)
    print("\n".join(lines))
    for name in (SITE_TABLE, TIPO_TABLE, CLASS_TABLE, BUILDING_TABLE, SUMMARY):
        print(f"[saved] {out / name}")


if __name__ == "__main__":
    main()
