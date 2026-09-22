"""Footprint pre-processing, morphological tessellation and the 25 indicators.

Manuscript items: Main Methods 4.2 'Building-footprint pre-processing' and
4.3 'Morphological feature calculation'; SI Section S1, Section S3 and
Table S2; Main Fig. 1 (five dimensions: 3/13/4/3/2 indicators).

Contents
* ``FEATURES``: the single registry of the 25 indicators (column name, full
  name and figure label as in SI Table S2, morphometric dimension, how the
  indicator is computed, the SI S3 formula and, where they differ, how the
  momepy definition relates to it). ``MODEL_FEATURE_ORDER`` is the column order
  used by every classifier.
* ``preprocess_footprints``: the pre-processing steps of Methods 4.2.
* ``tessellate``: morphological tessellation with one cell per building and
  exclusion of unsuitable geometries.
* ``compute_indicators``: the 25 indicators with momepy (Methods 4.3); the
  covered area ratio is each building's own area divided by the area of its
  own cell.
* ``table_s2_body``: the body of SI Table S2 generated from the registry.

The indicator code needs the momepy functional API (momepy >= 0.8) with
libpysal graphs; see environment.yml.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

# -----------------------------------------------------------------------------
# Registry of the 25 indicators
# -----------------------------------------------------------------------------
DIMENSIONS = ["Dimension", "Shape", "Spatial distribution", "Configuration", "Connectivity"]


@dataclass(frozen=True)
class Indicator:
    key: str          # column name in all tables
    full_name: str    # SI Table S2, column 1
    label: str        # SI Table S2, column 2 (figure label, plain text)
    label_tex: str    # the same label as typeset in SI Table S2
    dimension: str    # SI Table S2, column 3
    computed_on: str  # "building", "tessellation cell" or "building and cell"
    method: str       # how the value is computed in this repository
    si_formula: str   # formula printed in SI Section S3
    note: str = ""    # optional remark


# Order of SI Section S3 / SI Table S2.
FEATURES: list[Indicator] = [
    # --- Dimension (3) ---
    Indicator("area", "Building area", "area", "area", "Dimension", "building",
              "footprint area (geopandas .area)", "A = area(P)"),
    Indicator("perimeter", "Building perimeter", "perimeter", "perimeter", "Dimension", "building",
              "footprint perimeter (geopandas .length)", "P = length(dP)"),
    Indicator("lal", "Longest axis length", "longest axis", "longest axis", "Dimension", "building",
              "momepy.longest_axis_length(buildings)", "L_max = max(L_major, L_minor)"),
    # --- Shape (13) ---
    Indicator("shape_idx", "Shape index", "shape index", "shape index", "Shape", "building",
              "momepy.shape_index(buildings)", "SI = P / sqrt(A)"),
    Indicator("compact_weighted_axis", "Compactness-weighted axis", "compact. axis", r"compact.\ axis", "Shape",
              "building", "momepy.compactness_weighted_axis(buildings)",
              "CWA = (L_major/L_minor) * 4 pi A / P^2"),
    Indicator("convexity", "Convexity", "convexity", "convexity", "Shape", "building",
              "momepy.convexity(buildings)", "C = A / A_convex"),
    Indicator("facade_ratio", "Facade ratio", "facade ratio", "facade ratio", "Shape", "building",
              "momepy.facade_ratio(buildings)", "FR = L_facade / P"),
    Indicator("shared_walls", "Shared walls", "shared walls", "shared walls", "Shape", "building",
              "momepy.shared_walls(buildings)", "SW = sum of shared wall lengths"),
    Indicator("perimeter_wall", "Perimeter wall", "perimeter wall", "perimeter wall", "Shape", "building",
              "momepy.perimeter_wall(buildings)",
              "PW = length(dU), U = union of P with every footprint contiguous with it (PW = P when free-standing)"),
    Indicator("corners", "Number of corners", "corners", "corners", "Shape", "building",
              "momepy.corners(buildings, include_interiors=True)", "NC = #vertices(P)"),
    Indicator("rect_idx", "Equivalent rectangular index", "rect. index", r"rect.\ index", "Shape", "building",
              "momepy.equivalent_rectangular_index(buildings)", "ERI = A / A_rect"),
    Indicator("rectangularity", "Rectangularity", "rectangularity", "rectangularity", "Shape", "building",
              "momepy.rectangularity(buildings)", "R = A / (L_major * L_minor)"),
    Indicator("squareness", "Squareness", "squareness", "squareness", "Shape", "building",
              "momepy.squareness(buildings)", "SQ = 1 - mean(|theta_i - 90 deg|) / 90 deg"),
    Indicator("square_compact", "Square compactness", "square compact", "square compact", "Shape", "building",
              "momepy.square_compactness(buildings)", "SC = A / max(L_major, L_minor)^2"),
    Indicator("elongation", "Elongation", "elongation", "elongation", "Shape", "building",
              "momepy.elongation(buildings)", "E = L_major / L_minor"),
    Indicator("cwa", "Compactness-weighted axis of tessellation", "compact. axis (t)", r"compact.\ axis (t)",
              "Shape", "tessellation cell", "momepy.compactness_weighted_axis(cells)",
              "CWT = L_axis / sqrt(A_tess)"),
    # --- Spatial distribution (4) ---
    Indicator("neighbour_dist", "Mean distance between buildings", "neighbour dist.", "neighbour dist.",
              "Spatial distribution", "building and cell",
              "momepy.neighbor_distance(buildings, queen contiguity graph of the tessellation)",
              "d_mean = (1/N^2) sum_i sum_j d(i,j)"),
    Indicator("mean_interbuilding_distance", "Mean interbuilding distance (nearest neighbour)",
              "interbuild. dist.", r"interbuild.\ dist.", "Spatial distribution", "building and cell",
              "momepy.mean_interbuilding_distance(buildings, adjacency_graph=tessellation queen graph, "
              "neighborhood_graph=200 m distance band on centroids)",
              "d_NN = (1/N) sum_i min_j d(i,j)"),
    Indicator("adjacency", "Building adjacency", "adjacency", "adjacency", "Spatial distribution",
              "building and cell",
              "momepy.building_adjacency(contiguity_graph=queen graph of the tessellation cells "
              "(indicators.adjacency_contiguity), neighborhood_graph=200 m distance band on centroids)",
              "ADJ(i) = #{j : P_i intersects P_j}"),
    Indicator("CAR", "Covered area ratio", "cov-area ratio", "cov-area ratio", "Spatial distribution",
              "building and cell", "building area / area of its own tessellation cell",
              "CAR = sum_i A_i / A_site"),
    # --- Configuration (3) ---
    Indicator("tes_area", "Tessellation cell area", "tess. cell area", r"tess.\ cell area", "Configuration",
              "tessellation cell", "cell area (geopandas .area)", "A_v = area(V_i)"),
    Indicator("cell_alignment", "Cell alignment", "cell align.", "cell align.", "Configuration",
              "building and cell",
              "momepy.cell_alignment(momepy.orientation(buildings), momepy.orientation(cells))",
              "CA = (1/N) sum_i cos(theta_i - mean theta)"),
    Indicator("num_neighbors", "Tessellation number of neighbours", "no. neighbours", r"no.\ neighbours",
              "Configuration", "tessellation cell", "momepy.neighbors(cells, queen contiguity graph)",
              "TN(i) = #{j : V_i intersects V_j}"),
    # --- Connectivity (2) ---
    Indicator("fractal", "Fractal dimension", "fractal", "fractal", "Connectivity", "building",
              "momepy.fractal_dimension(buildings)", "FD = 2 ln(P) / ln(A)"),
    Indicator("avg_weighted_dist", "Average weighted distance of building centroids", "avg wt. dist.",
              r"avg wt.\ dist.", "Connectivity", "building",
              "mean length of the Delaunay-triangulation edges incident to the building centroid",
              "d_w = (1/N) sum_i sum_j w_ij d(i,j)"),
]

FEATURE_KEYS: list[str] = [f.key for f in FEATURES]            # SI Table S2 order
BY_KEY: dict[str, Indicator] = {f.key: f for f in FEATURES}

# Column order of every classifier. Tree ensembles can depend on column order
# through tie-breaking, so this order is used for fitting everywhere.
MODEL_FEATURE_ORDER: list[str] = [
    "area", "perimeter", "lal", "neighbour_dist", "mean_interbuilding_distance", "adjacency", "corners",
    "shape_idx", "facade_ratio", "compact_weighted_axis", "squareness", "square_compact", "rectangularity",
    "rect_idx", "perimeter_wall", "num_neighbors", "elongation", "fractal", "cwa", "CAR", "tes_area",
    "shared_walls", "convexity", "cell_alignment", "avg_weighted_dist",
]
SHAP_COLUMNS: list[str] = ["shap_" + k for k in MODEL_FEATURE_ORDER]


def label_of(key: str, tex: bool = False) -> str:
    """Figure label of an indicator (SI Table S2)."""
    return BY_KEY[key].label_tex if tex else BY_KEY[key].label


def dimension_of(key: str) -> str:
    return BY_KEY[key].dimension


def features_by_dimension() -> dict[str, list[str]]:
    """{dimension: [indicator keys]} in SI Table S2 order."""
    return {d: [f.key for f in FEATURES if f.dimension == d] for d in DIMENSIONS}


def check_registry(expected_sizes: dict[str, int] | None = None) -> None:
    """Assert that the registry matches the manuscript (25 indicators, 3/13/4/3/2)."""
    assert len(FEATURES) == 25 and len(set(FEATURE_KEYS)) == 25
    assert sorted(MODEL_FEATURE_ORDER) == sorted(FEATURE_KEYS)
    sizes = {d: len(v) for d, v in features_by_dimension().items()}
    expected = expected_sizes or {"Dimension": 3, "Shape": 13, "Spatial distribution": 4,
                                  "Configuration": 3, "Connectivity": 2}
    assert sizes == expected, f"dimension sizes {sizes} differ from {expected}"


def table_s2_body() -> str:
    """Rows of SI Table S2 (indicator, figure label, dimension), LaTeX."""
    rows = [f"{f.full_name} & {f.label_tex} & {f.dimension} \\\\" for f in FEATURES]
    return "\n".join(rows) + "\n"


# -----------------------------------------------------------------------------
# Pre-processing (Main Methods 4.2; SI S1)
# -----------------------------------------------------------------------------
def _polygonal_part(geom):
    """Keep only the polygonal part of a geometry returned by make_valid."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not parts:
        return None
    return shapely.union_all(parts)


def preprocess_footprints(gdf: gpd.GeoDataFrame, cfg) -> tuple[gpd.GeoDataFrame, dict]:
    """Clean raw footprints with the steps of Main Methods 4.2 / SI S1.

    Steps, in the order of the manuscript:
      1. validity check; invalid geometries repaired with make_valid, structure
         method (QGIS 'Fix Geometries', structure method); null, empty and
         non-polygonal results removed;
      2. duplicate geometries removed (exact equality after normalisation);
      3. multipart features converted to singlepart;
      4. Douglas-Peucker simplification with a tolerance of 0.1 (metres);
      5. second validity check (invalid or empty results removed).

    Returns the cleaned footprints (column ``src_row`` = row of the input) and
    a dict of counts for the pre-processing report.
    """
    p = cfg.section("preprocessing")
    report: dict[str, int] = {"input": int(len(gdf))}
    out = gdf.copy()
    out["src_row"] = np.arange(len(out))
    out = out.reset_index(drop=True)
    gname = out.geometry.name

    geoms = np.asarray(out.geometry.values, dtype=object)
    null = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    report["removed_null_or_empty"] = int(null.sum())
    out = out.loc[~null].reset_index(drop=True)

    # 1. validity check and repair
    geoms = np.asarray(out.geometry.values, dtype=object)
    invalid = ~shapely.is_valid(geoms)
    report["repaired_invalid"] = int(invalid.sum())
    if invalid.any():
        repaired = shapely.make_valid(geoms[invalid], method=p["make_valid_method"], keep_collapsed=False)
        geoms = geoms.copy()
        geoms[invalid] = [_polygonal_part(g) for g in repaired]
        out[gname] = gpd.GeoSeries(geoms, index=out.index, crs=out.crs)
    bad = out.geometry.isna() | out.geometry.is_empty
    report["removed_unrepairable"] = int(bad.sum())
    out = out.loc[~bad].reset_index(drop=True)

    # 2. duplicate geometries
    if p.get("remove_duplicate_geometries", True):
        key = shapely.to_wkb(shapely.normalize(np.asarray(out.geometry.values, dtype=object)), hex=True)
        dup = pd.Series(key).duplicated(keep="first").to_numpy()
        report["removed_duplicates"] = int(dup.sum())
        out = out.loc[~dup].reset_index(drop=True)

    # 3. multipart to singlepart
    if p.get("multipart_to_singlepart", True):
        n_before = len(out)
        out = out.explode(index_parts=False).reset_index(drop=True)
        report["added_by_singlepart"] = int(len(out) - n_before)

    # 4. Douglas-Peucker simplification
    tol = float(p["simplify_tolerance_m"])
    gname = out.geometry.name
    out[gname] = out.geometry.simplify(tol, preserve_topology=bool(p["simplify_preserve_topology"]))

    # 5. second validity check
    if p.get("final_validity_check", True):
        g = out.geometry
        bad = g.isna() | g.is_empty | ~g.is_valid | ~g.geom_type.isin(["Polygon"])
        report["removed_final_check"] = int(bad.sum())
        out = out.loc[~bad].reset_index(drop=True)
    report["output"] = int(len(out))
    return out, report


def select_window(gdf: gpd.GeoDataFrame, window: tuple[float, float, float, float], predicate: str = "within"
                  ) -> gpd.GeoDataFrame:
    """Footprints of a site window (EPSG:31983 bbox), by 'within' or 'intersects'."""
    box = shapely.box(*window)
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    if predicate == "within":
        keep = shapely.within(geoms, box)
    elif predicate == "intersects":
        keep = shapely.intersects(geoms, box)
    else:
        raise ValueError(f"unknown window predicate '{predicate}'")
    return gdf.loc[keep].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Tessellation
# -----------------------------------------------------------------------------
def tessellate(buildings: gpd.GeoDataFrame, cfg):
    """Morphological tessellation with a 1:1 building-cell match.

    Buildings that are null, invalid, empty, not a single Polygon or smaller than
    ``tessellation.min_area_m2`` are excluded before tessellating; buildings that
    receive no cell are excluded afterwards ("tessellation failure").

    Returns ``(kept, cells, excluded, report)``: ``kept`` and ``cells`` are
    indexed by the same unique id (also stored as the column ``uID`` of
    ``kept``); ``excluded`` lists the excluded buildings with a reason.
    """
    import momepy

    t = cfg.section("tessellation")
    b = buildings.reset_index(drop=True).copy()
    geoms = b.geometry
    reason = pd.Series("", index=b.index, dtype=object)
    reason[geoms.isna()] = "null_geometry"
    ok = reason == ""
    reason[ok & ~geoms.is_valid] = "invalid_geometry"
    ok = reason == ""
    reason[ok & geoms.is_empty] = "empty_geometry"
    ok = reason == ""
    reason[ok & (geoms.geom_type != "Polygon")] = "not_polygon"
    ok = reason == ""
    reason[ok & (geoms.area < float(t["min_area_m2"]))] = "too_small"
    ok = reason == ""

    excluded = b.loc[~ok].copy()
    excluded["exclusion_reason"] = reason[~ok]
    # uID = position after the first exclusion; the index holds the
    # same values (left unnamed so that it never clashes with the uID column).
    kept = b.loc[ok].reset_index(drop=True)
    kept["uID"] = kept.index.to_numpy()

    limit = momepy.buffered_limit(kept, buffer=float(t["limit_buffer_m"]))
    cells = momepy.morphological_tessellation(kept, clip=limit, shrink=float(t["shrink"]),
                                              segment=float(t["segment"]))
    cells = gpd.GeoDataFrame(geometry=cells.geometry if hasattr(cells, "geometry") else cells, crs=kept.crs)
    cells.index = cells.index.astype(int)

    missing = kept.index.difference(cells.index)
    if len(missing):
        failed = kept.loc[missing].copy()
        failed["exclusion_reason"] = "tessellation_failure"
        excluded = pd.concat([excluded, failed.drop(columns="uID")], ignore_index=True)
        kept = kept.drop(index=missing)
    cells = cells.loc[kept.index]          # same uIDs, same order as the buildings
    assert cells.index.equals(kept.index)

    report = {"tessellation_input": int(len(b)), "tessellation_kept": int(len(kept))}
    for r, n in excluded["exclusion_reason"].value_counts().items():
        report[f"excluded_{r}"] = int(n)
    return kept, cells, excluded, report


# -----------------------------------------------------------------------------
# The 25 indicators
# -----------------------------------------------------------------------------
def average_weighted_distance(centroids: gpd.GeoSeries) -> pd.Series:
    """Mean length of the Delaunay edges incident to each centroid.

    scipy Delaunay triangulation, one edge per connected pair, mean over the
    neighbours of a node. Points that belong to no triangle (e.g. exact
    duplicates) get NaN.
    """
    from scipy.spatial import Delaunay

    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    n = len(xy)
    if n < 3:
        return pd.Series(np.nan, index=centroids.index)
    tri = Delaunay(xy)
    s = tri.simplices
    pairs = np.vstack([s[:, [0, 1]], s[:, [0, 2]], s[:, [1, 2]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    length = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    total = np.bincount(pairs[:, 0], weights=length, minlength=n) + np.bincount(pairs[:, 1], weights=length, minlength=n)
    count = np.bincount(pairs[:, 0], minlength=n) + np.bincount(pairs[:, 1], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return pd.Series(mean, index=centroids.index)


def compute_indicators(kept: gpd.GeoDataFrame, cells: gpd.GeoDataFrame, cfg) -> pd.DataFrame:
    """The 25 indicators of every building (index uID, columns MODEL_FEATURE_ORDER).

    ``kept`` and ``cells`` are the outputs of :func:`tessellate` (same uID index).
    The neighbourhood of the spatial indicators is a distance band of
    ``indicators.neighbourhood_band_m`` (200 m, Main Methods 4.5) on building
    centroids and the adjacency graph is the queen contiguity of the tessellation
    cells. ``indicators.adjacency_contiguity`` sets the contiguity graph of the
    building-adjacency indicator:
    * ``tessellation``: queen contiguity of the tessellation cells;
    * ``buildings``: queen contiguity of the footprints.
    """
    import momepy
    from libpysal import graph

    ind = cfg.section("indicators")
    assert kept.index.equals(cells.index), "buildings and cells must share the uID index"
    b = kept[[kept.geometry.name]].copy()
    c = cells[[cells.geometry.name]].copy()
    out = pd.DataFrame(index=kept.index)

    # Dimension and building shape
    out["area"] = b.geometry.area
    out["perimeter"] = b.geometry.length
    out["facade_ratio"] = momepy.facade_ratio(b)
    out["lal"] = momepy.longest_axis_length(b)
    out["shape_idx"] = momepy.shape_index(b)
    out["compact_weighted_axis"] = momepy.compactness_weighted_axis(b)
    out["convexity"] = momepy.convexity(b)

    # Spatial distribution
    rook = ind.get("tessellation_contiguity", "queen") == "rook"
    contiguity = graph.Graph.build_contiguity(c, rook=rook)
    band = graph.Graph.build_distance_band(b.geometry.centroid, float(ind["neighbourhood_band_m"]))
    out["neighbour_dist"] = momepy.neighbor_distance(b, contiguity)
    out["mean_interbuilding_distance"] = momepy.mean_interbuilding_distance(
        b, adjacency_graph=contiguity, neighborhood_graph=band)
    adj_graph = ind.get("adjacency_contiguity", "tessellation")
    if adj_graph == "tessellation":
        adjacency_contiguity = contiguity
    elif adj_graph == "buildings":
        adjacency_contiguity = graph.Graph.build_contiguity(b, rook=False)
    else:
        raise ValueError("indicators.adjacency_contiguity must be 'tessellation' or 'buildings'")
    out["adjacency"] = momepy.building_adjacency(contiguity_graph=adjacency_contiguity, neighborhood_graph=band)
    out["num_neighbors"] = momepy.neighbors(c, contiguity)

    # Further shape indicators
    out["shared_walls"] = momepy.shared_walls(b)
    out["perimeter_wall"] = momepy.perimeter_wall(b)
    out["corners"] = momepy.corners(b, include_interiors=bool(ind.get("corners_include_interiors", True)))
    out["rect_idx"] = momepy.equivalent_rectangular_index(b)
    out["rectangularity"] = momepy.rectangularity(b)
    out["squareness"] = momepy.squareness(b)
    out["square_compact"] = momepy.square_compactness(b)
    out["elongation"] = momepy.elongation(b)
    out["fractal"] = momepy.fractal_dimension(b)

    # Tessellation-based indicators
    out["cwa"] = momepy.compactness_weighted_axis(c)
    out["tes_area"] = c.geometry.area
    out["CAR"] = out["area"] / out["tes_area"]      # own area / own cell area (uID-aligned)
    out["cell_alignment"] = momepy.cell_alignment(momepy.orientation(b), momepy.orientation(c))

    # Connectivity: Delaunay neighbours of the centroids
    out["avg_weighted_dist"] = average_weighted_distance(b.geometry.centroid)

    return out[MODEL_FEATURE_ORDER].astype(float)
