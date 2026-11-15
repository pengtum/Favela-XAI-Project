"""Footprint pre-processing, morphological tessellation and the 25 indicators.

Manuscript items: Main Methods 4.2 'Building-footprint pre-processing' and
4.3 'Morphological feature calculation'; SI Section S1, Section S3 and
Table S2; Main Fig. 1 (five dimensions: 3/13/4/3/2 indicators).

Contents
* ``FEATURES``: the single registry of the 25 indicators (column name, full
  name and figure label as in SI Table S2, morphometric dimension, how the
  indicator is computed and the SI S3 formula). ``MODEL_FEATURE_ORDER`` is the column order
  used by every classifier.
* ``preprocess_footprints``: the pre-processing steps of Methods 4.2.
* ``tessellate``: morphological tessellation with one cell per building and
  exclusion of unsuitable geometries.
* ``minimum_area_rectangle``: the minimum-area bounding rectangle of SI
  Section S3, on which momepy evaluates items 11, 12, 15 and 22.
* ``compute_indicators``: the 25 indicators (Methods 4.3; SI Section S3), with
  momepy or, for areas, perimeters and the covered area ratio (each building's
  own area divided by the area of its own cell), directly from the geometries;
  average weighted distance from a Delaunay triangulation of the centroids.
* ``table_s2_body``: the body of SI Table S2 generated from the registry.

The indicator code uses the momepy functional API with libpysal graphs. The
per-building indicator table of the analysis was computed with momepy 0.8
(Methods 4.14; README, Installation). Later momepy releases (0.11, the version
in environment.yml, for example) simplify the tessellation cells by default,
which changes the cells and the indicators computed on them.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.affinity import affine_transform
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


# Order of SI Section S3 / SI Table S2. The si_formula strings use the notation of
# SI Section S3 for a focal building i: P_i its footprint polygon, A_i its area,
# p_i its perimeter, c_i its centroid; D_i the diameter of the smallest circle that
# encloses P_i; a_i >= b_i the sides of its minimum-area bounding rectangle; V_i its
# cell in the morphological tessellation; d(j,k) the shortest distance between the
# footprints P_j and P_k (0 when they touch or overlap); T_i the buildings whose
# cells are contiguous with V_i (queen contiguity); B_i the buildings whose
# centroids lie within 200 m of c_i, i included, and N_i = |B_i|.
FEATURES: list[Indicator] = [
    # --- Dimension (3) ---
    Indicator("area", "Building area", "area", "area", "Dimension", "building",
              "footprint area (geopandas .area)", "A_i = area(P_i)"),
    Indicator("perimeter", "Building perimeter", "perimeter", "perimeter", "Dimension", "building",
              "footprint perimeter (geopandas .length)", "p_i = length(boundary(P_i))"),
    Indicator("lal", "Longest axis length", "longest axis", "longest axis", "Dimension", "building",
              "momepy.longest_axis_length(buildings)", "D_i, the diameter of the smallest circle that encloses P_i"),
    # --- Shape (13) ---
    Indicator("shape_idx", "Shape index", "shape index", "shape index", "Shape", "building",
              "momepy.shape_index(buildings)", "SI_i = sqrt(A_i / pi) / (D_i / 2)"),
    Indicator("compact_weighted_axis", "Compactness-weighted axis", "compact. axis", r"compact.\ axis", "Shape",
              "building", "momepy.compactness_weighted_axis(buildings)",
              "CWA_i = D_i * (4 / pi - 16 A_i / p_i^2)"),
    Indicator("convexity", "Convexity", "convexity", "convexity", "Shape", "building",
              "momepy.convexity(buildings)", "C_i = A_i / area(H_i), H_i the convex hull of P_i"),
    Indicator("facade_ratio", "Facade ratio", "facade ratio", "facade ratio", "Shape", "building",
              "momepy.facade_ratio(buildings)", "FR_i = A_i / p_i (metres)"),
    Indicator("shared_walls", "Shared walls", "shared walls", "shared walls", "Shape", "building",
              "momepy.shared_walls(buildings)",
              "SW_i = sum over j in W_i of length(intersection(P_i, P_j)), W_i the footprints that touch P_i "
              "without overlapping it: the boundary length P_i shares with other footprints (metres)"),
    Indicator("perimeter_wall", "Perimeter wall", "perimeter wall", "perimeter wall", "Shape", "building",
              "momepy.perimeter_wall(buildings)",
              "PW_i = length of the outer boundary of U_i, U_i the union of P_i with every footprint connected "
              "to it through a chain of footprints that share at least one boundary vertex; the outer boundary "
              "of P_i for a free-standing building (a length in metres, not a ratio)"),
    Indicator("corners", "Number of corners", "corners", "corners", "Shape", "building",
              "momepy.corners(buildings, include_interiors=True)",
              "NC_i = number of vertices of P_i, on its outer boundary and around any interior ring, at which "
              "the outline turns by at least 10 deg"),
    Indicator("rect_idx", "Equivalent rectangular index", "rect. index", r"rect.\ index", "Shape", "building",
              "momepy.equivalent_rectangular_index(buildings), evaluated on the minimum-area bounding rectangle "
              "(SI S3; minimum_area_rectangle)",
              "ERI_i = sqrt(A_i / (a_i b_i)) * 2 (a_i + b_i) / p_i"),
    Indicator("rectangularity", "Rectangularity", "rectangularity", "rectangularity", "Shape", "building",
              "momepy.rectangularity(buildings), evaluated on the minimum-area bounding rectangle "
              "(SI S3; minimum_area_rectangle)", "R_i = A_i / (a_i b_i)"),
    Indicator("squareness", "Squareness", "squareness", "squareness", "Shape", "building",
              "momepy.squareness(buildings)",
              "SQ_i = (1/n_i) sum over k of |theta_ik - 90 deg|, the mean deviation from a right angle (deg) of "
              "the angles at the n_i corners of the outer boundary of P_i (corners as for NC_i); 0 when every "
              "corner is a right angle"),
    Indicator("square_compact", "Square compactness", "square compact", "square compact", "Shape", "building",
              "momepy.square_compactness(buildings)", "SC_i = (4 sqrt(A_i) / p_i)^2"),
    Indicator("elongation", "Elongation", "elongation", "elongation", "Shape", "building",
              "momepy.elongation(buildings), evaluated on the minimum-area bounding rectangle "
              "(SI S3; minimum_area_rectangle)",
              "E_i = b_i / a_i, the shorter over the longer side of the minimum-area bounding rectangle "
              "(0 < E_i <= 1)"),
    Indicator("cwa", "Compactness-weighted axis of tessellation", "compact. axis (t)", r"compact.\ axis (t)",
              "Shape", "tessellation cell", "momepy.compactness_weighted_axis(cells)",
              "CWT_i = D(V_i) * (4 / pi - 16 area(V_i) / length(boundary(V_i))^2), D(V_i) the diameter of the "
              "smallest circle that encloses V_i (the expression of CWA_i evaluated on the cell)"),
    # --- Spatial distribution (4) ---
    Indicator("neighbour_dist", "Mean distance to neighbouring buildings", "neighbour dist.", "neighbour dist.",
              "Spatial distribution", "building and cell",
              "momepy.neighbor_distance(buildings, queen contiguity graph of the tessellation)",
              "ND_i = (1/|T_i|) sum over j in T_i of d(i,j), the mean distance from the footprint of i to the "
              "footprints of the buildings whose cells are contiguous with its own; if T_i is empty, ND_i is "
              "undefined and recorded as missing"),
    Indicator("mean_interbuilding_distance", "Mean interbuilding distance",
              "interbuild. dist.", r"interbuild.\ dist.", "Spatial distribution", "building and cell",
              "momepy.mean_interbuilding_distance(buildings, adjacency_graph=tessellation queen graph, "
              "neighborhood_graph=200 m distance band on centroids)",
              "IBD_i = (1/|E_i|) sum over (j,k) in E_i of d(j,k), E_i = {(j,k): j in B_i, k in B_i and in T_j} "
              "together with (j,j) for each j in B_i with T_j empty: the mean distance between the footprints of "
              "all pairs of buildings within 200 m of i whose cells are contiguous, a building whose cell is "
              "contiguous with no other cell entering once, at distance zero (an average over the neighbourhood "
              "of i, not the distance from i to its nearest neighbour)"),
    Indicator("adjacency", "Adjacency ratio (tessellation contiguity)", "adjacency", "adjacency",
              "Spatial distribution", "building and cell",
              "momepy.building_adjacency(contiguity_graph=queen graph of the tessellation cells "
              "(indicators.adjacency_contiguity), neighborhood_graph=200 m distance band on centroids)",
              "ADJ_i = K_i / N_i, K_i the number of connected components of the site's cell-contiguity graph "
              "(buildings j and k joined when k is in T_j) that are represented among the buildings of B_i: "
              "the momepy building-adjacency statistic on the contiguity graph of the tessellation cells "
              "(not a count of the footprints that touch building i)"),
    Indicator("CAR", "Covered area ratio", "cov-area ratio", "cov-area ratio", "Spatial distribution",
              "building and cell", "building area / area of its own tessellation cell",
              "CAR(i) = A_i / area(V_i)",
              note="not bounded by 1: where footprints touch or overlap, the boundary between their cells "
                   "(drawn from the footprints shrunk by tessellation.shrink and discretised at "
                   "tessellation.segment) can cut into a footprint; 872 of the 188,292 analysed buildings "
                   "have CAR > 1"),
    # --- Configuration (3) ---
    Indicator("tes_area", "Tessellation cell area", "tess. cell area", r"tess.\ cell area", "Configuration",
              "tessellation cell", "cell area (geopandas .area)", "area(V_i), the area of the cell of building i"),
    Indicator("cell_alignment", "Cell alignment", "cell align.", "cell align.", "Configuration",
              "building and cell",
              "momepy.cell_alignment(momepy.orientation(buildings), momepy.orientation(cells)), both "
              "orientations evaluated on the minimum-area bounding rectangle (SI S3; minimum_area_rectangle)",
              "CA_i = |o(P_i) - o(V_i)| (deg), o() the angle between 0 and 45 deg by which the sides of a "
              "polygon's minimum-area bounding rectangle deviate from the cardinal directions"),
    Indicator("num_neighbors", "Tessellation number of neighbours", "no. neighbours", r"no.\ neighbours",
              "Configuration", "tessellation cell", "momepy.neighbors(cells, queen contiguity graph)",
              "TN_i = |T_i|, the number of cells contiguous with V_i"),
    # --- Connectivity (2) ---
    Indicator("fractal", "Fractal dimension", "fractal", "fractal", "Connectivity", "building",
              "momepy.fractal_dimension(buildings)", "FD_i = 2 ln(p_i / 4) / ln(A_i)"),
    Indicator("avg_weighted_dist", "Average weighted distance of building centroids", "avg wt. dist.",
              r"avg wt.\ dist.", "Connectivity", "building",
              "mean length of the Delaunay-triangulation edges incident to the building centroid",
              "d_w(i), the mean length of the Delaunay-triangulation edges incident to the centroid c_i"),
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
      5. second validity check (null, empty, invalid or non-Polygon results removed).

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
# Minimum-area bounding rectangle (SI Section S3; items 11, 12, 15 and 22)
# -----------------------------------------------------------------------------
def _least_area_rectangle(geom):
    """Minimum-area bounding rectangle of one geometry (see minimum_area_rectangle)."""
    if geom is None:
        return None
    if geom.is_empty:
        return Polygon()
    hull = geom.convex_hull
    if hull.geom_type != "Polygon":                 # collinear or coincident points
        return hull
    coords = hull.exterior.coords                   # (x, y) or (x, y, z); z is not used
    best = None
    for p1, p2 in zip(coords[:-1], coords[1:]):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        ux, uy = dx / length, dy / length              # unit vector along the hull edge
        # area of the envelope of the hull rotated into the frame of the edge
        area = affine_transform(hull, (ux, uy, -uy, ux, 0, 0)).envelope.area
        if best is None or area < best[0]:            # the first edge of least area
            best = (area, ux, uy)
    _, ux, uy = best
    # The rectangle along that edge, from the hull coordinates relative to the
    # first hull vertex (s along the edge, t across it).
    xy = np.asarray(coords)[:, :2]
    x0, y0 = xy[0]
    s = (xy[:, 0] - x0) * ux + (xy[:, 1] - y0) * uy
    t = (xy[:, 1] - y0) * ux - (xy[:, 0] - x0) * uy
    corners = [(s.min(), t.min()), (s.max(), t.min()), (s.max(), t.max()), (s.min(), t.max())]
    return Polygon([(x0 + (a * ux - b * uy), y0 + (a * uy + b * ux)) for a, b in corners])


_least_area_rectangles = np.frompyfunc(_least_area_rectangle, 1, 1)


def minimum_area_rectangle(geometry):
    """Minimum-area bounding rectangle of each geometry (SI Section S3).

    The rotated rectangle of least area that contains the geometry. Its sides
    are the a_i >= b_i of the equivalent rectangular index, rectangularity and
    elongation (items 11, 12 and 15), and the orientations of the rectangles of
    a footprint and of its cell give the cell alignment (item 22).

    One side of the rectangle lies along an edge of the convex hull. For each
    hull edge, the hull is rotated into the frame of the edge and the area of
    its envelope is computed, as in shapely's implementation for GEOS < 3.12
    (shapely.algorithms._oriented_envelope); the first edge of least area is
    kept. Its rectangle is computed from the hull coordinates relative to the
    first hull vertex, so that at projected coordinates such as EPSG:31983 it
    contains the geometry up to the rounding of its corner coordinates. The
    computation is the same for every GEOS version: from GEOS 3.12 on,
    shapely.minimum_rotated_rectangle calls a GEOS algorithm instead, whose
    rectangle at coordinates of this size can differ from this one and can
    leave the geometry by up to a few millimetres.

    Takes a geometry or an array-like of geometries, as a shapely function
    does, and returns the rectangles as polygons with four corners; a hull
    that is a point or a line is returned unchanged.
    """
    return _least_area_rectangles(geometry)


@contextlib.contextmanager
def _momepy_on_minimum_area_rectangle():
    """Let momepy take its bounding rectangles from minimum_area_rectangle.

    momepy.equivalent_rectangular_index, momepy.rectangularity,
    momepy.elongation and momepy.orientation obtain the rectangle from
    shapely.minimum_rotated_rectangle. Inside the block that name refers to
    minimum_area_rectangle; it is restored when the block is left.
    """
    previous = shapely.minimum_rotated_rectangle
    shapely.minimum_rotated_rectangle = minimum_area_rectangle
    try:
        yield
    finally:
        shapely.minimum_rotated_rectangle = previous


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
    adjacency ratio (momepy.building_adjacency):
    * ``tessellation``: queen contiguity of the tessellation cells (as analysed);
    * ``buildings``: queen contiguity of the footprints (SI Section S15).
    Items 11, 12, 15 and 22 are evaluated with momepy on the minimum-area
    bounding rectangle of SI Section S3 (:func:`minimum_area_rectangle`).
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

    # Spatial distribution (items 17-19) and tessellation number of neighbours (item 23,
    # configuration)
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

    # Further shape indicators, and fractal dimension (connectivity)
    out["shared_walls"] = momepy.shared_walls(b)
    out["perimeter_wall"] = momepy.perimeter_wall(b)
    out["corners"] = momepy.corners(b, include_interiors=bool(ind.get("corners_include_interiors", True)))
    out["squareness"] = momepy.squareness(b)
    out["square_compact"] = momepy.square_compactness(b)
    out["fractal"] = momepy.fractal_dimension(b)

    # Tessellation-based indicators
    out["cwa"] = momepy.compactness_weighted_axis(c)
    out["tes_area"] = c.geometry.area
    out["CAR"] = out["area"] / out["tes_area"]      # own area / own cell area (uID-aligned)

    # Items 11, 12 and 15 and, from the orientations of footprint and cell, item 22:
    # momepy on the minimum-area bounding rectangle of SI Section S3
    with _momepy_on_minimum_area_rectangle():
        out["rect_idx"] = momepy.equivalent_rectangular_index(b)
        out["rectangularity"] = momepy.rectangularity(b)
        out["elongation"] = momepy.elongation(b)
        out["cell_alignment"] = momepy.cell_alignment(momepy.orientation(b), momepy.orientation(c))

    # Connectivity: Delaunay neighbours of the centroids
    out["avg_weighted_dist"] = average_weighted_distance(b.geometry.centroid)

    return out[MODEL_FEATURE_ORDER].astype(float)
