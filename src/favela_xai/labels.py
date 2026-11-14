"""Favela / formal-settlement labels from the official favela polygons.

Manuscript items: Main Methods 4.1 (favela labels of the buildings of the ten
sites, from the official 2013 favela boundaries) and Methods 4.12 / SI S10
(2013-2019 overlay: a building counts as inside in a given year when its
footprint intersects the dissolved favela polygons of that year).

Label rule (``labels.rule`` in config.yaml): with ``footprint_intersects`` a
building is labelled favela when its footprint intersects the dissolved 2013
favela polygons (the rule of the 2013-2019 overlay); ``centroid_within`` uses
the footprint centroid instead.

Label values are the strings 'favela' and 'not-favela'; ``label_to_int`` gives
1 for favela and 0 for formal settlement.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

FAVELA = "favela"
FORMAL = "not-favela"

OVERLAY_STATUS = ["favela_in_both", "formal_in_both", "added_by_2019", "removed_by_2019"]


def polygon_union(polygons: gpd.GeoDataFrame):
    """Dissolved union of favela polygons (invalid polygons repaired first), prepared."""
    geoms = np.asarray(polygons.geometry.values, dtype=object)
    geoms = geoms[~(shapely.is_missing(geoms) | shapely.is_empty(geoms))]
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])
    union = shapely.union_all(geoms)
    shapely.prepare(union)
    return union


def inside_polygons(buildings: gpd.GeoDataFrame, union, rule: str = "footprint_intersects") -> np.ndarray:
    """Boolean array: is each building inside the (dissolved) favela polygons?

    ``footprint_intersects``: the footprint intersects the union (labels, overlay);
    ``centroid_within``: the footprint centroid lies inside the union.
    """
    geoms = np.asarray(buildings.geometry.values, dtype=object)
    if rule == "footprint_intersects":
        return shapely.intersects(union, geoms)
    if rule == "centroid_within":
        return shapely.contains(union, shapely.centroid(geoms))
    raise ValueError(f"unknown rule '{rule}'")


def assign_labels(buildings: gpd.GeoDataFrame, polygons_2013: gpd.GeoDataFrame, cfg) -> pd.Series:
    """Label every building 'favela' or 'not-favela' from the 2013 polygons (config ``labels``)."""
    lab = cfg.section("labels")
    if not lab.get("dissolve_polygons", True):
        raise ValueError("labels.dissolve_polygons must be true (the labels are computed on the dissolved 2013 polygons)")
    union = polygon_union(polygons_2013.to_crs(buildings.crs))
    inside = inside_polygons(buildings, union, lab["rule"])
    return pd.Series(np.where(inside, lab.get("favela", FAVELA), lab.get("formal", FORMAL)),
                     index=buildings.index, name="label")


def label_to_int(labels) -> np.ndarray:
    """1 = favela, 0 = formal settlement."""
    return (np.asarray(labels).astype(str) == FAVELA).astype(int)


def overlay_status(buildings: gpd.GeoDataFrame, polygons_2013: gpd.GeoDataFrame,
                   polygons_2019: gpd.GeoDataFrame, rule: str = "footprint_intersects") -> pd.Series:
    """Status of each building in the 2013 and 2019 boundaries (Methods 4.12; SI S10).

    favela_in_both, formal_in_both, added_by_2019 (outside 2013, inside 2019) or
    removed_by_2019 (inside 2013, outside 2019).
    """
    i13 = inside_polygons(buildings, polygon_union(polygons_2013.to_crs(buildings.crs)), rule)
    i19 = inside_polygons(buildings, polygon_union(polygons_2019.to_crs(buildings.crs)), rule)
    status = np.select([i13 & i19, ~i13 & ~i19, ~i13 & i19, i13 & ~i19], OVERLAY_STATUS, default="")
    return pd.Series(pd.Categorical(status, categories=OVERLAY_STATUS), index=buildings.index, name="status")
