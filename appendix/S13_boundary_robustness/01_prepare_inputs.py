"""Step 1 of SI Section S13: per-site inputs, and the boundary-straddling and near-line buildings.

Manuscript items
* SI Section S13 'Buildings on the line, and the position of the line': the
  number of buildings whose footprint crosses a 2013 favela polygon boundary
  and of further buildings whose centroid lies closer to the line than the
  square root of their footprint area (straddler_counts.csv, row 'all':
  n_straddlers, n_near_not_straddler).
* Inputs of every later step of this folder (ED Fig. 5, SI Fig. S22, SI Table S17).

Definitions (SI S13; SI Table S17)
* Signed distance: the Methods 4.8 distance of main-text Figure 4 (centroid to
  the nearest edge of the 2013 polygons, negative for favela-labelled
  buildings), computed with favela_xai.boundary.signed_distance, the function
  that writes products.signed_distance for main/04_boundary_profiles.
* Polygons considered: the 2013 polygons that intersect the site's footprint
  extent padded by ``signed_distance.polygon_search_pad_m`` (300 m).
* Boundary-straddling: the footprint intersects that boundary.
* Near-line: centroid distance to the boundary smaller than sqrt(footprint
  area) (``boundary_robustness.near_line_rule``).

Inputs
* outputs/main/02_classification/<site>_buildings_shap.gpkg (``products.shap_gpkg``):
  footprints, label, the 25 indicators and the Tree-SHAP attributions of the
  main per-site model (Main Methods 4.4, 4.6).
* data/raw/Limite_Favelas_2013/Limite_Favelas_2013.shp (``paths.raw.favelas_2013``).

Outputs (outputs/appendix/S13_boundary_robustness/)
* cache/<site>_buildings.parquet: [uID,] x, y (footprint centroid), y_true
  (1 = favela), footprint_area, signed_dist, straddler, near_line, the 25
  indicators and the 25 shap_ columns (row order of the per-site table).
* cache/<site>_polygons.parquet: the nearby 2013 polygons, repaired with
  shapely.make_valid, in the order of the polygon layer (moved in steps 5 and 6).
* straddler_counts.csv: counts per site and in total.

Run from the repository root:
    python appendix/S13_boundary_robustness/01_prepare_inputs.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

import robustness_common as rc
from favela_xai.boundary import boundary_lines, nearby_polygons, signed_distance
from favela_xai.config import ConfigError
from favela_xai.io import read_favela_polygons, read_site_shap, write_table
from favela_xai.labels import label_to_int


def main() -> None:
    cfg = rc.cfg
    rc.CACHE.mkdir(parents=True, exist_ok=True)
    if rc.BR["near_line_rule"] != "centroid_distance_below_sqrt_area":
        raise ConfigError("only near_line_rule: centroid_distance_below_sqrt_area is implemented (SI S13)")
    pad = float(cfg.get("signed_distance.polygon_search_pad_m"))
    dissolve = bool(cfg.get("signed_distance.dissolve_polygons"))
    polygons_2013 = read_favela_polygons(cfg, 2013)

    counts = []
    for site in cfg.sites:
        path = cfg.require_product("shap_gpkg", site=site)
        fields = set(pyogrio.read_info(path)["fields"])
        missing = [c for c in ["label"] + rc.INDICATORS + rc.SHAP if c not in fields]
        if missing:
            raise ConfigError(f"{path.name} lacks the columns {missing}; re-run main/02_classification.")
        id_cols = ["uID"] if "uID" in fields else []
        g = read_site_shap(cfg, site, columns=id_cols + ["label"] + rc.INDICATORS + rc.SHAP).reset_index(drop=True)
        fav = label_to_int(g["label"]).astype(bool)

        # Methods 4.8 signed distance (negative on the favela-labelled side)
        signed = signed_distance(g, fav, polygons_2013, cfg)
        dist = np.abs(signed)

        geoms = np.asarray(g.geometry.values, dtype=object)
        centroids = shapely.centroid(geoms)
        area = shapely.area(geoms)

        # 2013 polygons near the site and their boundary (same selection and
        # boundary definition as the signed distance)
        near = nearby_polygons(polygons_2013.to_crs(g.crs), g.total_bounds, pad)
        near_geoms = np.asarray(near.geometry.values, dtype=object)
        line_parts = shapely.get_parts(boundary_lines(near, dissolve))

        # boundary-straddling: the footprint intersects the boundary
        hits = shapely.STRtree(geoms).query(line_parts, predicate="intersects")
        straddler = np.zeros(len(g), dtype=bool)
        straddler[np.unique(hits[1])] = True
        # near-line: centroid closer to the boundary than sqrt(footprint area)
        near_line = dist < np.sqrt(area)

        table = pd.DataFrame({"x": shapely.get_x(centroids), "y": shapely.get_y(centroids),
                              "y_true": fav.astype(int), "footprint_area": area, "signed_dist": signed,
                              "straddler": straddler, "near_line": near_line})
        if id_cols:
            table.insert(0, "uID", g["uID"].to_numpy())
        for c in rc.INDICATORS + rc.SHAP:
            table[c] = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
        table.to_parquet(rc.buildings_path(site), index=False)
        gpd.GeoDataFrame(geometry=shapely.make_valid(near_geoms), crs=g.crs).to_parquet(rc.polygons_path(site))

        within = dist <= rc.HALF
        either = straddler | near_line
        counts.append(dict(
            site=site, site_name=cfg.site_name(site), n_buildings=len(g), n_favela=int(fav.sum()),
            n_polygons=len(near), n_within_40m=int(within.sum()),
            n_straddlers=int(straddler.sum()), n_straddlers_favela=int((straddler & fav).sum()),
            n_straddlers_formal=int((straddler & ~fav).sum()),
            n_near_line=int(near_line.sum()), n_near_not_straddler=int((near_line & ~straddler).sum()),
            n_straddler_or_near=int(either.sum()), n_straddler_or_near_favela=int((either & fav).sum()),
            n_straddler_or_near_formal=int((either & ~fav).sum())))
        print(f"{site:24s} n={len(g):6d} polygons={len(near):3d} straddling={counts[-1]['n_straddlers']:5d} "
              f"near-line only={counts[-1]['n_near_not_straddler']:5d}", flush=True)

    cnt = pd.DataFrame(counts)
    total = cnt.drop(columns=["site", "site_name"]).sum()
    total["site"], total["site_name"] = "all", "All sites"
    cnt = pd.concat([cnt, pd.DataFrame([total])], ignore_index=True)
    for part in ("straddlers", "straddler_or_near"):
        cnt[f"pct_{part}"] = 100 * cnt[f"n_{part}"] / cnt["n_buildings"]
        cnt[f"pct_{part}_of_within_40m"] = 100 * cnt[f"n_{part}"] / cnt["n_within_40m"]
    write_table(cnt, rc.OUT / "straddler_counts.csv")
    tot = cnt.set_index("site").loc["all"]
    print(f"\nboundary-straddling buildings: {int(tot.n_straddlers):,} ({int(tot.n_straddlers_favela):,} "
          f"favela-labelled); near-line, not straddling: {int(tot.n_near_not_straddler):,}")
    print("Reported in SI S13: 4,110 straddling (4,106 favela-labelled) and a further 6,742 near-line buildings.")


if __name__ == "__main__":
    main()
