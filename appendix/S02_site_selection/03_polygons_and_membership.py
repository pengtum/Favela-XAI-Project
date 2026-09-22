"""Step 3: descriptors of the 1,076 official 2013 favela polygons and the study-favela membership.

Manuscript items
* Supplementary Fig. S1: a polygon is a study favela if it contains the
  centroid of at least one favela-labelled building of a site; panels b-d
  (polygon area, compactness 4 pi A / P^2, distance from the polygon centroid
  to Praca XV).
* SI S2.1: number of study polygons among the 1,076 official 2013 polygons
  (counted in 04_compare_study_vs_city.py).

Inputs
* ``paths.raw.favelas_2013``: official 2013 favela polygons (Limite Favelas 2013).
* ``products.features_gpkg`` (main/01_features): analysed buildings of each site
  (centroid and favela label).

Outputs (outputs/appendix/S02_site_selection/)
* ``polygons_base.parquet``: every 2013 polygon (poly_index = row in the
  official layer) with its attributes, area_ha, perimeter_m, compactness,
  n_parts, cx, cy, dist_centre_km and the geometry as WKB.
* ``study_membership.csv``: one row per study polygon and site: number of the
  site's favela-labelled building centroids inside it, number of formal
  (not favela-labelled) centroids inside it, share of the polygon area inside
  the convex hull of the site's buildings.
* ``label_polygon_agreement.csv``: per site, favela-labelled centroids outside
  every polygon and formal centroids inside a polygon (labels are assigned by
  footprint intersection, so a few centroids fall just outside).
* ``site_hulls.parquet``: convex hull of the building centroids of each site.

Settings
* ``site_context.study_favela_rule`` (SI Fig. S1 caption), ``site_context.city_centre_lonlat``.

Run from the repository root:
    python appendix/S02_site_selection/03_polygons_and_membership.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from favela_xai.config import load_config
from favela_xai.io import write_table

import s02_lib as L


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    if cfg.get("site_context.study_favela_rule") != "contains_favela_centroid":
        raise ValueError("only site_context.study_favela_rule = contains_favela_centroid is implemented "
                         "(SI Fig. S1 caption)")
    names = L.site_display_names(cfg)

    pol = L.polygon_descriptors(L.read_polygons_2013(cfg), cfg)
    print(f"2013 favela polygons: {len(pol):,}")

    rows, agree, hulls = [], [], {}
    for site in cfg.sites:
        d = L.site_buildings(cfg, site)
        pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.x, d.y), crs=cfg.get("crs"))
        hulls[site] = shapely.convex_hull(shapely.multipoints(np.c_[d.x, d.y]))
        # centroid within polygon; a centroid inside two overlapping polygons keeps the first one
        j = gpd.sjoin(pts, pol[["cod_favela", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        fav = j[j.y_true == 1]
        agree.append(dict(site=site, n_fav=int(len(fav)), n_fav_outside_polygons=int(fav.index_right.isna().sum()),
                          n_formal_inside_polygon=int(((j.y_true == 0) & j.index_right.notna()).sum())))
        n_fav = fav.dropna(subset=["index_right"]).groupby("index_right").size()
        n_formal = j[j.y_true == 0].dropna(subset=["index_right"]).groupby("index_right").size()
        for idx, n in n_fav.items():
            idx = int(idx)
            g = pol.geometry.iloc[idx]
            rows.append(dict(poly_index=idx, cod_favela=int(pol.cod_favela.iloc[idx]), nome=pol.nome.iloc[idx],
                             site=site, site_name=names[site], n_fav_buildings_in_site=int(n),
                             n_notfav_labelled_inside=int(n_formal.get(idx, 0)),
                             frac_polygon_area_in_site_hull=float(g.intersection(hulls[site]).area / g.area)))
    mem = pd.DataFrame(rows)
    in_two = mem.groupby("poly_index").site.nunique()
    print(f"study polygons: {mem.poly_index.nunique()} ({int((in_two > 1).sum())} in more than one site); per site "
          + ", ".join(f"{k} {v}" for k, v in mem.groupby("site").size().items()))
    print(pd.DataFrame(agree).to_string(index=False))

    write_table(mem, out / L.MEMBERSHIP_CSV)
    write_table(pd.DataFrame(agree), out / "label_polygon_agreement.csv")
    gpd.GeoDataFrame({"site": list(hulls)}, geometry=list(hulls.values()), crs=cfg.get("crs")).to_parquet(
        out / L.SITE_HULLS_PARQUET)
    base = pol.drop(columns="geometry").assign(
        poly_index=np.arange(len(pol)), wkb=shapely.to_wkb(np.asarray(pol.geometry.values, dtype=object)))
    write_table(base, out / L.POLYGONS_PARQUET)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
