"""Step 6: Supplementary Table S1, context of the ten study sites (and the site numbers of SI S2.1 / Methods).

Manuscript items
* Supplementary Table S1: bairro, planning area, window,
  favela share, favela polygons, mean slope of favela and formal-settlement
  buildings and setting of each site, ordered by distance from Praca XV.
* SI S2.1 and Methods 'Study area and building footprints': distance range of
  the sites from the centre, number of hillside and flat sites, planning areas,
  window sizes and favela share (written to site_context_numbers.txt).

Definitions (Supplementary Table S1)
* Location of a site: the mean position of its favela-labelled buildings.
  Bairro and planning area (AP1-AP5) = the Data.Rio polygons containing it.
  Distance from the centre = Euclidean distance from Praca XV.
* Window: east-west x north-south extent of the site's analysed building centroids.
* Favela share: favela-labelled buildings / all analysed buildings.
* Favela polygons: 2013 favela polygons containing at least one favela-labelled
  building centroid of the site.
* Mean slope: mean of the per-building mean slope (5 m LiDAR terrain model,
  SI S11) over the site's favela and formal-settlement buildings; setting =
  hillside if the favela mean slope is at least
  ``terrain.hillside_min_favela_slope_deg``, flat if it is below
  ``terrain.flat_max_favela_slope_deg``.
The CSV also holds the share of the window filled by the convex hull of the
site's buildings and the share of the site's buildings inside the convex hull
of its favela polygons.

Inputs
* ``products.features_gpkg`` (main/01_features): analysed buildings of each site.
* context_layers/rio_bairros.gpkg, context_layers/rio_planning_areas.gpkg (step 01).
* ``paths.raw.favelas_2013``: official 2013 favela polygons.
* outputs/appendix/S11_terrain/<site>_buildings_with_terrain.gpkg (step 01 of
  appendix/S11_terrain; needs the 5 m terrain model, available from IPP on
  request, path in ``paths.raw.elevation_raster``). Without these files the
  slope and setting columns are left empty.

Outputs (outputs/appendix/S02_site_selection/)
* site_context_table.csv: all columns above, one row per site.
* table_R1_site_context.tex: Supplementary Table S1 (LaTeX).
* ap_site_check.csv: planning area at the site location and at a point inside
  the site's bairro (they must agree).
* site_context_numbers.txt: the SI S2.1 and Methods numbers.

Settings: ``site_context`` and ``appendix_S02_site_selection`` (table layout),
``terrain.hillside_min_favela_slope_deg`` / ``terrain.flat_max_favela_slope_deg``.

Run from the repository root:
    python appendix/S02_site_selection/06_site_context_table.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from favela_xai.config import load_config
from favela_xai.io import write_table, write_text
from favela_xai.labels import FAVELA

import s02_lib as L

# LaTeX accents used in the site and bairro names
TEX_ACCENTS = {"ã": r"\~a", "á": r"\'a", "í": r"\'i", "ç": r"\c{c}", "é": r"\'e", "ó": r"\'o", "ô": r"\^o", "ú": r"\'u"}

# Caption of Supplementary Table S1
CAPTION = [
    r"\caption{Context of the ten study sites. Sites are ordered by distance from the city centre (Pra\c{c}a XV).",
    r"  Bairro: municipal neighbourhood containing the mean position of the site's favela buildings; planning",
    r"  area: AP1 Centro, AP2 Zona Sul, AP3 Zona Norte, AP4 and AP5 Zona Oeste. Window: east--west $\times$",
    r"  north--south extent of the site's analysed buildings. Favela share: share of the window's buildings",
    r"  labelled favela from the official 2013 favela boundaries. Polygons: 2013 favela polygons containing the",
    r"  site's favela buildings. Mean slope:",
    r"  mean building slope from the 5\,m LiDAR terrain model (Section~\ref{sec:terrain}) for favela and",
    r"  formal-settlement buildings; hillside if the favela mean slope is at least 10$^{\circ}$, flat if it is",
    r"  below 6$^{\circ}$ (no site lies in between).",
    r"  The Palmeiras site is named `Palmares' in the official 2013 favela boundaries.}",
]


def tex(s: str) -> str:
    for k, v in TEX_ACCENTS.items():
        s = s.replace(k, v)
    return s


def dec(x: float, nd: int = 1) -> str:
    """Decimal as in the SI tables; '--' when missing."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def terrain_means(cfg, site: str):
    """Mean building slope of favela and formal-settlement buildings (SI S11 per-building terrain).

    Returns (favela mean, formal mean) or (nan, nan) when the terrain step has not been run.
    """
    rel = cfg.get("appendix_S02_site_selection.terrain_buildings").format(site=site)
    path = cfg.outputs_root / rel
    if not path.exists():
        return np.nan, np.nan
    df = pyogrio.read_dataframe(path, columns=["label", "slope_mean"], read_geometry=False)
    fav = df["label"].astype(str) == FAVELA
    return float(df.loc[fav, "slope_mean"].mean()), float(df.loc[~fav, "slope_mean"].mean())


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    s = cfg.section("appendix_S02_site_selection")
    crs = cfg.get("crs")
    hill = float(cfg.get("terrain.hillside_min_favela_slope_deg"))
    flat = float(cfg.get("terrain.flat_max_favela_slope_deg"))
    tol = float(s["polygon_hull_tolerance_m"])
    zone = {f"AP{k}": v for k, v in s["planning_area_names"].items()}

    bai = gpd.read_file(L.require(out / L.BAIRROS_GPKG, "01_context_layers.py"), layer="bairros")
    ap = gpd.read_file(L.require(out / L.PLANNING_AREAS_GPKG, "01_context_layers.py"), layer="planning_areas")
    pol = L.read_polygons_2013(cfg)
    cx0, cy0 = L.city_centre_xy(cfg)

    rows, checks = [], []
    missing_terrain = []
    for site in cfg.sites:
        d = L.site_buildings(cfg, site)
        pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.x, d.y), crs=crs)
        fav, formal = pts[pts.y_true == 1], pts[pts.y_true == 0]

        # site location: mean position of the favela-labelled buildings
        fx, fy = float(fav.x.mean()), float(fav.y.mean())
        loc = shapely.Point(fx, fy)
        hit = bai[bai.contains(loc)]
        if len(hit):
            bairro = hit.bairro.iloc[0]
        else:                                        # not expected; nearest bairro, flagged
            bairro = bai.bairro.iloc[int(np.argmin(bai.distance(loc)))] + " (nearest)"
        hit_ap = ap[ap.contains(loc)]
        planning_area = hit_ap.ap.iloc[0] if len(hit_ap) else "none"
        # check: the planning area of a point inside the site's bairro
        bpoly = bai[bai.bairro == bairro.replace(" (nearest)", "")].geometry
        ap_of_bairro = "none"
        if len(bpoly):
            rp = bpoly.iloc[0].representative_point()
            h2 = ap[ap.contains(rp)]
            ap_of_bairro = h2.ap.iloc[0] if len(h2) else "none"
        checks.append(dict(site=site, name=cfg.site_name(site), bairro=bairro, x=fx, y=fy,
                           planning_area_at_site=planning_area, planning_area_of_bairro=ap_of_bairro,
                           agree=planning_area == ap_of_bairro))

        # window, favela share, favela polygons
        wx, wy = d.x.max() - d.x.min(), d.y.max() - d.y.min()
        used = gpd.sjoin(pol[["cod_favela", "nome", "geometry"]], fav[["geometry"]], predicate="intersects",
                         how="inner").index.unique()
        poly_hull = shapely.union_all(np.asarray(pol.loc[used].geometry.values, dtype=object)).convex_hull
        bld_hull = shapely.convex_hull(shapely.multipoints(np.c_[d.x, d.y]))
        in_poly_hull = shapely.contains_xy(poly_hull.buffer(tol), d.x.to_numpy(), d.y.to_numpy())
        formal_bairros = gpd.sjoin(formal[["geometry"]], bai[["bairro", "geometry"]], predicate="within", how="left")

        # terrain (SI S11 per-building slope on the main-analysis building set)
        fs, ns = terrain_means(cfg, site)
        if not np.isfinite(fs):
            missing_terrain.append(site)
            setting = ""
        else:
            setting = "hillside" if fs >= hill else ("flat" if fs < flat else "intermediate")

        rows.append(dict(
            site=site, site_name=cfg.site_name(site), bairro=bairro, planning_area=planning_area,
            planning_area_zone=zone.get(planning_area, ""),
            distance_from_centre_km=float(np.hypot(fx - cx0, fy - cy0) / 1e3),
            window_ew_km=wx / 1e3, window_ns_km=wy / 1e3, window_km=f"{wx / 1e3:.1f} x {wy / 1e3:.1f}",
            n_buildings=len(pts), n_favela=len(fav), window_favela_share=len(fav) / len(pts),
            n_favela_polygons_2013=len(used),
            favela_polygon_names="; ".join(sorted(pol.loc[used, "nome"].astype(str))),
            mean_slope_favela_deg=fs, mean_slope_formal_deg=ns, setting=setting,
            window_fill_hull_over_rectangle=float(bld_hull.area / (wx * wy)),
            share_buildings_in_polygon_hull=float(in_poly_hull.mean()),
            formal_bairros_top3=", ".join(formal_bairros.bairro.value_counts().index[:3].tolist())))

    T = pd.DataFrame(rows).sort_values("distance_from_centre_km").reset_index(drop=True)
    write_table(T, out / L.TABLE_S1_CSV)
    C = pd.DataFrame(checks)
    write_table(C, out / "ap_site_check.csv")
    if not C.agree.all():
        print("WARNING planning area at the site location and of its bairro differ for: "
              + ", ".join(C.loc[~C.agree, "site"]))
    if missing_terrain:
        print("NOTE the per-building terrain of appendix/S11_terrain (step 01) is missing for "
              + ", ".join(missing_terrain) + "; the slope and setting columns are left empty. "
              "The 5 m terrain model is not distributed (IPP on request; paths.raw.elevation_raster).")
    if (T.setting == "intermediate").any():
        print("NOTE a site has a favela mean slope between the flat and hillside thresholds; the caption "
              "sentence '(no site lies in between)' no longer holds: "
              + ", ".join(T.loc[T.setting == "intermediate", "site_name"]))

    # ---- Palmeiras site: names of its 2013 polygons (log only)
    pal = T[T.site == "palmeras"]
    if len(pal):
        print(f"Palmeiras site, 2013 polygon names: {pal.favela_polygon_names.iloc[0]}")

    # ---- LaTeX
    lines = [r"% Supplementary Table S1 (SI Section S2.1): study-site context. Generated by",
             r"% appendix/S02_site_selection/06_site_context_table.py",
             r"\begin{table}[H]", r"\centering"] + CAPTION + [
             r"\label{tab:site_context}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r" & & Planning & Window & Favela & Favela & Mean slope ($^{\circ}$), & \\",
             r"Site & Bairro & area & (km) & share (\%) & polygons & favela / formal & Setting \\",
             r"\midrule"]
    for r in T.itertuples():
        lines.append(f"{tex(r.site_name)} & {tex(r.bairro)} & {r.planning_area} & "
                     f"{dec(r.window_ew_km)} $\\times$ {dec(r.window_ns_km)} & {dec(100 * r.window_favela_share)} & "
                     f"{int(r.n_favela_polygons_2013)} & {dec(r.mean_slope_favela_deg)} / {dec(r.mean_slope_formal_deg)} & "
                     f"{r.setting or '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_text("\n".join(lines) + "\n", out / L.TABLE_S1_TEX)

    # ---- in-text numbers (SI S2.1; Main Methods 'Study area and building footprints')
    sh = 100 * T.window_favela_share
    big = T.loc[(T.window_ew_km * T.window_ns_km).idxmax()]
    small = T.loc[(T.window_ew_km * T.window_ns_km).idxmin()]
    rest = T[T.site != big.site]
    largest_rest = rest.loc[(rest.window_ew_km * rest.window_ns_km).idxmax()]
    nums = [
        f"distance from the historic centre: {T.distance_from_centre_km.min():.1f}-{T.distance_from_centre_km.max():.1f} km",
        f"setting: hillside {int((T.setting == 'hillside').sum())}, flat {int((T.setting == 'flat').sum())}, "
        f"intermediate {int((T.setting == 'intermediate').sum())}, no terrain {int((T.setting == '').sum())}",
        "planning areas: " + "; ".join(f"{a} ({', '.join(T.loc[T.planning_area == a, 'site_name'])})"
                                       for a in sorted(T.planning_area.unique())),
        f"windows: smallest {small.window_km} km ({small.site_name}); largest apart from {big.site_name} "
        f"{largest_rest.window_km} km ({largest_rest.site_name}); {big.site_name} {big.window_km} km",
        f"favela share of a window: {sh.min():.1f}-{sh.max():.1f}% (median {sh.median():.1f}%); "
        f"rounded {sh.min():.0f}-{sh.max():.0f}% (median {sh.median():.0f}%)",
        f"analysed buildings {int(T.n_buildings.sum()):,}, favela-labelled {int(T.n_favela.sum()):,}",
        f"favela polygons per site: {int(T.n_favela_polygons_2013.min())}-{int(T.n_favela_polygons_2013.max())}",
    ]
    write_text("\n".join(nums) + "\n", out / "site_context_numbers.txt")
    print("\n".join(nums))
    print(T[["site_name", "bairro", "planning_area", "window_km", "window_favela_share", "n_favela_polygons_2013",
             "mean_slope_favela_deg", "mean_slope_formal_deg", "setting"]].round(3).to_string(index=False))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
