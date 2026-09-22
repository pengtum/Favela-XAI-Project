"""Step 2: held-out favela component groups (the 'held-out favela component' design).

Manuscript items: the held-out favela component design of SI S12 'Designs'
(groups of neighbouring 2013 favela polygons with their nearest formal-settlement
buildings, at least 100 buildings of each class; 58 groups in eight sites);
groups per site in SI Fig. S21; used in SI Tables S10, S12-S16 and ED Figs 3-4.

Rules
1. Component of a favela building: the 2013 favela polygon its footprint
   intersects. A footprint that intersects several polygons is given to the one
   that contains its centroid; if none or more than one does, to the polygon
   nearest to the centroid (ties: smaller polygon code).
2. Component of a formal-settlement building: the component polygon (among the
   polygons holding favela buildings of the site) nearest to its centroid
   (ties: smaller code).
3. Merging: while any group has fewer than
   spatial_cv.component_min_buildings_per_class buildings of either class, the
   deficient group with the smallest min(favela, formal) count (ties: fewer
   buildings, then smaller code) is merged into the group whose polygons are
   nearest to it (ties: fewer buildings, then smaller code). A group keeps the
   code of its founding polygon. A site left with one group cannot be tested.

Inputs  outputs/main/01_features/<site>_buildings_features.gpkg (footprints)
        site tables of step 1; data/raw/Limite_Favelas_2013 (paths.raw.favelas_2013)
Outputs outputs/appendix/S12_spatial_cv/component_groups/<site>_groups.parquet
          (uID, component, group, is_favela; row order of the site table)
        outputs/appendix/S12_spatial_cv/component_groups/components.csv (one row per polygon)
        outputs/appendix/S12_spatial_cv/component_groups/component_log.csv (one row per site)
Run     python appendix/S12_spatial_cv/02_component_groups.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from favela_xai.io import read_favela_polygons, read_site_features

import s12_common as C

cfg, s, P = C.setup(__file__)
S12 = C.section(cfg)
log = C.log_to(P.logs / "02_component_groups.log")
CODE, NAME = S12["polygon_code_field"], S12["polygon_name_field"]
MINN = int(cfg.get("spatial_cv.component_min_buildings_per_class"))
CRS = cfg.get("crs")

POLY = read_favela_polygons(cfg, 2013)[[CODE, NAME, "geometry"]].copy()
POLY[CODE] = POLY[CODE].astype(int)
POLY[NAME] = POLY[NAME].astype(str).str.strip()
# invalid polygons are repaired (as for the favela labels, favela_xai.labels) so that
# the spatial predicates below cannot fail; polygon codes are unchanged
bad = ~POLY.geometry.is_valid
if bad.any():
    POLY.loc[bad, "geometry"] = shapely.make_valid(POLY.loc[bad, "geometry"].values)
    log(f"repaired {int(bad.sum())} invalid 2013 polygons with shapely.make_valid")
PBY = POLY.set_index(CODE)

comp_rows, site_rows = [], []
for site in cfg.sites:
    d = C.load_site(P, site)
    fp = read_site_features(cfg, site, columns=["uID"])
    fp = fp.set_index("uID").reindex(d.table["uID"].to_numpy())       # footprints in site-table order
    assert fp.geometry.notna().all(), site
    fav = d.y == 1
    pts = shapely.points(d.xy)
    comp = np.full(d.n, -1, dtype=int)

    # ---- rule 1: favela buildings -> intersected 2013 polygon -------------------------------
    fav_idx = np.where(fav)[0]
    fg = gpd.GeoDataFrame({"i": fav_idx}, geometry=fp.geometry.values[fav_idx], crs=CRS)
    sj = gpd.sjoin(fg, POLY[[CODE, "geometry"]], how="left", predicate="intersects")
    cand = sj.dropna(subset=[CODE]).groupby("i")[CODE].apply(lambda v: sorted(set(int(c) for c in v)))
    n_multi = n_by_centroid = n_none = 0
    for i in fav_idx:
        cands = cand.get(i, [])
        if len(cands) == 1:
            comp[i] = cands[0]
            continue
        if len(cands) == 0:                         # label and polygons disagree: nearest polygon overall
            n_none += 1
            cands = POLY[CODE].tolist()
        else:
            n_multi += 1
        inside = [k for k in cands if PBY.loc[k, "geometry"].contains(pts[i])]
        if len(inside) == 1:
            comp[i] = inside[0]
            n_by_centroid += 1
        else:
            dist = [PBY.loc[k, "geometry"].distance(pts[i]) for k in cands]
            comp[i] = cands[int(np.argmin(dist))]
    codes = sorted(np.unique(comp[fav]).tolist())
    cpoly = PBY.loc[codes, "geometry"]

    # ---- rule 2: formal buildings -> nearest component polygon ------------------------------
    frm_idx = np.where(~fav)[0]
    fo = gpd.GeoDataFrame({"i": frm_idx}, geometry=pts[frm_idx], crs=CRS)
    cg = gpd.GeoDataFrame({"code": codes}, geometry=cpoly.values, crs=CRS)
    nj = gpd.sjoin_nearest(fo, cg, how="left", distance_col="dist")
    nj = nj.sort_values(["i", "code"]).drop_duplicates("i").set_index("i")
    comp[frm_idx] = nj.loc[frm_idx, "code"].to_numpy(int)

    nfav0 = {k: int(((comp == k) & fav).sum()) for k in codes}
    nfor0 = {k: int(((comp == k) & ~fav).sum()) for k in codes}

    # ---- rule 3: merge groups below the minimum size ----------------------------------------
    groups = {k: [k] for k in codes}
    geom = {k: cpoly.loc[k] for k in codes}
    nf, nn = dict(nfav0), dict(nfor0)
    merges = []
    while len(groups) > 1:
        bad = [g for g in groups if nf[g] < MINN or nn[g] < MINN]
        if not bad:
            break
        g = sorted(bad, key=lambda k: (min(nf[k], nn[k]), nf[k] + nn[k], k))[0]
        others = [h for h in groups if h != g]
        dist = {h: geom[g].distance(geom[h]) for h in others}
        h = sorted(others, key=lambda k: (dist[k], nf[k] + nn[k], k))[0]
        merges.append(f"{g}->{h} ({dist[h]:.1f} m)")
        groups[h] += groups.pop(g)
        geom[h] = geom[h].union(geom.pop(g))
        nf[h] += nf.pop(g)
        nn[h] += nn.pop(g)
    member_to_group = {m: g for g, ms in groups.items() for m in ms}
    grp = np.array([member_to_group[k] for k in comp])
    testable = len(groups) >= 2 and all(nf[g] >= MINN and nn[g] >= MINN for g in groups)

    pd.DataFrame({"uID": d.table["uID"].to_numpy(), "component": comp.astype(np.int32),
                  "group": grp.astype(np.int32), "is_favela": d.y.astype(np.int8)}).to_parquet(
        P.groups / f"{site}_groups.parquet", index=False)
    for k in codes:
        g = member_to_group[k]
        comp_rows.append(dict(site=site, site_name=cfg.site_name(site), polygon_code=k, polygon_name=PBY.loc[k, NAME],
                              n_favela_buildings=nfav0[k], n_formal_assigned=nfor0[k], group=g,
                              group_members="+".join(map(str, sorted(groups[g]))), group_n_favela=nf[g],
                              group_n_formal=nn[g], n_groups_site=len(groups), site_testable=testable))
    site_rows.append(dict(site=site, n=d.n, n_favela=int(fav.sum()), n_components=len(codes), n_groups=len(groups),
                          testable=testable, n_footprints_in_several_polygons=n_multi,
                          n_assigned_by_centroid=n_by_centroid, n_favela_without_polygon=n_none,
                          merges="; ".join(merges)))
    log(f"{site:<24} components {len(codes):>2}  groups {len(groups):>2}  testable {testable}  "
        f"footprints in several polygons {n_multi} (by centroid {n_by_centroid})  favela without polygon {n_none}")

pd.DataFrame(comp_rows).to_csv(P.groups / "components.csv", index=False)
L = pd.DataFrame(site_rows)
L.to_csv(P.groups / "component_log.csv", index=False)
log(f"testable sites {int(L.testable.sum())}; groups in testable sites {int(L[L.testable].n_groups.sum())} "
    "(SI S12: 58 groups in eight sites)")
