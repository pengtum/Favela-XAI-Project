"""Step 1: per-site building tables for the spatial cross-validation.

Manuscript items: input of every result of SI Section S12 'Spatial
cross-validation' (Tables S10-S16, Figs S19-S21), Extended Data Figs 3-4 and the
blocked boundary error rates quoted in Results 2.1 and SI S5.

For every site two tables are written, in the row order of the per-site
feature table of main/01_features:
  <site>.parquet             uID, x, y (footprint centroid, EPSG:31983),
                             is_favela (1 = favela, 0 = formal settlement),
                             signed_dist (signed distance to the 2013 boundary,
                             Methods 4.8) and the 25 indicators. Missing
                             indicator values stay missing: every fold imputes
                             them with the median of its own training rows
                             (SI S12 'Designs').
  <site>_main_shap.parquet   the 25 Tree-SHAP values of the main per-site model
                             (Methods 4.4 and 4.6; positive toward favela), the
                             reference ranking of SI S12 'Stability of the
                             attributions', SI Table S16 and ED Figs 3-4.

Inputs  (from main/01_features and main/02_classification, see config.yaml 'products')
  outputs/main/01_features/<site>_buildings_features.gpkg
  outputs/main/01_features/signed_distance.parquet (computed and written by the package if absent)
  outputs/main/02_classification/<site>_buildings_shap.gpkg
Outputs
  outputs/appendix/S12_spatial_cv/site_tables/<site>.parquet, <site>_main_shap.parquet
Run
  python appendix/S12_spatial_cv/01_build_site_tables.py
"""
import numpy as np
import pandas as pd

from favela_xai.boundary import load_signed_distance
from favela_xai.io import read_site_features, read_site_shap
from favela_xai.labels import label_to_int

import s12_common as C

cfg, s, P = C.setup(__file__)
log = C.log_to(P.logs / "01_build_site_tables.log")

# Signed distance of every building (Methods 4.8), keyed by site and uID
SD = load_signed_distance(cfg, cfg.sites)

for site in cfg.sites:
    g = read_site_features(cfg, site)
    missing = [c for c in ["uID", "label"] + C.FEATURES if c not in g.columns]
    if missing:
        raise SystemExit(f"{site}: feature table lacks columns {missing}")
    cen = g.geometry.centroid
    t = pd.DataFrame({"uID": g["uID"].to_numpy(), "x": cen.x.to_numpy(), "y": cen.y.to_numpy(),
                      "is_favela": label_to_int(g["label"]).astype(np.int8)})
    for f in C.FEATURES:
        t[f] = g[f].to_numpy(dtype=float)

    # signed distance, joined by uID (the package table covers the same buildings)
    sd = SD[SD["site"] == site].set_index("uID")["signed_dist"]
    t["signed_dist"] = t["uID"].map(sd).to_numpy(float)
    if t["signed_dist"].isna().any():
        raise SystemExit(f"{site}: {int(t.signed_dist.isna().sum())} buildings without a signed distance")
    # sign convention of Methods 4.8: negative on the favela-labelled side
    assert ((t.signed_dist < 0) == (t.is_favela == 1)).mean() > 0.999, site

    # Tree-SHAP of the main per-site model, aligned to the same rows
    sh = read_site_shap(cfg, site)
    sh = pd.DataFrame(sh.drop(columns="geometry"))
    lacking = [c for c in C.SHAP_COLS if c not in sh.columns]
    if lacking:
        raise SystemExit(f"{site}: SHAP table lacks columns {lacking[:3]} ...")
    if "uID" in sh.columns:
        sh = sh.set_index("uID").reindex(t["uID"].to_numpy())
        if sh[C.SHAP_COLS].isna().any().any():
            raise SystemExit(f"{site}: buildings of the feature table without main-model SHAP values")
    else:
        # no identifier: the SHAP table must list the same buildings in the same order
        assert len(sh) == len(t), site
        assert (label_to_int(sh["label"]) == t["is_favela"].to_numpy()).all(), site
    shap_main = pd.DataFrame(sh[C.SHAP_COLS].to_numpy(float), columns=C.SHAP_COLS)
    shap_main.insert(0, "uID", t["uID"].to_numpy())

    t.to_parquet(P.site_tables / f"{site}.parquet", index=False)
    shap_main.to_parquet(P.site_tables / f"{site}_main_shap.parquet", index=False)
    log(f"{site:<24} buildings {len(t):>6}  favela {int(t.is_favela.sum()):>6}  "
        f"missing indicator values {int(np.isnan(t[C.FEATURES].to_numpy()).sum())}")
log("done")
