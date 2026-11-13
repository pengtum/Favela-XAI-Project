"""Boundary profiles of the 25 Tree-SHAP contributions and their transition metrics.

Manuscript items
* Methods 4.8: signed distance of every building to the official 2013 favela
  boundary (negative for favela buildings, positive for non-favela buildings),
  per-bin mean of each indicator's Tree-SHAP contribution in 2.5 m bins,
  cross-site mean and +-1 s.d. across the ten sites, and the metrics |dS|, x0
  and w.
* Fig. 4: data of the 25 panels (drawn by 02_figure4.py).
* SI Table S5: all values (written by 03_table_S5.py).
* SI Section S6: x0 = +4.7 m (mean interbuilding distance), +4.2 m
  (adjacency), +4.9 m (perimeter wall), +3.0 m (squareness), widths of about
  16-18 m, average weighted distance changing sign about 6 m inside.
* Results 2.3: the leading spatial-distribution indicators change sign a few
  metres outside the polygon over a graded transition with narrow cross-site
  bands.

Definitions (config.yaml, sections 'signed_distance' and 'boundary_profiles')
* Signed distance: Euclidean distance from the building centroid to the nearest
  edge of the 2013 favela polygons, negative for favela-labelled buildings
  (Methods 4.8). It is the shared product of main/01_features
  (favela_xai.boundary.load_signed_distance), the same distance as in the
  misclassification-rate analysis (Methods 4.7).
* Profile of one site: mean SHAP value in 2.5 m bins between -40 and +40 m; a
  bin enters only with at least 20 buildings of that site. Cross-site profile:
  mean and standard deviation (ddof 0) of the site profiles, bin by bin, over
  the sites that have the bin.
* |dS|: mean of the cross-site profile over bins centred at <= -25 m minus the
  mean over bins centred at >= +25 m.
* x0: first change of sign from favela to formal evidence (inside to outside)
  of the cross-site profile after Gaussian smoothing (smoothing_sigma_bins,
  1 bin), linear interpolation between bin centres.
* w = sqrt(sum |S'| (x - xbar)^2 / sum |S'|) on the smoothed profile.
* x0 and w are reported only where |dS| >= 0.1 (Methods 4.8).
* Fig. 4 panel order: dimensions by the mean over their indicators of the mean
  |SHAP| pooled over all buildings of all sites, indicators within a dimension
  by their own pooled mean |SHAP|.

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg (uID, label, 25 shap_ columns)
  outputs/main/01_features/signed_distance.parquet (products.signed_distance;
      computed from outputs/main/01_features/<site>_buildings_features.gpkg and
      data/raw/Limite_Favelas_2013/Limite_Favelas_2013.shp when it does not exist)

Outputs (outputs/main/04_boundary_profiles/)
  profiles_persite.csv     site x indicator x bin: number of buildings, mean SHAP
                           (empty where the bin has fewer than 20 buildings)
  profiles_crosssite.csv   indicator x bin: cross-site mean, s.d., number of sites
  transition_metrics.csv   indicator: panel order, pooled mean |SHAP|, dS, |dS|,
                           x0 (m), w (m), resolved (products.transition_metrics;
                           also read by appendix/S06_attribution_analyses)

Run (from the repository root)
  python main/04_boundary_profiles/01_boundary_profiles.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.boundary import (binned_mean, cross_site_profile, load_signed_distance, profile_edges,
                                 signed_distance, transition_metrics)
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, dimension_of, features_by_dimension, label_of
from favela_xai.io import read_favela_polygons, read_site_shap, write_table
from favela_xai.labels import label_to_int


def bin_counts(d, edges: np.ndarray) -> np.ndarray:
    """Number of buildings per bin, with the bin assignment of favela_xai.boundary.binned_mean."""
    d = np.asarray(d, dtype=float)
    m = np.isfinite(d) & (d >= edges[0]) & (d <= edges[-1])
    idx = np.clip(np.digitize(d[m], edges) - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1)


class SignedDistanceSource:
    """Signed distance of the buildings of the main/02 tables.

    The shared product of main/01_features is joined on the building id uID.
    If a main/02 table carries no uID, the distance is computed from its own
    geometry with the same function and settings.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._shared = None
        self._polygons = None

    def shared(self) -> pd.DataFrame:
        if self._shared is None:
            self._shared = load_signed_distance(self.cfg)
        return self._shared

    def polygons(self):
        if self._polygons is None:
            self._polygons = read_favela_polygons(self.cfg, 2013)
        return self._polygons


def site_table(cfg, site: str, sd: SignedDistanceSource) -> pd.DataFrame:
    """is_favela, signed_dist and the 25 SHAP columns of every building of one site."""
    path = cfg.require_product("shap_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    if "uID" in fields:
        df = pyogrio.read_dataframe(path, columns=["uID", "label"] + SHAP_COLUMNS, read_geometry=False)
        dist = sd.shared()
        dist = dist.loc[dist["site"] == site, ["uID", "is_favela", "signed_dist"]]
        df = df.merge(dist, on="uID", how="left", validate="one_to_one")
        if df["signed_dist"].isna().any():
            raise ValueError(f"{site}: {int(df['signed_dist'].isna().sum())} buildings have no signed distance; "
                             "re-run main/01_features")
        if not np.array_equal(label_to_int(df["label"]).astype(bool), df["is_favela"].astype(bool).to_numpy()):
            raise ValueError(f"{site}: labels of main/02 and of the signed-distance product differ")
    else:
        g = read_site_shap(cfg, site, columns=["label"] + SHAP_COLUMNS)
        fav = label_to_int(g["label"]).astype(bool)
        df = pd.DataFrame(g.drop(columns=g.geometry.name))
        df["is_favela"] = fav
        df["signed_dist"] = signed_distance(g, fav, sd.polygons(), cfg)
    df[SHAP_COLUMNS] = df[SHAP_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    bp = cfg.section("boundary_profiles")
    edges, centres = profile_edges(cfg)                  # 2.5 m bins on [-40, 40] m
    min_n = int(bp["min_buildings_per_site_bin"])
    sd = SignedDistanceSource(cfg)

    per_site = {f: [] for f in MODEL_FEATURE_ORDER}      # feature -> list of site profiles
    persite_rows = []
    pooled_abs_sum = pd.Series(0.0, index=MODEL_FEATURE_ORDER)   # sum of |SHAP| over all buildings
    pooled_n = pd.Series(0, index=MODEL_FEATURE_ORDER)            # number of non-missing values
    for site in cfg.sites:
        df = site_table(cfg, site, sd)
        d = df["signed_dist"].to_numpy(float)
        counts = bin_counts(d, edges)
        for f in MODEL_FEATURE_ORDER:
            prof = binned_mean(d, df["shap_" + f].to_numpy(float), edges, min_n)
            per_site[f].append(prof)
            for b in range(len(centres)):
                persite_rows.append(dict(site=site, feature=f, bin_lo=edges[b], bin_hi=edges[b + 1],
                                         bin_centre=centres[b], n_buildings=int(counts[b]), mean_shap=prof[b]))
        pooled_abs_sum += df[SHAP_COLUMNS].abs().sum().to_numpy()
        pooled_n += df[SHAP_COLUMNS].notna().sum().to_numpy()
        print(f"  {cfg.site_name(site):24s} {len(df):>7d} buildings, "
              f"{int(counts.sum()):>6d} within {edges[0]:g}..{edges[-1]:g} m")
    write_table(pd.DataFrame(persite_rows), out / "profiles_persite.csv")

    # Panel order of Fig. 4: dimensions by the mean of their indicators' pooled
    # mean |SHAP| (all buildings, all sites), indicators within a dimension by
    # their own pooled mean |SHAP|
    imp = pooled_abs_sum / pooled_n.clip(lower=1)
    groups = features_by_dimension()
    group_mean = {g: float(np.mean([imp[f] for f in fs])) for g, fs in groups.items()}
    group_order = sorted(groups, key=lambda g: -group_mean[g])
    order25 = [f for g in group_order for f in sorted(groups[g], key=lambda f: -imp[f])]

    cross_rows, metric_rows = [], []
    print(f"\n{'indicator':20s} {'x0 (m)':>7s} {'w (m)':>6s} {'|dS|':>6s}")
    for k, f in enumerate(order25):
        stack = np.vstack(per_site[f])
        mu, sdv = cross_site_profile(stack)
        n_sites = np.isfinite(stack).sum(axis=0)
        for b in range(len(centres)):
            cross_rows.append(dict(feature=f, label=label_of(f), dimension=dimension_of(f), bin_lo=edges[b],
                                   bin_hi=edges[b + 1], bin_centre=centres[b], mean_shap=mu[b], sd_shap=sdv[b],
                                   n_sites=int(n_sites[b])))
        m = transition_metrics(mu, centres, cfg)
        metric_rows.append(dict(panel_order=k + 1, feature=f, label=label_of(f), dimension=dimension_of(f),
                                pooled_mean_abs_shap=float(imp[f]), delta_s=m["delta_s"],
                                abs_delta_s=m["abs_delta_s"], x0=m["x0"], w=m["w"], resolved=m["resolved"]))
        print(f"{label_of(f):20s} {m['x0']:7.1f} {m['w']:6.1f} {m['abs_delta_s']:6.2f}")
    write_table(pd.DataFrame(cross_rows), out / "profiles_crosssite.csv")
    write_table(pd.DataFrame(metric_rows), cfg.product("transition_metrics", create_parent=True))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
