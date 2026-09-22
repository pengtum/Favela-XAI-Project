"""Signed distance to the official 2013 favela boundary, distance bins and profiles.

Manuscript items
* Main Methods 4.7 'Boundary-distance decomposition of misclassifications' and
  4.8 'Signed distance to the favela boundary and boundary profiles': the
  Euclidean distance (metres, EPSG:31983) from each building centroid to the
  nearest edge of the official 2013 favela polygon, negative for favela
  buildings and positive for non-favela buildings.
* 6 m error-rate bins and distance bands (Main Fig. 2c; ED Fig. 1; SI S5;
  SI Table S4).
* 2.5 m boundary profiles: per-site bin means of each indicator's Tree-SHAP
  value (at least 20 buildings per bin and site), cross-site mean and standard
  deviation, and the transition metrics |dS|, x0 and w (Main Fig. 4; SI
  Table S5; SI S13 and ED Fig. 5 use the same helpers).

The settings are in ``signed_distance``, ``boundary_errors`` and
``boundary_profiles`` of config.yaml.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


# -----------------------------------------------------------------------------
# Signed distance
# -----------------------------------------------------------------------------
def nearby_polygons(polygons: gpd.GeoDataFrame, bounds, pad_m: float) -> gpd.GeoDataFrame:
    """Polygons that intersect ``bounds`` (xmin, ymin, xmax, ymax) padded by ``pad_m``."""
    box = shapely.box(bounds[0] - pad_m, bounds[1] - pad_m, bounds[2] + pad_m, bounds[3] + pad_m)
    return polygons[polygons.intersects(box)]


def boundary_lines(polygons: gpd.GeoDataFrame, dissolve: bool):
    """Boundary geometry used for the distance (``signed_distance.dissolve_polygons``).

    * ``dissolve=False``: union of the boundaries of the individual polygons
      (an edge shared by two adjacent favela polygons counts as boundary);
    * ``dissolve=True``: boundary of the dissolved union (favela/non-favela
      interface only).
    """
    geoms = np.asarray(polygons.geometry.values, dtype=object)
    if dissolve:
        return shapely.union_all(geoms).boundary
    return shapely.union_all(shapely.boundary(geoms))


def signed_distance(buildings: gpd.GeoDataFrame, is_favela, polygons_2013: gpd.GeoDataFrame, cfg) -> np.ndarray:
    """Signed centroid distance to the 2013 favela boundary for the buildings of one site.

    Negative on the favela-labelled side, positive on the formal side
    (``signed_distance.sign_by: label``, Methods 4.8).
    """
    s = cfg.section("signed_distance")
    if s.get("sign_by", "label") != "label":
        raise ValueError("only sign_by: label is implemented (Methods 4.8)")
    polys = polygons_2013.to_crs(buildings.crs)
    near = nearby_polygons(polys, buildings.total_bounds, float(s["polygon_search_pad_m"]))
    line = boundary_lines(near, bool(s["dissolve_polygons"]))
    centroids = shapely.centroid(np.asarray(buildings.geometry.values, dtype=object))
    dist = shapely.distance(centroids, line)
    fav = np.asarray(is_favela, dtype=bool)
    return np.where(fav, -dist, dist)


def load_signed_distance(cfg, sites=None, recompute: bool = False) -> pd.DataFrame:
    """Signed distance of every building (columns site, uID, is_favela, signed_dist).

    Read from ``products.signed_distance`` when it exists; otherwise computed from
    the per-site building tables of main/01_features and the 2013 polygons, and
    written to that product for later steps.
    """
    from .io import read_favela_polygons, read_site_features, write_table
    from .labels import label_to_int

    sites = list(cfg.sites if sites is None else sites)
    path = cfg.product("signed_distance")
    stored = pd.read_parquet(path) if path.exists() else None
    if stored is not None and not recompute and set(sites) <= set(stored["site"].unique()):
        return stored[stored["site"].isin(sites)].reset_index(drop=True)

    polys = read_favela_polygons(cfg, 2013)
    parts = []
    for site in sites:
        g = read_site_features(cfg, site, columns=["uID", "label"])
        fav = label_to_int(g["label"]).astype(bool)
        parts.append(pd.DataFrame({"site": site, "uID": g["uID"].to_numpy(), "is_favela": fav,
                                   "signed_dist": signed_distance(g, fav, polys, cfg)}))
    new = pd.concat(parts, ignore_index=True)
    # keep the sites already in the product file that were not recomputed, in config site order
    if stored is not None:
        new = pd.concat([stored[~stored["site"].isin(sites)], new], ignore_index=True)
    order = {s: i for i, s in enumerate(cfg.all_sites)}
    new = new.sort_values("site", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    write_table(new, path)
    return new[new["site"].isin(sites)].reset_index(drop=True)


# -----------------------------------------------------------------------------
# 6 m error-rate bins and distance bands (Methods 4.7; Fig. 2c; ED Fig. 1)
# -----------------------------------------------------------------------------
def error_bins(width_m: float, half_range_m: float, tails: bool = True) -> list[tuple[float, float, str]]:
    """Signed-distance bins [lo, hi) of ``width_m`` between -half_range and +half_range.

    With ``tails=True`` buildings beyond the range are pooled into one end bin per
    side (ED Fig. 1, SI Table S4); without tails they are left out (Fig. 2c).
    """
    edges = np.arange(-half_range_m, half_range_m + width_m / 2, width_m)
    bins = [(float(a), float(b), f"[{a:g}, {b:g})") for a, b in zip(edges[:-1], edges[1:])]
    if tails:
        bins = [(-np.inf, float(edges[0]), f"< {edges[0]:g} m")] + bins + [(float(edges[-1]), np.inf, f">= {edges[-1]:g} m")]
    return bins


def bin_index(x, bins) -> np.ndarray:
    """Index of the bin of each value (-1 if outside every bin)."""
    x = np.asarray(x, dtype=float)
    idx = np.full(len(x), -1)
    for i, (lo, hi, _) in enumerate(bins):
        idx[(x >= lo) & (x < hi)] = i
    return idx


def band_index(abs_dist, band_edges_m) -> np.ndarray:
    """Distance band of |signed distance|: [e0, e1), [e1, e2), ..., [e_last, inf)."""
    edges = list(band_edges_m) + [np.inf]
    bins = [(float(a), float(b), "") for a, b in zip(edges[:-1], edges[1:])]
    return bin_index(np.abs(np.asarray(abs_dist, dtype=float)), bins)


# -----------------------------------------------------------------------------
# 2.5 m boundary profiles (Methods 4.8; Fig. 4; SI Table S5)
# -----------------------------------------------------------------------------
def profile_edges(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and centres of the boundary profiles (2.5 m bins on [-40, 40] m)."""
    p = cfg.section("boundary_profiles")
    half, width = float(p["half_range_m"]), float(p["bin_width_m"])
    n = int(round(2 * half / width))
    edges = np.linspace(-half, half, n + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def binned_mean(d, v, edges: np.ndarray, min_n: int) -> np.ndarray:
    """Mean of ``v`` in signed-distance bins; NaN where a bin has fewer than ``min_n`` values.

    Values with d exactly at the upper edge fall into the last bin.
    """
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    lo, hi = edges[0], edges[-1]
    m = np.isfinite(d) & np.isfinite(v) & (d >= lo) & (d <= hi)
    d, v = d[m], v[m]
    nb = len(edges) - 1
    idx = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    out = np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= min_n:
            out[b] = v[sel].mean()
    return out


def cross_site_profile(per_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-site mean and standard deviation (ddof 0) of per-site profiles (sites x bins)."""
    per_site = np.asarray(per_site, dtype=float)
    with np.errstate(invalid="ignore"), _quiet():
        return np.nanmean(per_site, axis=0), np.nanstd(per_site, axis=0)


def sign_flip(x: np.ndarray, y: np.ndarray) -> float:
    """First crossing from >= 0 to < 0 (inside to outside), by linear interpolation."""
    for i in range(len(x) - 1):
        if y[i] >= 0 and y[i + 1] < 0 and y[i] != y[i + 1]:
            return float(x[i] + (0 - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return np.nan


def smooth_profile(y: np.ndarray, sigma_bins) -> np.ndarray:
    """Gaussian smoothing of a profile over its finite bins (``sigma_bins`` in bins).

    ``sigma_bins`` of None or 0 returns the binned profile unchanged, so that x0
    and w can be taken on the unsmoothed profile
    (``boundary_profiles.smoothing_sigma_bins``).
    """
    y = np.asarray(y, dtype=float)
    if sigma_bins is None or float(sigma_bins) <= 0:
        return y.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma=float(sigma_bins))


def transition_metrics(mu: np.ndarray, centres: np.ndarray, cfg) -> dict:
    """|dS|, x0 and w of a cross-site mean profile (Methods 4.8).

    * dS = mean of the profile over bins centred at <= -25 m minus the mean over
      bins centred at >= +25 m (unsmoothed profile);
    * x0 = first sign change (favela to formal evidence) of the profile after
      Gaussian smoothing with ``boundary_profiles.smoothing_sigma_bins`` (1 bin;
      0 or null = the binned profile without smoothing);
    * w = sqrt(sum |S'| (x - xbar)^2 / sum |S'|) on the same profile, xbar the
      |S'|-weighted mean distance.
    x0 and w are flagged ``resolved`` only where |dS| >= 0.1.
    """
    p = cfg.section("boundary_profiles")
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(mu)
    res = dict(delta_s=np.nan, abs_delta_s=np.nan, x0=np.nan, w=np.nan, resolved=False)
    if ok.sum() < int(p["min_valid_bins"]):
        return res
    x = centres[ok]
    ys = smooth_profile(mu[ok], p.get("smoothing_sigma_bins"))
    g = np.abs(np.gradient(ys, x))
    if g.sum() > 0:
        xbar = np.sum(g * x) / np.sum(g)
        res["w"] = float(np.sqrt(np.sum(g * (x - xbar) ** 2) / np.sum(g)))
    deep = float(p["deep_band_m"])
    with _quiet():
        ds = float(np.nanmean(mu[centres <= -deep]) - np.nanmean(mu[centres >= deep]))
    res.update(delta_s=ds, abs_delta_s=abs(ds), x0=sign_flip(x, ys),
               resolved=bool(abs(ds) >= float(p["resolved_min_abs_delta_s"])))
    return res


class _quiet:
    """Silence 'mean of empty slice' warnings inside a block."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
