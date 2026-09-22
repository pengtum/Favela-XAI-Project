"""Shared helpers of appendix/S13_boundary_robustness (Supplementary Section S13).

Imported by the numbered scripts of this folder; not run on its own. It holds:

* the boundary-profile conventions of main-text Figure 4 and SI Table S5
  (2.5 m signed-distance bins on [-40, 40] m, at least 20 buildings per bin and
  site, pooled profile = mean of the per-site profiles, Gaussian smoothing of
  one bin, plateau contrast dS, sign-change distance x0 and width w), taken from
  ``favela_xai.boundary``;
* two further profile summaries: the maximum slope of the smoothed profile
  within +/-20 m (ED Fig. 5) and the half-way midpoint (SI Fig. S22);
* the robust standardisation of raw indicators, (value - median) / IQR within a
  site (ED Fig. 5e; SI Fig. S22), and the within-class gradient G (ED Fig. 5e);
* signed distances to the 2013 favela polygons and to moved copies of them
  (shifted boundaries, ED Fig. 5; polygon shifts, SI Table S17);
* the per-site cache written by 01_prepare_inputs.py and the inputs of the
  line tests (05_shifted_boundaries.py, 06_polygon_shifts.py);
* scaling of a figure PDF to the double-column width (183 mm).

Settings (config.yaml): ``boundary_profiles`` (profile conventions),
``signed_distance`` (boundary definition), ``boundary_robustness`` (SI S13) and
``appendix_S13_boundary_robustness`` (settings specific to this folder).
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from favela_xai.boundary import binned_mean, profile_edges, smooth_profile, transition_metrics
from favela_xai.config import ConfigError, load_config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS

# -----------------------------------------------------------------------------
# Configuration and folders
# -----------------------------------------------------------------------------
cfg = load_config()
OUT = cfg.output_dir(__file__)          # outputs/appendix/S13_boundary_robustness/
CACHE = OUT / "cache"                   # per-site inputs written by 01_prepare_inputs.py

BP = cfg.section("boundary_profiles")   # Main Methods 4.8; Main Fig. 4; SI Table S5
BR = cfg.section("boundary_robustness") # SI S13; ED Fig. 5; SI Fig. S22; SI Table S17
FS = cfg.section("appendix_S13_boundary_robustness")   # settings specific to this folder

# The five leading indicators of main-text Figure 4 / SI Table S5 (rows of SI
# Table S17, panels of SI Fig. S22, x-axis of ED Fig. 5e).
KEY5: list[str] = list(FS["leading_indicators"])

EDGES, CEN = profile_edges(cfg)         # 2.5 m bins on [-40, 40] m
HALF = float(BP["half_range_m"])
MIN_N = int(BP["min_buildings_per_site_bin"])
SIGMA = BP.get("smoothing_sigma_bins")   # None or 0 = no smoothing
DEEP = float(BP["deep_band_m"])
MIN_BINS = int(BP["min_valid_bins"])

INDICATORS = list(MODEL_FEATURE_ORDER)
SHAP = list(SHAP_COLUMNS)

# Quantities profiled against the real and the moved lines (ED Fig. 5):
# SHAP of the five leading indicators, the label step (label - 0.5) and the
# oriented raw indicators.
QUANTITIES = ["shap_" + f for f in KEY5] + ["label"] + ["raw_" + f for f in KEY5]


def quantity_parts(q: str) -> tuple[str, str]:
    """('shap' | 'raw' | 'label', indicator) of a quantity name."""
    if q == "label":
        return "label", "label"
    kind, ind = q.split("_", 1)
    return kind, ind


# -----------------------------------------------------------------------------
# Boundary profiles (conventions of Main Fig. 4 / SI Table S5)
# -----------------------------------------------------------------------------
def site_profile(d, v) -> np.ndarray:
    """Per-site profile: mean of v in the 2.5 m bins (NaN below 20 buildings)."""
    return binned_mean(d, v, EDGES, MIN_N)


def pooled_profile(per_site: np.ndarray) -> np.ndarray:
    """Pooled profile = mean of the per-site profiles (sites x bins), ignoring empty bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(per_site, dtype=float), axis=0)


def smoothed(mu: np.ndarray):
    """Finite bins of a profile and their Gaussian-smoothed values.

    Smoothing as in Main Fig. 4 / SI Table S5 (``boundary_profiles.smoothing_sigma_bins``,
    1 bin; 0 or null = no smoothing). Returns (None, None) when fewer than
    ``min_valid_bins`` bins are finite.
    """
    ok = np.isfinite(mu)
    if ok.sum() < MIN_BINS:
        return None, None
    return CEN[ok], smooth_profile(mu[ok], SIGMA)


def plateaus(mu: np.ndarray) -> tuple[float, float]:
    """Inner plateau (mean of bins centred at <= -25 m) and outer plateau (>= +25 m)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(mu[CEN <= -DEEP])), float(np.nanmean(mu[CEN >= DEEP]))


def profile_metrics(mu: np.ndarray) -> dict:
    """x0, w and dS of a profile (favela_xai.boundary.transition_metrics, as in Main
    Fig. 4 / SI Table S5), plus the maximum |slope| of the smoothed profile within
    +/- max_slope_window_m (20 m; ED Fig. 5)."""
    m = transition_metrics(mu, CEN, cfg)
    x, ys = smoothed(mu)
    maxslope = np.nan
    if x is not None:
        slope = np.abs(np.gradient(ys, x))
        inner = np.abs(x) <= float(BR["max_slope_window_m"])
        if inner.any():
            maxslope = float(slope[inner].max())
    return dict(x0=m["x0"], w=m["w"], dS=m["delta_s"], maxslope=maxslope,
                n_bins=int(np.isfinite(mu).sum()))


def normalise(mu: np.ndarray) -> np.ndarray:
    """Profile rescaled so that the outer plateau is 0 and the inner plateau is 1."""
    inner, outer = plateaus(mu)
    return (mu - outer) / (inner - outer)


def midpoint(mu: np.ndarray) -> tuple[float, int]:
    """Half-way midpoint of a profile (SI Fig. S22, open circles).

    The smoothed profile's first crossing, from the inner to the outer side, of
    the level half-way between the two plateaus. Returns (midpoint, number of
    crossings of that level).
    """
    x, ys = smoothed(mu)
    if x is None:
        return np.nan, 0
    inner, outer = plateaus(mu)
    if not (np.isfinite(inner) and np.isfinite(outer)) or inner == outer:
        return np.nan, 0
    level = 0.5 * (inner + outer)
    z = (ys - level) * np.sign(inner - outer)      # > 0 on the inner side of the level
    first, n_cross = np.nan, 0
    for i in range(len(x) - 1):
        crossing = (z[i] >= 0 > z[i + 1]) or (z[i] < 0 <= z[i + 1])
        if crossing and z[i] != z[i + 1]:
            n_cross += 1
            if np.isnan(first) and z[i] >= 0 > z[i + 1]:
                first = float(x[i] + (0 - z[i]) / (z[i + 1] - z[i]) * (x[i + 1] - x[i]))
    return first, n_cross


# -----------------------------------------------------------------------------
# Raw indicators and within-class gradient (ED Fig. 5e; SI Fig. S22)
# -----------------------------------------------------------------------------
def robust_z(v) -> tuple[np.ndarray, bool]:
    """(value - median) / IQR within a site.

    If the interquartile range is zero, 1.4826 x the median absolute deviation
    (or, failing that, the standard deviation) is used instead; the second
    return value flags that fallback.
    """
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    q1, q3 = np.nanpercentile(v, [25, 75])
    scale, fallback = q3 - q1, False
    if not scale > 0:
        mad = 1.4826 * np.nanmedian(np.abs(v - med))
        scale, fallback = (mad if mad > 0 else np.nanstd(v)), True
    return (v - med) / scale, fallback


def within_class_gradient(sd, v, fav) -> tuple[float, float]:
    """Within-class gradient of one site (ED Fig. 5e; SI S13 'Shifted boundaries').

    G_fav = mean over favela-labelled buildings at 25-40 m minus the mean over
            favela-labelled buildings within 10 m of the line;
    G_non = mean over formal-settlement buildings within 10 m minus the mean at
            25-40 m.
    Distances are |signed distance|; a band enters only with at least
    ``gradient_min_buildings_per_band`` buildings (NaN otherwise). G > 0: values
    move toward the other class as the line is approached; the label step gives
    G = 0 by construction.
    """
    near_m = float(BR["gradient_near_m"])
    far_lo, far_hi = (float(a) for a in BR["gradient_far_m"])
    min_n = int(FS["gradient_min_buildings_per_band"])

    def band_mean(sel):
        return float(v[sel].mean()) if sel.sum() >= min_n else np.nan

    a = np.abs(np.asarray(sd, dtype=float))
    v = np.asarray(v, dtype=float)
    fav = np.asarray(fav, dtype=bool)
    ok = np.isfinite(a) & np.isfinite(v)
    near = ok & (a <= near_m)
    far = ok & (a >= far_lo) & (a <= far_hi)
    return (band_mean(far & fav) - band_mean(near & fav),
            band_mean(near & ~fav) - band_mean(far & ~fav))


# -----------------------------------------------------------------------------
# Distances to the 2013 polygons and to moved copies of them
# -----------------------------------------------------------------------------
def boundary_segments(geoms) -> np.ndarray:
    """All straight boundary segments of an array of (multi)polygons or (multi)lines."""
    geoms = np.asarray(geoms, dtype=object)
    lines = shapely.boundary(geoms) if np.any(shapely.get_type_id(geoms) >= 3) else geoms
    parts = shapely.get_parts(lines)
    while np.any(shapely.get_type_id(parts) >= 4):          # flatten multi-geometries and collections
        parts = shapely.get_parts(parts)
    parts = parts[~shapely.is_empty(parts)]
    xy, part = shapely.get_coordinates(parts, return_index=True)
    same = part[:-1] == part[1:]
    return shapely.linestrings(np.stack([xy[:-1][same], xy[1:][same]], axis=1))


def distance_to_segments(segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact distance of each point to the nearest segment."""
    tree = STRtree(segments)
    idx, dist = tree.query_nearest(points, return_distance=True, all_matches=False)
    out = np.full(len(points), np.nan)
    out[idx[0]] = dist
    return out


def polygon_side_distance(polygons: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed distance of the points (x, y) to a set of polygons, negative when the
    point lies inside one of them.

    This is the "polygon side" convention of ED Fig. 5 (side set by polygon
    membership, as for the shifted boundaries). The line is the boundary of the
    individual polygons, or of their union when
    ``signed_distance.dissolve_polygons`` is true (the same choice as for the
    Methods 4.8 distance).
    """
    polygons = np.asarray(polygons, dtype=object)
    union = shapely.union_all(polygons)
    dissolve = bool(cfg.get("signed_distance.dissolve_polygons"))
    segments = boundary_segments(np.array([union], dtype=object)) if dissolve else boundary_segments(polygons)
    d = distance_to_segments(segments, shapely.points(x, y))
    shapely.prepare(union)
    inside = shapely.contains_xy(union, x, y)
    return np.where(inside, -d, d)


def label_side(signed_dist: np.ndarray, is_favela: np.ndarray) -> np.ndarray:
    """Same distance with the side set by the building's label (Methods 4.8 convention)."""
    a = np.abs(signed_dist)
    return np.where(is_favela, -a, a)


def translate(polygons: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Rigid translation of all polygons by (dx, dy)."""
    return shapely.transform(polygons, lambda c: c + np.array([dx, dy]))


def translate_each(polygons: np.ndarray, dxs, dys) -> np.ndarray:
    """Translation of each polygon by its own vector."""
    return np.array([shapely.transform(p, lambda c, a=a, b=b: c + np.array([a, b]))
                     for p, a, b in zip(polygons, dxs, dys)], dtype=object)


# -----------------------------------------------------------------------------
# Per-site cache (01_prepare_inputs.py)
# -----------------------------------------------------------------------------
def buildings_path(site: str) -> Path:
    return CACHE / f"{site}_buildings.parquet"


def polygons_path(site: str) -> Path:
    return CACHE / f"{site}_polygons.parquet"


def read_buildings(site: str, columns=None) -> pd.DataFrame:
    """Per-site building table of 01_prepare_inputs.py (one row per building)."""
    path = buildings_path(site)
    if not path.exists():
        raise ConfigError(f"{path} does not exist; run 01_prepare_inputs.py of this folder first.")
    return pd.read_parquet(path, columns=columns)


def read_polygons(site: str) -> np.ndarray:
    """Repaired 2013 favela polygons around a site (array of shapely geometries,
    in the order of the polygon layer)."""
    import geopandas as gpd

    path = polygons_path(site)
    if not path.exists():
        raise ConfigError(f"{path} does not exist; run 01_prepare_inputs.py of this folder first.")
    return np.asarray(gpd.read_parquet(path).geometry.values, dtype=object)


def bootstrap_site_order(sites) -> list[str]:
    """Order of the sites in the site-bootstrap draws of the raw-profile intervals
    (``appendix_S13_boundary_robustness.raw_bootstrap_site_order``; sites not
    listed follow in config order)."""
    listed = [s for s in FS["raw_bootstrap_site_order"] if s in sites]
    return listed + [s for s in sites if s not in listed]


# -----------------------------------------------------------------------------
# Inputs of the line tests (05_shifted_boundaries.py, 06_polygon_shifts.py)
# -----------------------------------------------------------------------------
@dataclass
class LineInputs:
    """Everything the line tests hold fixed: per site the centroids, the labels,
    the nearby 2013 polygons and the profiled quantities."""
    sites: list
    x: dict
    y: dict
    fav: dict
    polygons: dict
    values: dict          # site -> {quantity: array}
    orientation: dict     # indicator -> +1 / -1 (raw orientation)


def line_inputs(sites) -> LineInputs:
    """Load the cache of 01_prepare_inputs.py for the line tests.

    Raw indicators are robust-standardised within each site and oriented so that
    positive values are favela-like (ED Fig. 5e). The orientation of an
    indicator is the sign of the mean over sites of (median favela minus median
    formal-settlement standardised value).
    """
    cols = ["x", "y", "y_true"] + KEY5 + ["shap_" + f for f in KEY5]
    data = {s: read_buildings(s, columns=cols) for s in sites}
    fav = {s: data[s]["y_true"].to_numpy() == 1 for s in sites}
    z = {s: {f: robust_z(data[s][f].to_numpy(float))[0] for f in KEY5} for s in sites}

    orientation = {}
    for f in KEY5:
        diffs = [np.nanmedian(z[s][f][fav[s]]) - np.nanmedian(z[s][f][~fav[s]]) for s in sites]
        orientation[f] = 1.0 if np.nanmean(diffs) >= 0 else -1.0

    values = {}
    for s in sites:
        v = {"shap_" + f: data[s]["shap_" + f].to_numpy(float) for f in KEY5}
        v["label"] = data[s]["y_true"].to_numpy(float) - 0.5
        for f in KEY5:
            v["raw_" + f] = orientation[f] * z[s][f]
        values[s] = v
    return LineInputs(sites=list(sites),
                      x={s: data[s]["x"].to_numpy(float) for s in sites},
                      y={s: data[s]["y"].to_numpy(float) for s in sites},
                      fav=fav, polygons={s: read_polygons(s) for s in sites},
                      values=values, orientation=orientation)


def line_profiles(sd_by_site: dict, li: LineInputs, quantities) -> tuple[dict, dict]:
    """Per-site profiles (sites x bins) and pooled profiles of each quantity against one line."""
    per_site = {q: np.vstack([site_profile(sd_by_site[s], li.values[s][q]) for s in li.sites]) for q in quantities}
    return per_site, {q: pooled_profile(per_site[q]) for q in quantities}


def line_gradients(sd_by_site: dict, li: LineInputs, quantities) -> dict:
    """Within-class gradients (sites x [G_fav, G_non]) of each quantity against one line."""
    return {q: np.array([within_class_gradient(sd_by_site[s], li.values[s][q], li.fav[s]) for s in li.sites])
            for q in quantities}


def real_line(li: LineInputs) -> tuple[dict, dict]:
    """Signed distances to the real 2013 line: (polygon side, label side) per site."""
    geom = {s: polygon_side_distance(li.polygons[s], li.x[s], li.y[s]) for s in li.sites}
    label = {s: label_side(geom[s], li.fav[s]) for s in li.sites}
    return geom, label


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def scale_pdf_to_width(src: Path, dst: Path, width_mm: float) -> bool:
    """Scale a one-page PDF to ``width_mm`` (vector, fonts kept) with PyMuPDF.

    The figures are saved with a tight bounding box and then scaled to the
    double-column width in this way. PyMuPDF (``pymupdf``, listed in
    environment.yml) is imported here only; without it the unscaled PDF is
    copied and False is returned.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        shutil.copyfile(src, dst)
        print(f"PyMuPDF not installed: {dst.name} keeps the matplotlib page size "
              f"(install pymupdf to scale it to {width_mm:g} mm).")
        return False
    points_per_mm = 72 / 25.4
    source = fitz.open(src)
    page = source[0]
    width = width_mm * points_per_mm
    height = page.rect.height * width / page.rect.width
    out = fitz.open()
    new_page = out.new_page(width=width, height=height)
    new_page.show_pdf_page(new_page.rect, source, 0)
    out.save(dst, garbage=3, deflate=True)
    out.close()
    source.close()
    print(f"saved {dst} ({width_mm:.1f} x {height / points_per_mm:.1f} mm)")
    return True
