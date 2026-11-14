"""Small statistical helpers shared by several analyses.

Manuscript items
* Wilson 95% score intervals per distance bin (Main Methods 4.7; Main Fig. 2c;
  ED Fig. 1; SI Table S4).
* Site bootstrap: resampling the sites with replacement, 2,000 resamples
  (Results 2.1, ED Fig. 1, SI S5, SI S12 'Scoring'), in two forms: pooled
  rates (buildings pooled within each resample, ED Fig. 1 / SI Table S4) and
  the mean of site values (Results 2.1, SI S12, ED Fig. 3).
* Benjamini-Hochberg adjusted p values (SI S8, seven between-site measures).

Random draws use numpy ``default_rng(seed)`` and
``rng.integers(0, n_sites, size=(B, n_sites))``, with the sites in config order.
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(k, n, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval of k successes out of n (NaN where n = 0)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def site_bootstrap_draws(n_sites: int, resamples: int = 2000, seed: int = 42) -> np.ndarray:
    """Indices of resampled sites, shape (resamples, n_sites)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_sites, size=(resamples, n_sites))


def pooled_rate_bootstrap(n_by_site: np.ndarray, k_by_site: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Bootstrap of pooled rates: buildings pooled within each site resample.

    ``n_by_site``, ``k_by_site``: arrays (sites x cells) of counts and events.
    Returns rates of shape (resamples x cells); NaN where a resample has no building.
    """
    n_sites = n_by_site.shape[0]
    weights = np.stack([np.bincount(row, minlength=n_sites) for row in draws])
    n = weights @ n_by_site
    k = weights @ k_by_site
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, k / n, np.nan)


def bootstrap_mean_ci(site_values, resamples: int = 2000, seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean of site values over site resamples (NaN sites dropped first)."""
    v = np.asarray(site_values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    a = (1 - level) / 2 * 100
    return float(np.percentile(means, a)), float(np.percentile(means, 100 - a))


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (q values), in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0, 1)
    return q
