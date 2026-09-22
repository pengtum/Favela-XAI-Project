"""SI Fig. S6: favela classification probability across the 2013 favela boundary.

Share of buildings that the main per-site model classifies favela, as a
function of the signed distance to the official 2013 favela boundary
(SI Section S6, Methods 4.8): pooled over all sites (line) and the per-site
minimum to maximum (band). Prints the pooled share deep inside and deep
outside the polygons.

Method (config.yaml ``boundary_profiles`` and
``appendix_S06_attribution_analyses.probability``)
* signed distance: centroid to the nearest edge of the official 2013 favela
  polygons, negative for favela buildings (products.signed_distance);
* 2.5 m bins between -40 and +40 m; bins with fewer than
  ``min_buildings_per_bin`` buildings are left out;
* ``probability.buildings``: all (every building of the site) | test
  (held-out test buildings of the main split only).

Inputs
* products.shap_gpkg (main/02_classification): label and predicted class;
* products.signed_distance (main/01_features).
Outputs (outputs/appendix/S06_attribution_analyses/)
* boundary_probability_profile.csv: per bin the pooled share and count and the
  share and count of every site;
* Figure_R1_boundary_probability.pdf (SI Fig. S6).

Run from the repository root:
    python appendix/S06_attribution_analyses/02_boundary_probability.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from favela_xai.boundary import binned_mean, profile_edges
from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_table

from attribution_common import add_signed_distance, figure_style, save_pdf, settings, shap_table, site_order

FIGURE = "Figure_R1_boundary_probability.pdf"
TABLE = "boundary_probability_profile.csv"


def bin_counts(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Number of buildings per bin, with the same bin rule as favela_xai.boundary.binned_mean."""
    d = d[np.isfinite(d) & (d >= edges[0]) & (d <= edges[-1])]
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    p = settings(cfg)["probability"]
    if p.get("band") != "min_max":
        raise ConfigError("only probability.band: min_max is implemented (SI Fig. S6 'per-site range')")
    if p.get("buildings") not in ("all", "test"):
        raise ConfigError("probability.buildings must be 'all' (every building) or 'test' (held-out test buildings)")
    min_n = int(p["min_buildings_per_bin"])
    edges, centres = profile_edges(cfg)
    half = float(cfg.get("boundary_profiles.half_range_m"))
    sites = site_order(cfg)

    # ---- per-site and pooled share classified favela ---------------------------
    tables = []
    for site in sites:
        t = add_signed_distance(cfg, shap_table(cfg, site, prediction=True))
        if p["buildings"] == "test":
            if "is_test" not in t.columns:
                raise ConfigError(f"{site}: the main/02 table has no is_test column")
            t = t[t["is_test"]]
        tables.append(t[["site", "signed_dist", "pred_favela"]])
    allb = pd.concat(tables, ignore_index=True)

    prof = pd.DataFrame({"bin_left_m": edges[:-1], "bin_right_m": edges[1:], "bin_centre_m": centres})
    per_site = []
    for site in sites:
        t = allb[allb["site"] == site]
        share = binned_mean(t["signed_dist"].to_numpy(), t["pred_favela"].to_numpy(), edges, min_n)
        prof[f"share_{site}"] = share
        prof[f"n_{site}"] = bin_counts(t["signed_dist"].to_numpy(), edges)
        per_site.append(share)
    M = np.vstack(per_site)
    with warnings.catch_warnings():            # bins without any site value give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
    pooled = binned_mean(allb["signed_dist"].to_numpy(), allb["pred_favela"].to_numpy(), edges, min_n)
    prof.insert(3, "pooled_share", pooled)
    prof.insert(4, "pooled_n", bin_counts(allb["signed_dist"].to_numpy(), edges))
    prof.insert(5, "site_min", lo)
    prof.insert(6, "site_max", hi)
    write_table(prof, out / TABLE)

    # numbers quoted in SI Section S6
    inside = np.nanmean(pooled[centres <= -float(cfg.get("boundary_profiles.deep_band_m"))])
    outside = np.nanmean(pooled[centres >= float(cfg.get("boundary_profiles.deep_band_m"))])
    print(f"pooled share classified favela: {inside:.2f} deep inside (bins <= -25 m), "
          f"{outside:.2f} deep outside (bins >= +25 m)")

    # ---- figure (SI Fig. S6) ---------------------------------------------------
    figure_style(8)
    ok = np.isfinite(lo) & np.isfinite(hi)
    pk = np.isfinite(pooled)
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.axvspan(-half, 0, color="#E28E2C", alpha=0.06)
    ax.axvspan(0, half, color="#6BAED6", alpha=0.06)
    ax.fill_between(centres[ok], lo[ok], hi[ok], color="#444444", alpha=0.13, lw=0,
                    label=f"per-site range ({len(sites)} sites)")
    ax.plot(centres[pk], pooled[pk], color="#222222", lw=2.4, label="pooled")
    ax.axvline(0, color="#B64342", lw=1.0, ls="--")
    ax.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":")
    ax.text(-half + 2, 0.94, "inside favela", fontsize=7.5, color="#9c5a16", va="top")
    ax.text(half - 2, 0.94, "outside (formal)", fontsize=7.5, color="#2f5d80", va="top", ha="right")
    ax.set_xlim(-half, half)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("signed distance to boundary  (m;  $<0$ inside,  $>0$ outside)", fontsize=8)
    ax.set_ylabel("P(building classified favela)", fontsize=8)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    save_pdf(fig, out / FIGURE)
    plt.close(fig)
    print(f"[saved] {out / TABLE}\n[saved] {out / FIGURE}")


if __name__ == "__main__":
    main()
