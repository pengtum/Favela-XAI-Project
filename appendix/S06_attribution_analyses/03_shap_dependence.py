"""SI Fig. S7: pooled Tree-SHAP dependence of the 25 indicators.

For each indicator, the per-building Tree-SHAP contribution against the
indicator value, all buildings of the ten sites pooled (favela and non-favela
points), with the pooled binned-mean trend and a band spanning the per-site
binned-mean trends (SI Section S6, Results 2.2).

Method (config.yaml ``dependence.pooled``)
* x range per indicator: 1st to 99th percentile of the pooled values; 12 equal
  bins; a bin enters a trend when it holds at least 20 buildings;
  ``appendix_S06_attribution_analyses.dependence_tails_in_end_bins``: true
  (values outside the range counted in the end bins) | false (left out); the
  scatter shows values inside the range only;
* band: minimum to maximum of the ten per-site binned means;
* scatter: up to 1,200 randomly drawn buildings per class and panel (seed 42);
* panels grouped by morphometric dimension and ordered by mean |SHAP| within
  and across groups (attribution_common.dimension_panel_order; same rule as
  Fig. 4).

Inputs
* products.shap_gpkg (main/02_classification): label, 25 indicator values and
  25 Tree-SHAP columns of the main per-site models.
Outputs (outputs/appendix/S06_attribution_analyses/)
* dependence_pooled_trends.csv: per indicator and bin the pooled trend and the
  per-site minimum and maximum;
* Figure_X_shap_dependence.pdf (SI Fig. S7).

Run from the repository root:
    python appendix/S06_attribution_analyses/03_shap_dependence.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from favela_xai.config import load_config
from favela_xai.features import dimension_of, label_of
from favela_xai.io import write_table
from favela_xai.plotting_style import DIMENSION_COLOURS, FAVELA_SHAP, FORMAL_SHAP

from attribution_common import dimension_panel_order, figure_style, save_pdf, settings, shap_table, site_order

FIGURE = "Figure_X_shap_dependence.pdf"
TABLE = "dependence_pooled_trends.csv"
TREND = "#14746A"      # pooled trend and per-site band
REF = "#555555"        # SHAP = 0 reference line


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int, tails_in_end_bins: bool) -> np.ndarray:
    """Mean of y in bins of x; NaN where a bin holds fewer than min_n buildings.

    With ``tails_in_end_bins`` values below the first or above the last edge are
    counted in the first or last bin; otherwise they are left out.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if not tails_in_end_bins:
        m &= (x >= edges[0]) & (x <= edges[-1])
    x, y = x[m], y[m]
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_n:
            out[b] = y[sel].mean()
    return out


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    p = cfg.section("dependence")["pooled"]
    tails = bool(settings(cfg)["dependence_tails_in_end_bins"])
    sites = site_order(cfg)

    A = pd.concat([shap_table(cfg, s, indicators=True) for s in sites], ignore_index=True)
    order, gorder, _ = dimension_panel_order(A)
    print("panel order:", [label_of(f) for f in order])

    figure_style(11)
    rng = np.random.default_rng(int(p["seed"]))
    nr, nc = 5, 5
    fig, axes = plt.subplots(nr, nc, figsize=(12.0, 12.6))
    axf = axes.ravel()
    rows = []
    for k, f in enumerate(order):
        ax = axf[k]
        sh = "shap_" + f
        x, y = A[f].to_numpy(float), A[sh].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        xlo, xhi = np.percentile(x[m], p["percentile_range"])
        if xhi <= xlo:
            xhi = xlo + 1e-9
        edges = np.linspace(xlo, xhi, int(p["n_bins"]) + 1)
        cen = 0.5 * (edges[:-1] + edges[1:])

        # class-coloured scatter (subsampled, rasterised)
        for cls, col in [(False, FORMAL_SHAP), (True, FAVELA_SHAP)]:
            sub = A[A["is_favela"] == cls]
            xs, ys = sub[f].to_numpy(float), sub[sh].to_numpy(float)
            mm = np.isfinite(xs) & np.isfinite(ys) & (xs >= xlo) & (xs <= xhi)
            xs, ys = xs[mm], ys[mm]
            if len(xs) > int(p["points_per_class"]):
                sel = rng.choice(len(xs), int(p["points_per_class"]), replace=False)
                xs, ys = xs[sel], ys[sel]
            ax.scatter(xs, ys, s=1.4, c=col, alpha=0.14, lw=0, rasterized=True)

        # per-site band (min-max of the per-site trends) and pooled trend
        M = np.vstack([binned(A.loc[A["site"] == s, f].to_numpy(float), A.loc[A["site"] == s, sh].to_numpy(float),
                              edges, int(p["min_buildings_per_bin"]), tails) for s in sites])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(cen[ok], lo[ok], hi[ok], color=TREND, alpha=0.15, lw=0)
        pooled = binned(x[m], y[m], edges, int(p["min_buildings_per_bin"]), tails)
        pk = np.isfinite(pooled)
        ax.plot(cen[pk], pooled[pk], color=TREND, lw=1.8, solid_capstyle="round")
        ax.axhline(0, color=REF, lw=0.6, ls="--")
        ax.set_xlim(xlo, xhi)
        gc = DIMENSION_COLOURS[dimension_of(f)]
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(gc)
            sp.set_linewidth(1.7)
        ax.set_title(label_of(f), fontsize=11.5, fontweight="bold", pad=3, color=gc)
        ax.tick_params(labelsize=9, color="#999999")
        ax.locator_params(axis="both", nbins=3)
        if k % nc == 0:
            ax.set_ylabel("SHAP ($\\rightarrow$ fav.)", fontsize=10.5)
        for b in range(len(cen)):
            rows.append(dict(indicator=f, label=label_of(f), dimension=dimension_of(f), bin=b,
                             bin_left=edges[b], bin_right=edges[b + 1], bin_centre=cen[b],
                             pooled_mean_shap=pooled[b], site_min=lo[b], site_max=hi[b]))
    for j in range(len(order), nr * nc):
        axf[j].axis("off")

    handles = [Patch(fc=FAVELA_SHAP, label="favela building"), Patch(fc=FORMAL_SHAP, label="non-favela building"),
               Line2D([0], [0], color=TREND, lw=1.8, label="pooled trend"),
               Patch(fc=TREND, alpha=0.15, label=f"per-site range ({len(sites)} sites)"),
               Line2D([0], [0], color=REF, lw=0.6, ls="--", label="SHAP = 0  (formal / favela)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12.5, bbox_to_anchor=(0.5, -0.004))
    gleg = [Patch(ec=DIMENSION_COLOURS[g], fc="none", lw=3.0, label=g) for g in gorder]
    fig.legend(handles=gleg, loc="upper center", ncol=5, fontsize=12.5, bbox_to_anchor=(0.5, 0.965),
               title="morphometric dimension (panel-frame colour) — groups & within-group features "
                     "ordered by mean |SHAP|", title_fontsize=11.5)
    fig.suptitle("Deep-feature SHAP dependence for all 25 morphological indicators (x = feature value, y = SHAP, "
                 "positive $\\rightarrow$ favela; all ten sites)", fontsize=13.5, y=1.0)
    fig.tight_layout(rect=[0, 0.028, 1, 0.95])
    save_pdf(fig, out / FIGURE, dpi=350)
    plt.close(fig)
    write_table(pd.DataFrame(rows), out / TABLE)
    print(f"[saved] {out / TABLE}\n[saved] {out / FIGURE}")


if __name__ == "__main__":
    main()
