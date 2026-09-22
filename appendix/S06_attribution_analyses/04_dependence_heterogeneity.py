"""SI Fig. S8: cross-site heterogeneity of the Tree-SHAP dependence.

For each indicator, the ten per-site binned-mean dependence curves together
with the pooled curve, and a heterogeneity score: the mean cross-site spread
of the binned means (SI Section S6, Results 2.2). The indicators with a score
above 2 are printed.

Method (config.yaml ``dependence.heterogeneity``)
* x range per indicator: 1st to 98th percentile of the pooled values; 15 equal
  bins; buildings outside the range are left out; a bin enters a curve when it
  holds at least 25 buildings;
* heterogeneity score: the standard deviation (ddof 0) of the per-site binned
  means in each bin, averaged over the bins;
* panel order as in SI Fig. S7 (attribution_common.dimension_panel_order);
  site colours: matplotlib tab10 in ``appendix_S06_attribution_analyses.site_order``.

Inputs
* products.shap_gpkg (main/02_classification): label, 25 indicator values and
  25 Tree-SHAP columns of the main per-site models.
Outputs (outputs/appendix/S06_attribution_analyses/)
* dependence_heterogeneity_scores.csv: score per indicator, with its range and
  number of scored bins;
* dependence_persite_curves.csv: per indicator, bin and site the binned mean;
* Figure_X_dependence_heterogeneity_scan.pdf (SI Fig. S8).

Run from the repository root:
    python appendix/S06_attribution_analyses/04_dependence_heterogeneity.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER, dimension_of, label_of
from favela_xai.io import write_table
from favela_xai.plotting_style import DIMENSION_COLOURS

from attribution_common import (dimension_panel_order, figure_style, save_pdf, shap_table, site_label,
                                site_order)

FIGURE = "Figure_X_dependence_heterogeneity_scan.pdf"
SCORES = "dependence_heterogeneity_scores.csv"
CURVES = "dependence_persite_curves.csv"


def binned_in_range(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int) -> np.ndarray:
    """Mean of y in bins of x inside [edges[0], edges[-1]]; NaN where a bin has fewer than min_n values."""
    m = np.isfinite(x) & np.isfinite(y) & (x >= edges[0]) & (x <= edges[-1])
    x, y = x[m], y[m]
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    return np.array([y[idx == b].mean() if (idx == b).sum() >= min_n else np.nan for b in range(len(edges) - 1)])


def curves(A: pd.DataFrame, f: str, sites: list[str], p: dict) -> dict:
    """Per-site and pooled binned means of one indicator and its heterogeneity score."""
    x = A[f].to_numpy(float)
    lo, hi = np.nanpercentile(x, p["percentile_range"])
    edges = np.linspace(lo, hi, int(p["n_bins"]) + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    min_n = int(p["min_buildings_per_bin"])
    per = {s: binned_in_range(A.loc[A["site"] == s, f].to_numpy(float),
                              A.loc[A["site"] == s, "shap_" + f].to_numpy(float), edges, min_n) for s in sites}
    pooled = binned_in_range(x, A["shap_" + f].to_numpy(float), edges, min_n)
    with warnings.catch_warnings():           # bins without any site value
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(np.vstack([per[s] for s in sites]), axis=0)   # ddof 0 across sites
        score = float(np.nanmean(spread))
    return dict(edges=edges, cen=cen, per=per, pooled=pooled, score=score,
                n_bins_scored=int(np.isfinite(spread).sum()), lo=float(lo), hi=float(hi))


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    p = cfg.section("dependence")["heterogeneity"]
    sites = site_order(cfg)
    colours = dict(zip(sites, plt.cm.tab10(np.linspace(0, 1, 10))))

    A = pd.concat([shap_table(cfg, s, indicators=True) for s in sites], ignore_index=True)
    res = {f: curves(A, f, sites, p) for f in MODEL_FEATURE_ORDER}
    order, gorder, _ = dimension_panel_order(A)

    # ---- tables ----------------------------------------------------------------
    scores = pd.DataFrame([dict(indicator=f, label=label_of(f), dimension=dimension_of(f),
                                heterogeneity_score=res[f]["score"], n_bins_scored=res[f]["n_bins_scored"],
                                x_lo=res[f]["lo"], x_hi=res[f]["hi"], panel=i + 1) for i, f in enumerate(order)])
    write_table(scores.sort_values("heterogeneity_score", ascending=False), out / SCORES)
    rows = []
    for f in order:
        r = res[f]
        for b, c in enumerate(r["cen"]):
            rows.append(dict(indicator=f, bin=b, bin_left=r["edges"][b], bin_right=r["edges"][b + 1], bin_centre=c,
                             site="pooled", mean_shap=r["pooled"][b]))
            rows += [dict(indicator=f, bin=b, bin_left=r["edges"][b], bin_right=r["edges"][b + 1], bin_centre=c,
                          site=s, mean_shap=r["per"][s][b]) for s in sites]
    write_table(pd.DataFrame(rows), out / CURVES)
    print("heterogeneity scores > 2:",
          [(label_of(f), round(res[f]["score"], 2)) for f in order if res[f]["score"] > 2])

    # ---- figure ----------------------------------------------------------------
    figure_style(11)
    fig, axes = plt.subplots(5, 5, figsize=(13.0, 13.4))
    axf = axes.ravel()
    for k, f in enumerate(order):
        ax, r = axf[k], res[f]
        gc = DIMENSION_COLOURS[dimension_of(f)]
        ax.axhline(0, color="#cccccc", lw=0.6, ls="--")
        for s in sites:
            yv = r["per"][s]
            ok = np.isfinite(yv)
            ax.plot(r["cen"][ok], yv[ok], color=colours[s], lw=1.0, alpha=0.85)
        ok = np.isfinite(r["pooled"])
        ax.plot(r["cen"][ok], r["pooled"][ok], color="black", lw=2.4, zorder=10)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(gc)
            sp.set_linewidth(1.7)
        ax.set_title(label_of(f), fontsize=11, color=gc, fontweight="bold", pad=3)
        ax.text(0.96, 0.06, f"heterogeneity {r['score']:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color="#555555")
        ax.tick_params(labelsize=8.5)
        ax.locator_params(axis="both", nbins=3)
        if k % 5 == 0:
            ax.set_ylabel("SHAP ($\\rightarrow$ fav.)", fontsize=9.5)
    for j in range(len(order), 25):
        axf[j].axis("off")
    gleg = [Patch(ec=DIMENSION_COLOURS[g], fc="none", lw=2.6, label=g) for g in gorder]
    fig.legend(handles=gleg, loc="upper center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.963),
               title="morphometric dimension (panel-frame colour); ordered by mean |SHAP|", title_fontsize=10.5)
    sleg = [Line2D([0], [0], color="black", lw=2.4, label="pooled, all sites")]
    sleg += [Line2D([0], [0], color=colours[s], lw=1.4, label=site_label(cfg, s, short=True)) for s in sites]
    fig.legend(handles=sleg, loc="lower center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_pdf(fig, out / FIGURE, dpi=220)
    plt.close(fig)
    print(f"[saved] {out / SCORES}\n[saved] {out / CURVES}\n[saved] {out / FIGURE}")


if __name__ == "__main__":
    main()
