"""Step 4 of SI Section S13: Supplementary Figure S22 (raw-indicator and SHAP boundary profiles).

Manuscript item: SI Fig. S22.
* a-e: raw indicator (orange; (value - median)/IQR within each site) and its
  SHAP contribution (blue) against signed distance, each normalised so that the
  inner plateau (bins <= -25 m) is 1 and the outer plateau (bins >= +25 m) is 0;
  faint dashed orange where the site-bootstrap 95% interval of the raw plateau
  contrast includes 0 (off-scale parts hidden); open circles, half-way midpoints.
* f-j: un-normalised raw profiles (IQR units) with site-bootstrap 95% bands and
  the inner and outer plateau means (black bars, |d| >= 25 m).
The values quoted with SI Fig. S22 are in raw_profiles_metrics.csv and
raw_profiles_persite_summary.csv (03_raw_profiles.py).

The figure is drawn at 7.2 x 4.1 in, saved with a tight bounding box and scaled
to the double-column width (``appendix_S13_boundary_robustness.figure_width_mm``,
183 mm; needs PyMuPDF, see robustness_common.scale_pdf_to_width).

Inputs: raw_profiles.npz, raw_profiles_metrics.csv (03_raw_profiles.py).
Outputs (outputs/appendix/S13_boundary_robustness/)
* Figure_R1_raw_vs_shap.pdf (final, 183 mm wide) and
  Figure_R1_raw_vs_shap_unscaled.pdf (matplotlib page size).

Run from the repository root:
    python appendix/S13_boundary_robustness/04_figure_S22_raw_vs_shap.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import robustness_common as rc  # noqa: E402

# Figure style
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7, "axes.linewidth": 0.6,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False}
# Panel titles
TITLE = {"mean_interbuilding_distance": "Interbuilding distance", "adjacency": "Adjacency",
         "perimeter_wall": "Perimeter wall", "squareness": "Squareness", "avg_weighted_dist": "Avg. weighted dist."}
RAW_COLOUR, SHAP_COLOUR = "#D0641A", "#1F4E8C"
BOUNDARY_COLOUR = "#B64342"
YLO, YHI = -1.6, 2.6                       # y range of the normalised panels (a-e)
FIGSIZE_IN = (7.2, 4.1)


def main() -> None:
    plt.rcParams.update(RC)
    P = np.load(rc.OUT / "raw_profiles.npz")
    cen = P["cen"]
    M = pd.read_csv(rc.OUT / "raw_profiles_metrics.csv").set_index("feature")

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_IN, sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
    for j, f in enumerate(rc.KEY5):
        m = M.loc[f]
        # ---- top row: normalised profiles
        ax = axes[0, j]
        ax.axvline(0, color=BOUNDARY_COLOUR, lw=0.7, ls="--", zorder=1)
        for level in (0, 1):
            ax.axhline(level, color="#9A9A9A", lw=0.4, ls=":", zorder=1)
        ax.axhline(0.5, color="#C8C8C8", lw=0.4, zorder=1)
        out_of_range = False
        faint = not bool(m.raw_dS_ci_excludes_0)   # raw plateau contrast indistinguishable from 0
        for kind, colour, lw in (("shap", SHAP_COLOUR, 1.4), ("raw", RAW_COLOUR, 1.2)):
            mu = rc.normalise(P[f + "__" + kind])
            ok = np.isfinite(mu)
            if kind == "raw":
                if np.nanmax(mu) > YHI or np.nanmin(mu) < YLO:
                    out_of_range = True
                if out_of_range and faint:       # hide off-scale points instead of clipped vertical segments
                    mu = np.where((mu > YHI) | (mu < YLO), np.nan, mu)
                    ok = np.isfinite(mu)
            style = dict(alpha=0.45, lw=0.9, ls=(0, (3, 1.5))) if (kind == "raw" and faint) else dict(lw=lw)
            ax.plot(cen, np.where(ok, mu, np.nan), color=colour, zorder=3, clip_on=True, **style)
        for mid, colour in ((m.shap_midpoint, SHAP_COLOUR), (m.raw_midpoint, RAW_COLOUR)):
            if np.isfinite(mid):
                ax.plot(mid, 0.5, marker="o", ms=4, mfc="white", mec=colour, mew=1.1, zorder=5, clip_on=False)
        ax.set_ylim(YLO, YHI)
        ax.set_yticks([-1, 0, 0.5, 1])
        ax.set_title(TITLE[f], fontsize=7.5, pad=3)

        # ---- bottom row: raw robust-z profile with bootstrap band and plateau means
        bx = axes[1, j]
        bx.axvline(0, color=BOUNDARY_COLOUR, lw=0.7, ls="--", zorder=1)
        mu, band = P[f + "__raw"], P[f + "__raw_band"]
        ok = np.isfinite(mu)
        bx.fill_between(cen[ok], band[0][ok], band[1][ok], color=RAW_COLOUR, alpha=0.2, lw=0, zorder=2)
        bx.plot(cen[ok], mu[ok], color=RAW_COLOUR, lw=1.2, zorder=3)
        bx.hlines(m.raw_inner_plateau, -rc.HALF, -rc.DEEP, color="#333333", lw=1.4, zorder=4)
        bx.hlines(m.raw_outer_plateau, rc.DEEP, rc.HALF, color="#333333", lw=1.4, zorder=4)
        lo, hi = np.nanmin(band), np.nanmax(band)
        bx.set_ylim(lo - 0.08 * (hi - lo), hi + 0.62 * (hi - lo))
        ticks = MaxNLocator(nbins=4).tick_values(lo, hi)
        bx.set_yticks([t for t in ticks if lo - 0.08 * (hi - lo) <= t <= hi])

        for a in (ax, bx):
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.tick_params(length=2.5, pad=2, labelsize=6.3)
        bx.set_xlim(-rc.HALF, rc.HALF)
        bx.set_xticks([-40, -20, 0, 20, 40])
        if j > 0:
            ax.set_yticklabels([])
        # panel letters a-e (top) and f-j (bottom), kept out of the layout
        for a, letter in ((ax, "abcde"[j]), (bx, "fghij"[j])):
            t = a.text(-0.13, 1.0, letter, transform=a.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")
            t.set_in_layout(False)

    axes[0, 0].set_ylabel("Normalised profile\n(inner = 1, outer = 0)", fontsize=7)
    axes[1, 0].set_ylabel("Raw indicator\n(robust z, IQR units)", fontsize=7)
    fig.supxlabel("Signed distance to 2013 favela boundary (m; negative = inside)", fontsize=7, y=0.015)
    handles = [Line2D([0], [0], color=RAW_COLOUR, lw=1.2, label="Raw indicator (within-site median/IQR)"),
               Line2D([0], [0], color=SHAP_COLOUR, lw=1.4, label="SHAP contribution"),
               Line2D([0], [0], color=RAW_COLOUR, lw=0.9, alpha=0.45, ls=(0, (3, 1.5)), label="Raw, ΔS CI incl. 0"),
               Line2D([0], [0], marker="o", color="none", mfc="white", mec="#444444", ms=4, label="Half-way midpoint"),
               Line2D([0], [0], color="#333333", lw=1.4, label="Plateau (|d| ≥ 25 m)"),
               Patch(fc=RAW_COLOUR, alpha=0.2, label="Site bootstrap 95%")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=5.6, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.5, columnspacing=0.8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], w_pad=0.5, h_pad=0.6)

    unscaled = rc.OUT / "Figure_R1_raw_vs_shap_unscaled.pdf"
    fig.savefig(unscaled, bbox_inches="tight")
    print(f"saved {unscaled}")
    rc.scale_pdf_to_width(unscaled, rc.OUT / "Figure_R1_raw_vs_shap.pdf", float(rc.FS["figure_width_mm"]))


if __name__ == "__main__":
    main()
