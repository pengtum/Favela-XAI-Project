"""Step 7 of SI Section S13: Extended Data Fig. 5 (shifted boundaries and label controls).

Manuscript item: Extended Data Fig. 5.
* a, b: mean SHAP value of mean interbuilding distance and adjacency against
  signed distance; red, real boundary with the polygon side; dashed blue, real
  boundary with the label side; grey lines and band, the 20 shifted boundaries
  (range).
* c: the same for the label step (label - 0.5; no SHAP).
* d: profiles on the real boundary with each building on its labelled side,
  normalised so that the outer plateau is 0 and the inner plateau is 1 (label
  step, mean interbuilding distance, adjacency).
* e: within-class gradient G of the five leading indicators on the real
  boundary (red triangles, SHAP; open blue, oriented raw indicator) and on the
  20 shifted boundaries (grey, SHAP); triangles up, favela-labelled buildings;
  triangles down, formal-settlement buildings.
The values quoted with ED Fig. 5 are printed by 05_shifted_boundaries.py.

The figure is drawn at 7.2 x 5.3 in, saved with a tight bounding box and scaled
to the double-column width (``appendix_S13_boundary_robustness.figure_width_mm``,
183 mm; needs PyMuPDF, see robustness_common.scale_pdf_to_width).

Inputs: profiles_translation.npz, placebo_translation.csv (05_shifted_boundaries.py).
Outputs (outputs/appendix/S13_boundary_robustness/)
* Figure_R1_placebo_boundaries.pdf (final, 183 mm wide) and
  Figure_R1_placebo_boundaries_unscaled.pdf (matplotlib page size).

Run from the repository root:
    python appendix/S13_boundary_robustness/07_figure_ED5_shifted_boundaries.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import robustness_common as rc  # noqa: E402

# Figure style
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7.0, "axes.linewidth": 0.6,
      "legend.frameon": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
PRETTY = dict(mean_interbuilding_distance="Interbuilding distance", adjacency="Adjacency",
              perimeter_wall="Perimeter wall", squareness="Squareness", avg_weighted_dist="Avg. weighted dist.",
              label="Label step")
RED, GREY, BLUE, BLACK, ORANGE = "#B64342", "#8C8C8C", "#1F5A89", "#222222", "#E2700E"
FIGSIZE_IN = (7.2, 5.3)
DOT_JITTER_SEED = 0          # horizontal jitter of the grey dots in panel e (layout only)


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    plt.rcParams.update(RC)
    Z = np.load(rc.OUT / "profiles_translation.npz")
    PT = pd.read_csv(rc.OUT / "placebo_translation.csv")
    cen = Z["cen"]
    a_ind, b_ind = rc.FS["shifted_boundary_panels"]          # ED Fig. 5 a, b

    fig = plt.figure(figsize=FIGSIZE_IN)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42, left=0.07, right=0.99, top=0.90, bottom=0.08)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:])

    # ---- a-c: real line vs the 20 shifted lines
    for ax, key, letter in ((top[0], "shap_" + a_ind, "a"), (top[1], "shap_" + b_ind, "b"), (top[2], "label", "c")):
        kind, ind = rc.quantity_parts(key)
        shifted = Z[f"placebo_{key}"]
        for row in shifted:
            ok = np.isfinite(row)
            ax.plot(cen[ok], row[ok], color=GREY, lw=0.3, alpha=0.5, zorder=1)
        with np.errstate(all="ignore"):
            lo, hi = np.nanmin(shifted, 0), np.nanmax(shifted, 0)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(cen[ok], lo[ok], hi[ok], color=GREY, alpha=0.22, lw=0, zorder=0,
                        label="20 shifted boundaries (range)")
        ax.plot(cen, Z[f"real_geom_{key}"], color=RED, lw=1.6, zorder=3, label="Real boundary (polygon side)")
        ax.plot(cen, Z[f"real_label_{key}"], color=BLUE, lw=0.9, ls="--", zorder=4, label="Real boundary (label side)")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax.set_xlim(-40, 40)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_xlabel("Signed distance to line (m; <0 inside)")
        ax.set_ylabel("Mean SHAP (+ = toward favela)" if kind == "shap" else "Label − 0.5 (favela = +0.5)")
        ax.set_title(PRETTY[ind] + (" SHAP" if kind == "shap" else ""), fontsize=7.5, fontweight="bold", loc="left")
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
        despine(ax)
    top[2].set_ylim(-0.62, 0.9)
    handles, labels = top[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.6, handlelength=2.0, bbox_to_anchor=(0.5, 0.995))

    # ---- d: normalised profiles on the real boundary, label side (convention of main-text Figure 4)
    ax = ax_d
    for key, colour in (("label", BLACK), ("shap_" + a_ind, RED), ("shap_" + b_ind, ORANGE)):
        mu = Z[f"real_label_{key}"]
        inner, outer = np.nanmean(mu[cen <= -rc.DEEP]), np.nanmean(mu[cen >= rc.DEEP])
        nm = (mu - outer) / (inner - outer)
        kind, ind = rc.quantity_parts(key)
        label = "Label step" if kind == "label" else PRETTY[ind].replace("Interbuilding distance", "Interbuilding dist.") + " SHAP"
        ok = np.isfinite(nm)
        ax.plot(cen[ok], nm[ok], color=colour, lw=1.2, label=label, ls="--" if kind == "label" else "-")
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axhline(1, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-40, 40)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_ylim(-0.75, 1.25)
    ax.set_xlabel("Signed distance to real line (m; <0 inside)")
    ax.set_ylabel("Normalised profile (outer = 0, inner = 1)")
    ax.legend(loc="lower left", fontsize=5.6, handlelength=1.8, borderaxespad=0.2)
    ax.set_title("Smoothness vs label step", fontsize=7.5, fontweight="bold", loc="left")
    ax.text(-0.22, 1.06, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")
    despine(ax)

    # ---- e: within-class gradient, real line vs shifted lines
    ax = ax_e
    rng = np.random.default_rng(DOT_JITTER_SEED)
    shifted = PT[(PT["level"] == "pooled") & (PT["boundary"] == "placebo")]
    real = PT[(PT["level"] == "pooled") & (PT["boundary"] == "real_geom")]
    for i, f in enumerate(rc.KEY5):
        for j, (metric, dx, marker) in enumerate((("G_fav", -0.2, "^"), ("G_non", 0.2, "v"))):
            values = shifted[(shifted["quantity"] == "shap") & (shifted["indicator"] == f)][metric].to_numpy()
            ax.scatter(i + dx + rng.uniform(-0.07, 0.07, len(values)), values, s=6, color=GREY, alpha=0.8, lw=0,
                       zorder=2, label="SHAP, 20 shifted boundaries" if (i == 0 and j == 0) else None)
            rv = real[(real["quantity"] == "shap") & (real["indicator"] == f)][metric].to_numpy()[0]
            ax.scatter(i + dx, rv, s=34, marker=marker, color=RED, zorder=4,
                       label=("SHAP, real line: favela-labelled (▲)" if metric == "G_fav"
                              else "SHAP, real line: formal settlement (▼)") if i == 0 else None)
            rr = real[(real["quantity"] == "raw") & (real["indicator"] == f)][metric].to_numpy()[0]
            ax.scatter(i + dx, rr, s=26, marker=marker, facecolor="none", edgecolor=BLUE, lw=0.8, zorder=3,
                       label="Raw feature (IQR units), real line" if (i == 0 and j == 0) else None)
    ax.axhline(0, color=BLACK, lw=0.8, ls="--", zorder=1)          # label step: G = 0 by construction
    ax.set_xticks(range(len(rc.KEY5)))
    ax.set_xticklabels([PRETTY[f] for f in rc.KEY5], fontsize=6.3)
    ax.set_xlim(-0.55, len(rc.KEY5) - 0.45)
    ax.set_ylabel("Within-class gradient G")
    ax.legend(loc="upper right", fontsize=5.8, ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
    ax.set_ylim(-0.5, 1.15)
    ax.set_title("Within-class gradient", fontsize=7.5, fontweight="bold", loc="left")
    ax.text(-0.09, 1.06, "e", transform=ax.transAxes, fontsize=10, fontweight="bold")
    despine(ax)

    unscaled = rc.OUT / "Figure_R1_placebo_boundaries_unscaled.pdf"
    fig.savefig(unscaled, bbox_inches="tight")
    print(f"saved {unscaled}")
    rc.scale_pdf_to_width(unscaled, rc.OUT / "Figure_R1_placebo_boundaries.pdf", float(rc.FS["figure_width_mm"]))


if __name__ == "__main__":
    main()
