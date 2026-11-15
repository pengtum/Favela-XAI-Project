"""Step 4 of SI Section S14: Supplementary Figure S23 (label-free clustering against the official boundary).

Manuscript item: SI Fig. S23.
* a: profiles, pooled over the ten sites, of the label-free two-cluster score
  under the three weightings (orange) and of the supervised favela probability
  (blue, dashed) against signed distance to the 2013 boundary, each normalised
  to 1 in the favela interior and 0 in the formal-settlement interior; r,
  correlation with the supervised profile (0.865, 0.956, 0.960; SI Table S18).
* b: per-site balanced accuracy against the favela labels, for the supervised
  classifier under five-fold cross-validation (SI S12 pipeline; per-site mean
  over the five folds) and for the best label-free clustering of the same site
  (any weighting, 2-6 clusters, k-means or Ward); dotted line, chance. Sites
  ordered by the supervised value.

Figure size 6.8 x 2.9 in, saved with a tight bounding box; PNG preview at 100 dpi.

Inputs: profiles_by_weighting.npz, weighting_sensitivity.csv
(02_weighting_sensitivity.py); outputs/appendix/S12_spatial_cv/folds_all.parquet
(products.spatial_cv_folds).
Outputs (outputs/appendix/S14_label_free/): Figure_R2_labelfree.pdf, Figure_R2_labelfree.png.

Run from the repository root:
    python appendix/S14_label_free/04_figure_S23.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import label_free_common as lf  # noqa: E402

# Figure style: 6-7 pt Arial, TrueType glyphs so the PDF stays vector
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
      "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.7,
      "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.right": False,
      "axes.spines.top": False, "legend.frameon": False, "svg.fonttype": "none"}
BLUE, ORANGE = "#1F6FB4", "#E2700E"
# Panel a: legend text, colour and line style of each weighting
CURVES = {"all_equal": ("clusters, all 25 indicators equal", "#F2B880", "-"),
          "dimensions_balanced": ("clusters, five dimensions balanced", ORANGE, "-"),
          "one_component_per_dimension": ("clusters, one summary per dimension", "#9C4A08", ":")}
FIGSIZE_IN = (6.8, 2.9)


def main() -> None:
    cfg = lf.cfg
    plt.rcParams.update(RC)

    # ------------------------------------------------------------ panel a
    Z = np.load(lf.OUT / "profiles_by_weighting.npz")
    cen = Z["cen"]
    sup = lf.normalise(lf.pooled(Z["sup"]))
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE_IN, gridspec_kw=dict(width_ratios=[1, 1.05]))
    ax[0].plot(cen, sup, color=BLUE, lw=2.2, ls="--", label="supervised probability (five-fold cross-validation)")
    for w in lf.WEIGHTINGS:
        label, colour, style = CURVES[w]
        p = lf.normalise(lf.pooled(Z[w]))
        r = lf.profile_correlation(lf.pooled(Z[w]), lf.pooled(Z["sup"]))
        ax[0].plot(cen, p, color=colour, lw=1.6, ls=style, label=f"{label} (r = {r:.2f})")
    ax[0].axvline(0, color="#888", ls=":", lw=0.7)
    ax[0].set_ylim(-0.75, 1.75)
    ax[0].set_xlabel("signed distance to 2013 boundary (m)")
    ax[0].set_ylabel("normalised profile (inside = 1, outside = 0)")
    ax[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    ax[0].set_title("a", loc="left", fontweight="bold")

    # ------------------------------------------------------------ panel b
    res = pd.read_csv(lf.OUT / "weighting_sensitivity.csv")
    best = res.groupby("site")["best_bal_acc"].max()
    sp = lf.supervised_five_fold(cfg.sites)
    order = sp.sort_values().index
    yy = np.arange(len(order))
    ax[1].barh(yy + 0.2, sp[order], height=0.38, color=BLUE, label="supervised, five-fold cross-validation")
    ax[1].barh(yy - 0.2, best[order], height=0.38, color=ORANGE, label="best label-free clustering")
    ax[1].set_yticks(yy)
    ax[1].set_yticklabels([lf.site_display_name(s) for s in order], fontsize=6.5)
    ax[1].set_xlim(0.45, 1.0)
    ax[1].axvline(0.5, color="#888", ls=":", lw=0.7)
    ax[1].set_xlabel("balanced accuracy against the favela labels")
    ax[1].legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=0.8)
    ax[1].set_title("b", loc="left", fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(lf.OUT / "Figure_R2_labelfree.pdf", bbox_inches="tight")
    fig.savefig(lf.OUT / "Figure_R2_labelfree.png", dpi=100, bbox_inches="tight")
    print(f"saved {lf.OUT / 'Figure_R2_labelfree.pdf'}")

    # ------------------------------------------------------------ read-out
    print(f"panel a  supervised midpoint {lf.crossing(sup, 0.5):+.2f} m, "
          f"10-90% width {lf.crossing(sup, 0.1) - lf.crossing(sup, 0.9):.1f} m")
    print("panel b  site                     supervised  label-free")
    for s in order:
        print(f"         {s:<24}{sp[s]:>10.4f}{best[s]:>12.4f}")
    print(f"         {'per-site mean':<24}{sp.mean():>10.4f}{best.mean():>12.4f}")


if __name__ == "__main__":
    main()
