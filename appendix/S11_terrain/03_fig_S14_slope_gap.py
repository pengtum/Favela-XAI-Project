"""SI Fig. S14: slope gap between favela and formal-settlement buildings versus per-site accuracy.

Manuscript items
* SI Fig. S14 and SI S11 'Experiment 1': Spearman correlation of the per-site
  slope gap with the per-site accuracy of the morphological classifier and
  with the misclassification rate (n = 10 sites).

Inputs
* outputs/appendix/S11_terrain/exp1_global_site_summary.csv (02): slope gap per site.
* Per-site accuracy (``appendix_S11_terrain.figure_accuracy``): SI Table S3 from
  ``products.site_metrics`` (main/02_classification; default) or the Morph
  column of the ablation (02).

Outputs (outputs/appendix/S11_terrain/)
* ``exp1_global_correlations.csv``: Spearman rho and p of the slope gap against
  accuracy and against the misclassification rate (1 - accuracy), n sites.
* ``exp1_fig3_slope_gap_vs_accuracy.pdf``: SI Fig. S14.

Figure layout: fixed default axis ranges, widened only when a site falls
outside them.

Run from the repository root:
    python appendix/S11_terrain/03_fig_S14_slope_gap.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from favela_xai.config import load_config  # noqa: E402
from favela_xai.io import write_table  # noqa: E402
from favela_xai.plotting_style import FAVELA_MAP  # noqa: E402

import terrain_common as tc  # noqa: E402

# Figure style
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "pdf.fonttype": 42, "font.size": 9, "axes.linewidth": 0.8}
POINT_COLOUR = FAVELA_MAP          # "#3F83A2"
TREND_COLOUR = "#c2c2c2"
XLIM, YLIM = (-0.3, 5.4), (82.0, 100.0)   # default axis ranges (slope gap in deg, accuracy in %)

# Label placement per site (dx, dy in points; horizontal alignment; thin leader line).
# Layout only.
LABELS = {
    "rocinha": (-9, 3, "right", False), "faz_quem_quer": (11, 0, "left", False),
    "fazena_coqueiro": (11, 3, "left", False), "barao": (10, 6, "left", False),
    "catumbi": (11, -1, "left", False), "nova_cidade": (11, 1, "left", False),
    "grotao_de_costa_barros": (11, 1, "left", False), "morro_da_guaiba": (46, 22, "left", True),
    "jacarezinho": (10, -17, "left", True), "palmeras": (42, 2, "left", True),
}


def widened(limits, values, pad):
    """Keep the default axis range unless a value falls outside it."""
    lo, hi = limits
    return min(lo, float(np.nanmin(values)) - pad), max(hi, float(np.nanmax(values)) + pad)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    mpl.rcParams.update(RC)

    d = pd.read_csv(tc.require(out / tc.SITE_SUMMARY_CSV, "02_terrain_ablation.py")).set_index("site")
    d = d.loc[cfg.sites]
    d["accuracy"] = tc.figure_accuracy(cfg, out) * 100.0          # percent, as drawn
    d["misclass_rate"] = 100.0 - d["accuracy"]
    d["name"] = [tc.display_name(cfg, s) for s in d.index]

    rho, p = stats.spearmanr(d["slope_gap"], d["accuracy"])
    rho_m, p_m = stats.spearmanr(d["slope_gap"], d["misclass_rate"])
    write_table(pd.DataFrame([
        dict(comparison="slope_gap vs accuracy", spearman_rho=rho, p_value=p, n=len(d),
             accuracy_source=cfg.get("appendix_S11_terrain.figure_accuracy")),
        dict(comparison="slope_gap vs misclass_rate", spearman_rho=rho_m, p_value=p_m, n=len(d),
             accuracy_source=cfg.get("appendix_S11_terrain.figure_accuracy")),
    ]), out / tc.SLOPE_GAP_CORR_CSV)
    print(f"  Spearman rho = {rho:.3f}, p = {p:.3f} (n = {len(d)}); misclassification rate: rho = {rho_m:.3f}")

    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    # reference trend (least squares), drawn in light grey
    if len(d) >= 2:
        z = np.polyfit(d["slope_gap"], d["accuracy"], 1)
        xl = np.linspace(d["slope_gap"].min() - 0.2, d["slope_gap"].max() + 0.45, 50)
        ax.plot(xl, np.poly1d(z)(xl), "--", color=TREND_COLOUR, lw=1.3, zorder=1)
    ax.scatter(d["slope_gap"], d["accuracy"], s=130, c=POINT_COLOUR, ec="white", lw=1.0, zorder=5)
    for site, r in d.iterrows():
        dx, dy, ha, leader = LABELS.get(site, (8, 5, "left", False))
        ax.annotate(r["name"], (r["slope_gap"], r["accuracy"]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.8, color="#2b2b2b", ha=ha, va="center",
                    arrowprops=(dict(arrowstyle="-", color="#b8b8b8", lw=0.6, shrinkA=1, shrinkB=3)
                                if leader else None))
    ax.set_xlabel("Slope gap between favela and non-favela buildings  (°)", fontsize=9.6)
    ax.set_ylabel("Per-site classification accuracy  (%)", fontsize=9.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(*widened(XLIM, d["slope_gap"], 0.3))
    ax.set_ylim(*widened(YLIM, d["accuracy"], 1.0))
    sig = "  (n.s.)" if p >= 0.05 else ""
    ax.annotate(f"Spearman $\\rho$ = {rho:.2f},  $p$ = {p:.2f}{sig}", xy=(0.96, 0.06), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9.2, color="#444",
                bbox=dict(boxstyle="round,pad=0.45", fc="#f6f6f6", ec="#cfcfcf", lw=0.8))
    fig.tight_layout()
    fig.savefig(out / tc.FIG_S14, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / tc.FIG_S14}")


if __name__ == "__main__":
    main()
