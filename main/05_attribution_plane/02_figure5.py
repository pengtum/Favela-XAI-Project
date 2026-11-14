"""Fig. 5: building-level attribution planes, site by site.

Manuscript items
* Fig. 5a-j (Results 2.4): each site's buildings on the common PC1/PC2 axes of
  the pooled attribution PCA. Filled areas with solid outlines (favela) and
  dashed outlines (formal settlement) mark the 15% of grid cells with the
  highest kernel density of each class; points are random samples of up to
  2,000 buildings per class.
* Results 2.4: the two classes are displaced with overlap in every site.
  panel_stats.csv gives, for each site, the out-of-fold AUC of a logistic
  regression on PC1 and PC2 as a descriptive check of each panel; the paper
  does not report it.

Figure size 183 mm wide; vector output with the point layer rasterised at 600 dpi.

Settings (config.yaml)
* pca.density_top_share (0.15): density regions = grid cells above the
  (1 - 0.15) quantile of the class density on the grid.
* pca.points_per_class (2,000) and pca.sample_seed (42): plotted points.
* pca.kde_max_buildings_per_site (20,000), pca.kde_grid_size (150) and
  main_05_attribution_plane.kde_* : buildings used for the density (a random
  sample per site), grid on the 1st-99th percentile box, Gaussian kernel with
  bandwidth n^(-1/6) x mean s.d. of PC1 and PC2 (clipped to [0.05, 2]), no
  region for a class with fewer than 30 buildings.
* display_orders.figure5_panels: panel order a-j.
* main_05_attribution_plane.site_colours: colour of each site's panel.
The plotted points are drawn from the density sample of the site.

Inputs
  outputs/main/05_attribution_plane/attribution_pca_scores.parquet (01_attribution_pca.py)

Outputs (outputs/main/05_attribution_plane/)
  figure5.pdf, figure5.svg    the figure
  panel_stats.csv             per site: buildings, favela buildings, out-of-fold
                              AUC of a logistic regression on PC1 and PC2

Run (from the repository root)
  python main/05_attribution_plane/02_figure5.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# plotting_style selects the non-interactive Agg backend before pyplot is imported
from favela_xai.plotting_style import MM, WIDTH_DOUBLE_MM

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neighbors import KernelDensity  # noqa: E402

from favela_xai.config import load_config
from favela_xai.io import write_table

# ----------------------------------------------------------------------------- style
# Figure-specific rcParams
FIG_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
    "font.family": "sans-serif", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
}
HEIGHT_MM = 78                                   # 183 x 78 mm, two rows of five panels
POINT_SIZE = 0.45                                # scatter marker size (pt^2)
LW_FAVELA, LW_FORMAL, LW_HOLLOW = 0.6, 0.75, 0.2   # outline widths, hollow-marker edge
FILL_ALPHA, POINT_ALPHA = 0.35, 0.25
PANEL_ASPECT = 1 / 1.096                         # panel height / width
PANEL_LETTERS = "abcdefghij"


# ----------------------------------------------------------------------------- helpers
def quantile_box(x, y, q_low: float, q_high: float):
    """Grid extent: the q_low-q_high quantile box of the points (never degenerate)."""
    x_lo, x_hi = np.nanquantile(x, [q_low, q_high])
    y_lo, y_hi = np.nanquantile(y, [q_low, q_high])
    if x_hi <= x_lo:
        x_hi = x_lo + 1e-6
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    return (x_lo, x_hi), (y_lo, y_hi)


def kde_bandwidth(points: np.ndarray, clip: tuple[float, float]) -> float:
    """Bandwidth n^(-1/6) x mean sample s.d. of the two coordinates, clipped."""
    n = len(points)
    if n <= 1:
        return 0.3
    h = (n ** (-1.0 / 6.0)) * np.mean(np.std(points, axis=0, ddof=1))
    return float(np.clip(h, clip[0], clip[1]))


def panel_auc(pc: np.ndarray, y: np.ndarray, opts: dict) -> float:
    """Out-of-fold AUC of a logistic regression on PC1 and PC2 (panel_stats.csv only)."""
    cv = StratifiedKFold(int(opts["n_folds"]), shuffle=bool(opts["shuffle"]), random_state=int(opts["seed"]))
    prob = cross_val_predict(LogisticRegression(max_iter=int(opts["max_iter"])), pc, y, cv=cv,
                             method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    p = cfg.section("pca")
    opts = cfg.section("main_05_attribution_plane")
    q_dens = round(1.0 - float(p["density_top_share"]), 12)   # 0.85: top 15% of grid cells
    n_points = int(p["points_per_class"])
    seed = int(p["sample_seed"])
    n_kde = int(p["kde_max_buildings_per_site"])
    grid_n = int(p["kde_grid_size"])
    q_low, q_high = (float(v) for v in opts["kde_quantile_box"])
    bw_clip = tuple(float(v) for v in opts["kde_bandwidth_clip"])
    min_kde = int(opts["kde_min_points"])
    colours = opts["site_colours"]

    scores = pd.read_parquet(cfg.require_product("attribution_pca_scores"))
    panels = cfg.display_order("figure5_panels")
    if not panels:
        raise ValueError("no site of display_orders.figure5_panels is being processed")

    plt.rcParams.update(FIG_RC)
    n_cols = min(5, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH_DOUBLE_MM * MM, HEIGHT_MM * MM * n_rows / 2),
                             squeeze=False)
    stats = []
    for k, site in enumerate(panels):
        ax = axes.flat[k]
        g = scores[scores["site"] == site]
        colour = colours.get(site, "#4c72b0")

        # Separation of the classes on PC1-PC2, all buildings of the site (reported, not drawn)
        y = g["is_favela"].astype(int).to_numpy()
        stats.append(dict(site=site, site_name=cfg.site_name(site), n=len(g), n_favela=int(y.sum()),
                          auc_pc12=round(panel_auc(g[["PC1", "PC2"]].to_numpy(), y, opts["panel_auc"]), 3)))

        # Density sample of the site
        if len(g) > n_kde:
            g = g.sample(n_kde, random_state=seed)
        (x_lo, x_hi), (y_lo, y_hi) = quantile_box(g["PC1"].to_numpy(), g["PC2"].to_numpy(), q_low, q_high)
        xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, grid_n), np.linspace(y_lo, y_hi, grid_n))
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        # High-density region of each class: filled with a solid outline (favela),
        # dashed outline only (formal settlement)
        for is_fav in (True, False):
            pts = g.loc[g["is_favela"] == is_fav, ["PC1", "PC2"]].dropna().to_numpy()
            if len(pts) < min_kde:
                continue
            kde = KernelDensity(bandwidth=kde_bandwidth(pts, bw_clip), kernel="gaussian").fit(pts)
            dens = np.exp(kde.score_samples(grid)).reshape(grid_n, grid_n)
            thr = np.quantile(dens, q_dens)
            if is_fav:
                ax.contourf(xx, yy, dens, levels=[thr, dens.max()], alpha=FILL_ALPHA, colors=[colour],
                            antialiased=True)
                ax.contour(xx, yy, dens, levels=[thr], colors=[colour], linewidths=LW_FAVELA)
            else:
                ax.contour(xx, yy, dens, levels=[thr], colors=[colour], linewidths=LW_FORMAL, linestyles="--")

        # Random sample of up to 2,000 buildings per class: filled dots (favela),
        # hollow dots (formal settlement)
        for is_fav, style in [(True, dict(marker="o", facecolors=colour, edgecolors="none", alpha=POINT_ALPHA)),
                              (False, dict(marker="o", facecolors="none", edgecolors=colour, alpha=POINT_ALPHA,
                                           linewidths=LW_HOLLOW))]:
            sub = g[g["is_favela"] == is_fav]
            if len(sub):
                sub = sub.sample(min(len(sub), n_points), random_state=seed)
                ax.scatter(sub["PC1"], sub["PC2"], s=POINT_SIZE, rasterized=True, **style)

        ax.set_title(cfg.site_name(site), pad=3)
        ax.set_xlabel("PC1", labelpad=1)
        ax.set_ylabel("PC2", labelpad=1)
        ax.tick_params(pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2], integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2], integer=True))
        ax.set_box_aspect(PANEL_ASPECT)
        ax.text(-0.28, 1.12, PANEL_LETTERS[k], transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", ha="left")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray", markeredgecolor="gray",
               markersize=5, label="Favela"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="gray",
               markeredgewidth=0.8, markersize=5, label="Formal settlement"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.5, handletextpad=0.4)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.15, top=0.93, wspace=0.42, hspace=0.55)

    for ext in ("pdf", "svg"):
        fig.savefig(out / f"figure5.{ext}", dpi=600)
    plt.close(fig)
    stats = pd.DataFrame(stats)
    write_table(stats, out / "panel_stats.csv")
    print(stats.to_string(index=False))
    print(f"[saved] {out / 'figure5.pdf'} (+ .svg)")


if __name__ == "__main__":
    main()
