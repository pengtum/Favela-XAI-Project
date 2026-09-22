"""Fig. 4: boundary profile of the morphological signature, all 25 indicators.

Manuscript items
* Fig. 4 (Results 2.3): for each indicator the cross-site mean Tree-SHAP
  contribution (line) +-1 s.d. across the ten sites (band) versus the signed
  distance to the official 2013 favela boundary (x < 0 inside, x > 0 outside;
  red dashed line x = 0). Panels grouped by morphometric dimension (frame
  colour) and ordered by mean |SHAP|. The dot marks the sign-flip distance x0
  (shown where |dS| >= 0.1; opacity scales with strength).

The SVG output keeps the text editable for final layout.

Settings (config.yaml)
* boundary_profiles.half_range_m: x-axis range (-40 to +40 m).
* boundary_profiles.resolved_min_abs_delta_s: |dS| >= 0.1 for the dot (Methods 4.8).
* main_04_boundary_profiles.dot_opacity: opacity = clip(|dS| / 0.6, 0.42, 1).

Inputs (outputs/main/04_boundary_profiles/, written by 01_boundary_profiles.py)
  profiles_crosssite.csv, transition_metrics.csv

Outputs (outputs/main/04_boundary_profiles/)
  figure4.pdf, figure4.svg

Run (from the repository root)
  python main/04_boundary_profiles/02_figure4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# plotting_style selects the non-interactive Agg backend before pyplot is imported
from favela_xai.plotting_style import DIMENSION_COLOURS, FAVELA_MAP, FORMAL_MAP

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from favela_xai.config import load_config

# Figure-specific rcParams
FIG_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "legend.frameon": False, "axes.linewidth": 0.6,
}
BOUNDARY_LINE = "#B64342"          # red dashed line at x = 0 (Fig. 4 caption)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    half = float(cfg.get("boundary_profiles.half_range_m"))
    min_ds = float(cfg.get("boundary_profiles.resolved_min_abs_delta_s"))
    op = cfg.section("main_04_boundary_profiles")["dot_opacity"]

    metrics = pd.read_csv(cfg.require_product("transition_metrics")).sort_values("panel_order")
    cross_path = out / "profiles_crosssite.csv"
    if not cross_path.exists():
        raise FileNotFoundError(f"{cross_path} not found; run 01_boundary_profiles.py first")
    cross = pd.read_csv(cross_path)
    group_order = list(dict.fromkeys(metrics["dimension"]))      # dimensions in panel order

    plt.rcParams.update(FIG_RC)
    fig, axes = plt.subplots(5, 5, figsize=(12.5, 13.0))
    axf = axes.ravel()
    for k, r in enumerate(metrics.itertuples()):
        ax = axf[k]
        gc = DIMENSION_COLOURS[r.dimension]
        prof = cross[cross["feature"] == r.feature].sort_values("bin_centre")
        cen = prof["bin_centre"].to_numpy()
        mu = prof["mean_shap"].to_numpy(float)
        sd = prof["sd_shap"].to_numpy(float)
        ok = np.isfinite(mu)

        ax.axvspan(-half, 0, color=FAVELA_MAP, alpha=0.06)
        ax.axvspan(0, half, color=FORMAL_MAP, alpha=0.06)
        ax.fill_between(cen[ok], (mu - sd)[ok], (mu + sd)[ok], color=gc, alpha=0.20, lw=0)
        ax.plot(cen[ok], mu[ok], color=gc, lw=1.8)
        ax.axvline(0, color=BOUNDARY_LINE, lw=0.8, ls="--")
        ax.axhline(0, color="#999999", lw=0.5, ls=":")

        # Sign-flip dot on the curve, for resolved transitions only; opacity
        # (dot and label) scales with the strength |dS|
        x0, ds = float(r.x0), float(r.abs_delta_s)
        if np.isfinite(x0) and ds >= min_ds:
            aa = float(np.clip(ds / float(op["full_at_abs_delta_s"]), float(op["minimum"]), 1.0))
            ax.plot(x0, 0, marker="o", ms=8, mfc=(0.1, 0.1, 0.1, 0.40 * aa), mec=(0.1, 0.1, 0.1, aa), mew=1.4,
                    zorder=9)
            ha, dx = ("left", 7) if x0 < 25 else ("right", -7)
            ax.annotate(f"$x_0$={x0:+.1f} m", (x0, 0), xytext=(dx, 9), textcoords="offset points", ha=ha,
                        fontsize=7.3, color="#1a1a1a", alpha=aa, fontweight="bold", zorder=9)
        ax.set_xlim(-half, half)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(gc)
            sp.set_linewidth(1.6)
        ax.set_title(r.label, fontsize=9.0, color=gc, fontweight="bold", pad=4)
        ax.tick_params(labelsize=7.5)
        ax.locator_params(axis="both", nbins=3)
        if k % 5 == 0:
            ax.set_ylabel("SHAP ($\\rightarrow$ fav.)", fontsize=8.5)
    for ax in axf[len(metrics):]:
        ax.set_visible(False)

    handles = [Patch(ec=DIMENSION_COLOURS[g], fc="none", lw=2.8, label=g) for g in group_order]
    handles += [Line2D([0], [0], marker="o", color="none", mfc="#1a1a1a", mec="white", ms=8,
                       label="$x_0$ = SHAP sign-flip (favela $\\to$ formal); label = its distance from boundary")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.965),
               title="frame colour = morphometric dimension; line = cross-site mean, band = $\\pm$1 SD; "
                     "red dashed = official 2013 favela boundary", title_fontsize=9.5)
    fig.suptitle("Boundary profile of every indicator: SHAP vs signed distance to the official favela boundary"
                 " — dot marks where each feature flips favela$\\to$formal", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"figure4.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {out / 'figure4.pdf'} (+ .svg)")


if __name__ == "__main__":
    main()
