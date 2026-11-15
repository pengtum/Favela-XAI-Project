"""SI Fig. S9: direction of the boundary transition, by site.

Draws the per-site table of 05_boundary_direction.py (SI Section S6.1): one
row per site, ordered by net skew, and a bottom row 'All sites (mean)'.
Inward displacement (favela-side buildings closer to formal) is drawn to the
left, outward displacement (formal-side buildings closer to favela) to the
right, in per cent of the site's deep-to-deep contrast; the net skew is given
in a column on the right (mean +- s.d. across sites in the bottom row).

Inputs
* outputs/appendix/S06_attribution_analyses/boundary_direction_by_site.csv
  (05_boundary_direction.py).
Outputs (outputs/appendix/S06_attribution_analyses/)
* Figure_R1_boundary_direction.pdf (SI Fig. S9).

Run from the repository root (after 05_boundary_direction.py):
    python appendix/S06_attribution_analyses/06_figure_boundary_direction.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from favela_xai.config import ConfigError, load_config
from favela_xai.plotting_style import FAVELA_MAP, FORMAL_MAP

from attribution_common import figure_style, save_pdf, settings, site_label

FIGURE = "Figure_R1_boundary_direction.pdf"
TABLE = "boundary_direction_by_site.csv"
FAVELA_TEXT = "#2a5a70"      # value labels and header on the favela side
FORMAL_TEXT = "#2f6b47"      # value labels and header on the formal side
SKEW_TEXT = "#444444"        # net-skew column
MINUS = "−"


def signed(v: float) -> str:
    """Integer with an explicit sign and a true minus sign (U+2212)."""
    return f"{v:+.0f}".replace("-", MINUS)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    path = out / TABLE
    if not path.exists():
        raise ConfigError(f"{path} does not exist; run 05_boundary_direction.py first")
    df = pd.read_csv(path)
    sites = df[df["site"] != "all_sites"].sort_values("net_skew_pct", kind="stable")   # most negative at the top
    mean = df[df["site"] == "all_sites"].iloc[0]
    xmax = float(settings(cfg)["direction"]["figure_xlim"])

    figure_style(8)
    fig, ax = plt.subplots(figsize=(7.4, 4.02))
    ys = list(range(len(sites))) + [len(sites) + 1]              # one blank row before the mean
    inward = list(sites["inward_pct"]) + [mean["inward_pct"]]
    outward = list(sites["outward_pct"]) + [mean["outward_pct"]]
    skews = [signed(v) for v in sites["net_skew_pct"]]
    skews.append(f"{signed(mean['net_skew_pct'])} ± {mean['net_skew_sd_pct']:.0f}")     # mean +- s.d. across sites
    names = [site_label(cfg, s, short=True) for s in sites["site"]] + ["All sites (mean)"]

    ax.barh(ys, [-v for v in inward], height=0.62, color=FAVELA_MAP, edgecolor="white", linewidth=0.5)
    ax.barh(ys, outward, height=0.62, color=FORMAL_MAP, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333333", lw=1.0)
    pad = 1.5
    for y, vi, vo, sk in zip(ys, inward, outward, skews):
        bold = "bold" if y == ys[-1] else "normal"
        if np.isfinite(vi):
            ax.text(-vi - pad, y, f"{vi:.0f}", ha="right", va="center", fontsize=7.0, color=FAVELA_TEXT)
        if np.isfinite(vo):
            ax.text(vo + pad, y, f"{vo:.0f}", ha="left", va="center", fontsize=7.0, color=FORMAL_TEXT)
        ax.text(1.08, y, sk, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.8,
                color=SKEW_TEXT, fontweight=bold)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.4)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.set_xlim(-xmax, xmax)
    ticks = np.arange(-60, 61, 20)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(int(t))) for t in ticks], fontsize=8.5)
    ax.set_xlabel("displacement of the near-boundary group from its own reference class  (%)", fontsize=9.0)
    ax.tick_params(axis="y", left=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.7)

    # headers above the plot
    ax.text(-60, 1.038, "← closer to formal", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=8.0, color=FAVELA_TEXT)
    ax.text(60, 1.038, "closer to favela →", transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            fontsize=8.0, color=FORMAL_TEXT)
    ax.text(1.105, 1.038, "net skew", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.0,
            color=SKEW_TEXT, fontweight="bold")

    fig.legend(handles=[Patch(fc=FAVELA_MAP, label="favela-side buildings (inside the boundary)"),
                        Patch(fc=FORMAL_MAP, label="formal-side buildings (outside the boundary)")],
               loc="lower center", ncol=2, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 0.973, 1])
    save_pdf(fig, out / FIGURE)
    plt.close(fig)
    print(f"[saved] {out / FIGURE}")


if __name__ == "__main__":
    main()
