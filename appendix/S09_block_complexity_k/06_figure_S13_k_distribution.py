"""SI Fig. S13: distribution of the block-complexity index k by label.

Manuscript items
* SI Fig. S13: (A) pooled per-building histogram of k by label, k clipped at
  12 (density); (B) per-site mean k of favela and formal-settlement buildings
  (dumbbell), sites sorted by Cohen's d from negative (bottom) to positive
  (top). The statistics quoted in the caption (n, pooled means, Cohen's d) are
  in k_in_text_numbers.txt (05).

Figure settings: ``block_complexity.figure_k_clip`` (12) and
``appendix_S09_block_complexity_k.figure`` (layout: 10.5 in wide, two panels at
fixed positions, colours, short site names). Buildings: the valid-k subset
(k >= 1, complete Tree-SHAP attributions, label present).

Inputs (outputs/appendix/S09_block_complexity_k/, from 05_k_distribution.py)
* k_histogram.csv, k_distribution.csv

Output
* outputs/appendix/S09_block_complexity_k/Figure_R1_k_distribution.pdf

Run (from the repository root), after 05_k_distribution.py:
    python appendix/S09_block_complexity_k/06_figure_S13_k_distribution.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from favela_xai.config import ConfigError, load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    fig_cfg = cfg.section("appendix_S09_block_complexity_k")["figure"]
    kmax = int(cfg.get("block_complexity.figure_k_clip"))
    for name in ("k_histogram.csv", "k_distribution.csv"):
        if not (out / name).exists():
            raise ConfigError(f"{out / name} not found; run 05_k_distribution.py first")
    hist = pd.read_csv(out / "k_histogram.csv")
    ps = pd.read_csv(out / "k_distribution.csv")
    ps = ps[ps.site != "pooled"].sort_values("cohens_d", ascending=True).reset_index(drop=True)
    short = fig_cfg.get("short_names", {}) or {}
    names = [short.get(s, cfg.site_name(s)) for s in ps.site]
    fav, nfv = fig_cfg["favela_colour"], fig_cfg["formal_colour"]

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "font.size": 9, "axes.linewidth": 0.8})
    # Axes placed at fixed sizes and positions (inches, from config.yaml)
    FW = float(fig_cfg["width_in"])
    ax_w, ax_h = fig_cfg["axes_size_in"]
    bottom, top_margin = fig_cfg["bottom_in"], fig_cfg["top_margin_in"]
    FH = bottom + ax_h + top_margin
    fig = plt.figure(figsize=(FW, FH))
    axA = fig.add_axes([fig_cfg["left_a_in"] / FW, bottom / FH, ax_w / FW, ax_h / FH])
    axB = fig.add_axes([fig_cfg["left_b_in"] / FW, bottom / FH, ax_w / FW, ax_h / FH])

    # (A) pooled k histogram by class (density); counts of k = 1 ... kmax (last bin = kmax or more)
    bins = np.arange(0.5, kmax + 1.5, 1)
    axA.hist(hist["k"], bins=bins, weights=hist["n_formal"], density=True, color=nfv, alpha=0.75,
             label="non-favela")
    axA.hist(hist["k"], bins=bins, weights=hist["n_favela"], density=True, histtype="step", lw=2.0,
             edgecolor=fav, label="favela")
    axA.set_xlabel(f"block-complexity index $k$  (clipped at {kmax})")
    axA.set_ylabel("density")
    axA.legend(frameon=False, fontsize=8.5, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_title("(A)", loc="left", fontweight="bold", fontsize=11)

    # (B) per-site mean k by class, sorted by Cohen's d (dumbbell)
    y = np.arange(len(ps))
    for i, r in ps.iterrows():
        axB.plot([r["k_mean_formal"], r["k_mean_favela"]], [i, i], color="#d0d0d0", lw=2, zorder=1)
    axB.scatter(ps["k_mean_formal"], y, s=55, c=nfv, zorder=3, label="non-favela")
    axB.scatter(ps["k_mean_favela"], y, s=55, c=fav, zorder=3, label="favela")
    axB.set_yticks(y)
    axB.set_yticklabels(names, fontsize=8)
    axB.set_xlabel("per-site mean $k$")
    axB.set_xlim(0.8, float(ps["k_mean_favela"].max()) + 1.7)
    axB.set_ylim(-0.6, len(ps) - 0.4)
    axB.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("(B)", loc="left", fontweight="bold", fontsize=11)

    path = out / "Figure_R1_k_distribution.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
