"""Extended Data Fig. 2: per-site morphological feature importance for the favela class.

For each site, draws the mean |Tree SHAP| of the 25 indicators over the site's
favela buildings (computed by main/03_site_signatures/01_importance_tables.py
from the main per-site models, Methods 4.4 and 4.6). Indicators share a common
order (mean over sites of the per-site values) and bars are coloured by
morphometric dimension (SI Table S2). The top three indicators of each site
are printed (Results 2.2, SI Section S6).

Input
* products.importance_favela (main/03_site_signatures): site, feature,
  mean_abs_shap, share, rank.
Outputs (outputs/appendix/S06_attribution_analyses/)
* persite_importance_favela.csv: the values drawn, with the rank of every
  indicator in the common order;
* Figure_X_persite_importance_favela.pdf (Extended Data Fig. 2).

Run from the repository root (after main/03_site_signatures):
    python appendix/S06_attribution_analyses/01_persite_importance.py
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from favela_xai.config import ConfigError, load_config
from favela_xai.features import DIMENSIONS, MODEL_FEATURE_ORDER, dimension_of, label_of
from favela_xai.io import write_table
from favela_xai.plotting_style import DIMENSION_COLOURS

from attribution_common import figure_style, save_pdf, settings, site_label, site_order

FIGURE = "Figure_X_persite_importance_favela.pdf"
TABLE = "persite_importance_favela.csv"


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    if cfg.get("importance.population") != "favela_buildings":
        raise ConfigError("ED Fig. 2 is computed over favela buildings (importance.population: favela_buildings)")
    if settings(cfg)["importance_common_order"] != "mean_over_sites":
        raise ConfigError("only importance_common_order: mean_over_sites is implemented")
    sites = site_order(cfg)

    # ---- per-site mean |SHAP| over favela buildings (main/03) ------------------
    imp = pd.read_csv(cfg.require_product("importance_favela"))
    imp = imp[imp["site"].isin(sites)]
    wide = imp.pivot(index="feature", columns="site", values="mean_abs_shap").reindex(MODEL_FEATURE_ORDER)
    missing = [s for s in sites if s not in wide.columns]
    if missing or wide.isna().any().any():
        raise ConfigError(f"products.importance_favela is incomplete (missing sites {missing}); "
                          "re-run main/03_site_signatures/01_importance_tables.py")
    for site in sites:
        top3 = wide[site].sort_values(ascending=False).head(3)
        print(f"  {cfg.site_name(site):24s} top 3: " + ", ".join(f"{label_of(f)} ({v:.2f})" for f, v in top3.items()))

    # ---- common order: mean over sites ----------------------------------------
    overall = wide[sites].mean(axis=1).sort_values(ascending=False)
    table = imp.copy()
    table["rank_common_order"] = table["feature"].map({f: i + 1 for i, f in enumerate(overall.index)})
    write_table(table.sort_values(["site", "rank_common_order"]), out / TABLE)
    print("common order (top to bottom):", [label_of(f) for f in overall.index])

    # ---- figure (Extended Data Fig. 2) -----------------------------------------
    figure_style(7)
    order = list(overall.index)[::-1]                 # reversed: most important at the top of barh
    ylabels = [label_of(f) for f in order]
    colours = [DIMENSION_COLOURS[dimension_of(f)] for f in order]
    ncols = 5
    nrows = max(1, math.ceil(len(sites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for idx, (ax, site) in enumerate(zip(axes, sites)):
        ax.barh(range(len(order)), wide[site].reindex(order).to_numpy(), color=colours, height=0.82)
        ax.set_title(site_label(cfg, site), fontsize=8, fontweight="bold")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(ylabels if idx % ncols == 0 else [], fontsize=5.4)   # labels on the left column only
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="x", labelsize=5.6)
        ax.set_xlabel("mean |SHAP|", fontsize=6.5)
    for ax in axes[len(sites):]:
        ax.axis("off")
    fig.legend(handles=[Patch(fc=DIMENSION_COLOURS[g], label=g) for g in DIMENSIONS],
               loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_pdf(fig, out / FIGURE)
    plt.close(fig)
    print(f"[saved] {out / TABLE}\n[saved] {out / FIGURE}")


if __name__ == "__main__":
    main()
