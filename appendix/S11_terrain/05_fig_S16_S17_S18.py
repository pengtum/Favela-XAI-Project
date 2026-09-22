"""SI Figs S16, S17 and S18: Tree-SHAP importance and accuracies of the terrain ablation.

Manuscript items
* SI Fig. S16: mean |Tree SHAP| of the nine terrain features in the
  terrain-only classifier, averaged across the ten sites (SI S11
  'Experiment 2': ranking of the terrain features).
* SI Fig. S17: per-site accuracy of the morphology-only, terrain-only and
  combined classifiers.
* SI Fig. S18: the 20 features with the largest mean |Tree SHAP| in the
  combined classifier, averaged across the sites (SI S11 'Experiment 3':
  terrain features among the top five).

Inputs (outputs/appendix/S11_terrain/, written by 02_terrain_ablation.py)
* exp2_terrain_shap_importance.csv, exp3_combined_shap_importance.csv,
  summary_three_model_comparison.csv.

Outputs (outputs/appendix/S11_terrain/)
* exp2_fig6_terrain_shap_importance.pdf, summary_fig8_three_model_comparison.pdf,
  exp3_fig7_combined_shap_top20.pdf.

Figure labels: site names from config.yaml; tick labels of the morphological
indicators in Fig. S18 (``appendix_S11_terrain.morph_feature_labels``): ``keys``
(column names) or ``table_S2`` (SI Table S2 labels).

Run from the repository root:
    python appendix/S11_terrain/05_fig_S16_S17_S18.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from favela_xai.config import ConfigError, load_config  # noqa: E402
from favela_xai.features import label_of  # noqa: E402

import terrain_common as tc  # noqa: E402

# Figure style
RC = {"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
      "axes.labelsize": 12, "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
      "legend.fontsize": 10, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
      "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"}
MORPH_C = "#3498db"      # morphology
TERRAIN_C = "#e67e22"    # terrain
COMBINED_C = "#2ecc71"   # combined
SHAP_XLABEL = "Mean |SHAP Value| (averaged across sites)"


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {path}")


def fig_s16(imp: pd.DataFrame, path) -> None:
    """Horizontal bars of the site-averaged mean |SHAP| of the nine terrain features."""
    s = imp.sort_values("average", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(s)), s["average"], color=TERRAIN_C)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["feature"])
    ax.set_xlabel(SHAP_XLABEL)
    ax.set_title("Terrain Feature Importance (Terrain-Only Model)")
    fig.tight_layout()
    save(fig, path)


def fig_s17(summary: pd.DataFrame, names: list[str], path) -> None:
    """Grouped bars: per-site accuracy (%) of the three classifiers."""
    x = np.arange(len(summary))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, summary["morph_accuracy"] * 100, w, label="Morphology (25)", color=MORPH_C)
    ax.bar(x, summary["terrain_accuracy"] * 100, w, label="Terrain (9)", color=TERRAIN_C)
    ax.bar(x + w, summary["combined_accuracy"] * 100, w, label="Combined (34)", color=COMBINED_C)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 105)
    ax.legend()
    fig.tight_layout()
    save(fig, path)


def fig_s18(imp: pd.DataFrame, top_n: int, morph_labels: str, path) -> None:
    """Horizontal bars of the top-n features of the combined model, coloured by feature type."""
    top = imp.head(top_n).sort_values("average", ascending=True)
    if morph_labels == "keys":
        ticks = list(top["feature"])
    elif morph_labels == "table_S2":
        ticks = [f if t == "terrain" else label_of(f) for f, t in zip(top["feature"], top["type"])]
    else:
        raise ConfigError(f"appendix_S11_terrain.morph_feature_labels must be keys or table_S2, not '{morph_labels}'")
    colours = [TERRAIN_C if t == "terrain" else MORPH_C for t in top["type"]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(top)), top["average"], color=colours)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(ticks)
    ax.set_xlabel(SHAP_XLABEL)
    ax.set_title(f"Top {top_n} Features — Combined Model")
    ax.legend(handles=[Patch(facecolor=MORPH_C, label="Morphological"),
                       Patch(facecolor=TERRAIN_C, label="Terrain")], loc="lower right")
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    matplotlib.rcParams.update(RC)
    step = "02_terrain_ablation.py"

    terrain_imp = pd.read_csv(tc.require(out / tc.TERRAIN_SHAP_CSV, step))
    combined_imp = pd.read_csv(tc.require(out / tc.COMBINED_SHAP_CSV, step))
    summary = pd.read_csv(tc.require(out / tc.THREE_MODEL_CSV, step)).set_index("site")

    order = tc.figure_site_order(cfg)
    summary = summary.loc[order].reset_index()

    fig_s16(terrain_imp, out / tc.FIG_S16)
    fig_s17(summary, [tc.display_name(cfg, s) for s in order], out / tc.FIG_S17)
    fig_s18(combined_imp, int(cfg.get("appendix_S11_terrain.combined_top_n")),
            cfg.get("appendix_S11_terrain.morph_feature_labels"), out / tc.FIG_S18)

    print("  terrain-only ranking:", ", ".join(terrain_imp["feature"].head(3)))
    print("  combined top five:", ", ".join(combined_imp["feature"].head(5)))


if __name__ == "__main__":
    main()
