"""Step 5: Supplementary Figs S4 and S5, four building clusters read building by building.

Manuscript items (SI S5, 'Case studies: reading individual buildings')
* Supplementary Fig. S4: four building clusters (Rocinha: a boundary cluster
  and a within-site cluster; Grotao de Costa Barros: a perimeter-wall cluster
  and a false-negative cluster), footprints coloured by ground-truth label,
  false positives outlined in red and false negatives in purple.
* Supplementary Fig. S5: buildings classified as favela coloured by the
  leading feature that contributes most to their classification (mean
  interbuilding distance, adjacency, perimeter wall, squareness), with the
  per-feature SHAP attribution of a representative building of each cluster.

Definitions
* Predictions and Tree SHAP: the main per-site classifier (Methods 4.4) and its
  attributions for every building (main/02_classification); false positive =
  formal building classified favela, false negative = favela building
  classified formal.
* Dominant driver of a building: the indicator with the largest (signed)
  Tree-SHAP value among the 25; buildings whose driver is not one of the four
  leading features are drawn as 'other feature'.
* Windows: squares of half-width r around the centres of
  ``case_studies.windows`` (EPSG:31983), buildings selected by centroid.
* Representative building of each cluster (right column of Fig. S5): (a) the
  false positive with the largest SHAP sum; (b) the favela-classified building
  driven by adjacency with the largest adjacency SHAP; (c) the favela-classified
  building driven by the perimeter wall with the largest perimeter-wall SHAP;
  (d) the false negative with the smallest SHAP sum. The eight largest |SHAP|
  values are drawn.

Inputs:  ``products.shap_gpkg`` (main/02_classification).
Outputs (outputs/appendix/S05_boundary_errors/): Figure_R1_casestudies_class.pdf,
         Figure_R1_casestudies_driver.pdf (+ PNG previews), case_study_representatives.csv.

Run from the repository root:
    python appendix/S05_boundary_errors/05_case_studies.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from favela_xai.config import load_config  # noqa: E402
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, dimension_of, label_of  # noqa: E402
from favela_xai.io import write_table  # noqa: E402

import s05_common as C  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                     "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
# Palette: one neutral family, signal hues; red / violet reserved for errors
COL = dict(favela="#E28E2C", formal="#CFCECE", fp="#B64342", fn="#9A4D8E", other="#CFCECE", pred_formal="#ECECEC")
DRIVER_COL = {"mean_interbuilding_distance": "#14746A", "adjacency": "#7FCBC0", "perimeter_wall": "#E2700E",
              "squareness": "#B5341F"}
# Bar colours of the per-building SHAP panels: shades within the five dimension families
FEATURE_COL = {
    "mean_interbuilding_distance": "#14746A", "adjacency": "#7FCBC0", "CAR": "#2A9D8F", "neighbour_dist": "#0E5249",
    "perimeter_wall": "#E2700E", "squareness": "#B5341F", "cwa": "#F2B84B", "compact_weighted_axis": "#9C5418",
    "shared_walls": "#8C4A28", "rectangularity": "#D99441", "shape_idx": "#C77A2E", "convexity": "#A0511C",
    "facade_ratio": "#E8A05A", "corners": "#7A3A12", "rect_idx": "#D9742E", "square_compact": "#C46A1C",
    "elongation": "#F0A868", "avg_weighted_dist": "#4C9F70", "fractal": "#1E7A45", "num_neighbors": "#9A4D8E",
    "cell_alignment": "#C58EBC", "tes_area": "#6F3566", "area": "#0F4D92", "perimeter": "#5B8FC0", "lal": "#2C6FB0",
}


def load(cfg, site):
    g = C.main_outputs(cfg, site, geometry=True)
    g = g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    cen = g.geometry.centroid
    g["_x"], g["_y"] = cen.x, cen.y
    g["FP"] = (g.is_favela == 0) & (g.pred_favela == 1)
    g["FN"] = (g.is_favela == 1) & (g.pred_favela == 0)
    g["driver"] = np.asarray(MODEL_FEATURE_ORDER)[np.argmax(g[SHAP_COLUMNS].to_numpy(float), axis=1)]
    return g


def window(g, cx, cy, r):
    return g[(g._x > cx - r) & (g._x < cx + r) & (g._y > cy - r) & (g._y < cy + r)]


def panel_letter(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def scalebar(ax, length=100):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + (x1 - x0) * 0.06, y0 + (y1 - y0) * 0.07
    ax.plot([x, x + length], [y, y], color="k", lw=1.4, solid_capstyle="butt")
    ax.text(x + length / 2, y + (y1 - y0) * 0.02, f"{length} m", ha="center", va="bottom", fontsize=5.5)


def shapbar(ax, row, title):
    s = pd.Series([row["shap_" + k] for k in MODEL_FEATURE_ORDER], index=MODEL_FEATURE_ORDER)
    s = s.sort_values(key=np.abs, ascending=False).head(8)[::-1]
    ax.barh([label_of(k) for k in s.index], s.values, color=[FEATURE_COL.get(k, "#9e9e9e") for k in s.index],
            height=.66)
    ax.axvline(0, color="#767676", lw=.6)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("SHAP   ($\\leftarrow$ formal | favela $\\rightarrow$)", fontsize=6)


def representative(i, w, predfav):
    """Representative building of cluster i (rules in the docstring); returns (row, label) or (None, label)."""
    if i == 0:
        q = w[w.FP]
        return (q.assign(t=q[SHAP_COLUMNS].sum(1)).sort_values("t", ascending=False).iloc[0] if len(q) else None,
                "false positive (non-favela)")
    if i == 1:
        q = predfav[predfav.driver == "adjacency"]
        q = q if len(q) else predfav
        return (q.sort_values("shap_adjacency", ascending=False).iloc[0] if len(q) else None, "adjacency-driven building")
    if i == 2:
        q = predfav[predfav.driver == "perimeter_wall"]
        q = q if len(q) else predfav
        return (q.sort_values("shap_perimeter_wall", ascending=False).iloc[0] if len(q) else None,
                "perimeter-wall–driven building")
    q = w[w.FN]
    return (q.assign(t=q[SHAP_COLUMNS].sum(1)).sort_values("t").iloc[0] if len(q) else None, "false negative (favela)")


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    cs = C.settings(cfg)["case_studies"]
    cases = [(c["letter"], c["title"], c["site"], float(c["x"]), float(c["y"]), float(c["half_width_m"]))
             for c in cs["windows"] if c["site"] in cfg.sites]
    if not cases:
        print("no case-study site among the processed sites; nothing drawn")
        return
    drivers = list(cs["drivers"])
    data = {s: load(cfg, s) for s in sorted({c[2] for c in cases})}

    # ===== SI Fig. S4: classification outcome (2 x 2)
    figA, axA = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for ax, (L, title, site, cx, cy, r) in zip(axA.ravel(), cases):
        w = window(data[site], cx, cy, r)
        w[w.is_favela == 1].plot(ax=ax, color=COL["favela"], ec="white", lw=.06)
        w[w.is_favela == 0].plot(ax=ax, color=COL["formal"], ec="white", lw=.06)
        if w.FP.any():
            w[w.FP].boundary.plot(ax=ax, color=COL["fp"], lw=.9)
        if w.FN.any():
            w[w.FN].boundary.plot(ax=ax, color=COL["fn"], lw=.9)
        ax.set_title(title, fontsize=7.5)
        ax.set_aspect("equal")
        scalebar(ax)
        ax.axis("off")
        panel_letter(ax, L)
    figA.legend(handles=[Patch(fc=COL["favela"], label="favela (ground truth)"),
                         Patch(fc=COL["formal"], label="non-favela"),
                         Line2D([0], [0], color=COL["fp"], lw=1.3, label="false positive (formal→favela)"),
                         Line2D([0], [0], color=COL["fn"], lw=1.3, label="false negative (favela→formal)")],
                loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.015))
    figA.tight_layout(rect=[0, 0.04, 1, 1])
    figA.savefig(out / C.FIGURE_CASES_CLASS, bbox_inches="tight")
    figA.savefig((out / C.FIGURE_CASES_CLASS).with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(figA)

    # ===== SI Fig. S5: dominant driver + per-building SHAP (4 x 2)
    present, reps = set(), []
    figB, axB = plt.subplots(len(cases), 2, figsize=(7.4, 11.4 * len(cases) / 4),
                             gridspec_kw={"width_ratios": [1.1, 1]}, squeeze=False)
    for i, (L, title, site, cx, cy, r) in enumerate(cases):
        w = window(data[site], cx, cy, r)
        ax = axB[i, 0]
        w[w.pred_favela == 0].plot(ax=ax, color=COL["pred_formal"], ec="white", lw=.06)
        predfav = w[w.pred_favela == 1]
        for d, sub in predfav.groupby("driver"):
            sub.plot(ax=ax, color=DRIVER_COL.get(d, COL["other"]), ec="white", lw=.06)
            present.add(d)
        if w.FP.any():
            w[w.FP].boundary.plot(ax=ax, color="#272727", lw=.55, ls=":")
        ax.set_title(title, fontsize=7.5)
        ax.set_aspect("equal")
        scalebar(ax)
        ax.axis("off")
        panel_letter(ax, L)
        rep, rl = representative(i, w, predfav)
        if rep is not None:
            shapbar(axB[i, 1], rep, f"{L}  ·  {rl}")
            reps.append(dict(panel=L, site=site, label=rl, uID=rep.get("uID", np.nan),
                             x=float(rep["_x"]), y=float(rep["_y"]), dominant_driver=rep["driver"],
                             dimension=dimension_of(rep["driver"]), shap_sum=float(rep[SHAP_COLUMNS].sum())))
        else:
            axB[i, 1].axis("off")
    handles = [Patch(fc=DRIVER_COL[d], label=label_of(d)) for d in drivers if d in present]
    if present - set(drivers):
        handles.append(Patch(fc=COL["other"], label="other feature"))
    handles += [Patch(fc=COL["pred_formal"], label="predicted non-favela"),
                Line2D([0], [0], color="#272727", lw=.9, ls=":", label="false positive")]
    figB.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.3, bbox_to_anchor=(0.5, -0.008))
    figB.tight_layout(rect=[0, 0.03, 1, 1])
    figB.savefig(out / C.FIGURE_CASES_DRIVER, bbox_inches="tight")
    figB.savefig((out / C.FIGURE_CASES_DRIVER).with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(figB)
    write_table(pd.DataFrame(reps), out / "case_study_representatives.csv")
    print(f"drivers present: {sorted(present)}")
    print(f"written {out / C.FIGURE_CASES_CLASS} and {out / C.FIGURE_CASES_DRIVER}")


if __name__ == "__main__":
    main()
