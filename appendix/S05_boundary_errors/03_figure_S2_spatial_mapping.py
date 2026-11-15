"""Step 3: Supplementary Fig. S2, building-level classification outcomes in the ten sites.

Manuscript items
* Supplementary Fig. S2 (SI S5): buildings of the ten sites coloured by their
  classification outcome under the main per-site classifier (Methods 4.4),
  with the 2013 favela polygons as black outlines.

What is drawn
* ``appendix_S05_boundary_errors.spatial_mapping_figure.buildings``:
  ``test_only``: held-out test buildings of the 80/20 split of the main model coloured
  by outcome, training buildings in neutral grey;
  ``all_buildings`` (default, as published): every building coloured by the prediction
  of its site model (training and held-out test buildings).
* Outcome colours (``spatial_mapping_figure.colours``): favela correct (blue),
  formal settlement correct (green), favela misclassified (false negative,
  orange), formal settlement misclassified (false positive, red).
* One panel per site in the arrangement of ``spatial_mapping_figure.panel_rows``
  (a site key repeated in a row spans several columns). Each site map is also
  written as its own PDF (spatial_mapping_sites/) for figure assembly, where
  zoomed call-outs are added.

Inputs
* ``products.shap_gpkg`` (main/02_classification): footprints, labels,
  predicted class and test flag of the main per-site classifier.
* ``paths.raw.favelas_2013``: official 2013 favela polygons.

Outputs (outputs/appendix/S05_boundary_errors/)
* Figure_R1_spatial_mapping.pdf (+ PNG preview), spatial_mapping_sites/<site>.pdf,
  spatial_mapping_counts.csv (buildings per outcome and site, test set).

Run from the repository root:
    python appendix/S05_boundary_errors/03_figure_S2_spatial_mapping.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from favela_xai.config import load_config  # noqa: E402
from favela_xai.io import read_favela_polygons, write_table  # noqa: E402
from favela_xai.plotting_style import BOUNDARY, NEUTRAL_BUILDING, use_style  # noqa: E402

import s05_common as C  # noqa: E402

OUTCOME_LABEL = {"TP": "favela, correctly classified", "TN": "formal settlement, correctly classified",
                 "FN": "favela, misclassified", "FP": "formal settlement, misclassified"}


def draw_site(ax, g, polys, colours, lw_poly):
    """One site map: training buildings grey, outcomes coloured, 2013 polygons outlined."""
    b = g.total_bounds
    if (~g["shown"]).any():
        g[~g["shown"]].plot(ax=ax, color=NEUTRAL_BUILDING, lw=0, rasterized=True)
    for k in ("TN", "TP", "FN", "FP"):                   # errors drawn last, on top
        sub = g[g["shown"] & (g["outcome"] == k)]
        if len(sub):
            sub.plot(ax=ax, color=colours[k], lw=0, rasterized=True)
    box = shapely.box(*b)
    near = polys[polys.intersects(box)]
    if len(near):
        near.boundary.plot(ax=ax, color=BOUNDARY, lw=lw_poly)
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    f = C.settings(cfg)["spatial_mapping_figure"]
    mode = f["buildings"]
    if mode not in ("test_only", "all_buildings"):
        raise ValueError("appendix_S05_boundary_errors.spatial_mapping_figure.buildings must be test_only or all_buildings")
    colours = f["colours"]
    use_style()
    polys = read_favela_polygons(cfg, 2013)

    data, counts = {}, []
    for site in cfg.sites:
        g = C.main_outputs(cfg, site, geometry=True)
        g["outcome"] = C.outcome(g["is_favela"], g["pred_favela"])
        g["shown"] = g["is_test"].to_numpy() if mode == "test_only" else np.ones(len(g), bool)
        data[site] = g
        c = g[g["is_test"]].outcome.value_counts()
        counts.append(dict(site=site, n=len(g), n_test=int(g["is_test"].sum()),
                           **{f"test_{k}": int(c.get(k, 0)) for k in ("TP", "FN", "TN", "FP")},
                           test_error_rate=float(g.loc[g["is_test"], "outcome"].isin(["FN", "FP"]).mean())))
    write_table(pd.DataFrame(counts), out / "spatial_mapping_counts.csv")

    # ---- one PDF per site (for figure assembly)
    sdir = out / C.MAPPING_SITES_DIR
    sdir.mkdir(parents=True, exist_ok=True)
    for site, g in data.items():
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        draw_site(ax, g, polys, colours, float(f["polygon_lw_single"]))
        fig.savefig(sdir / f"{site}.pdf", bbox_inches="tight", dpi=int(f["raster_dpi"]))
        plt.close(fig)

    # ---- assembled figure
    rows = [[x for x in r if x in data] for r in f["panel_rows"]]
    rows = [r for r in rows if r]
    placed = {x for r in rows for x in r}
    extra = [x for x in cfg.sites if x not in placed]          # sites not listed in panel_rows
    if extra:
        rows.append(extra)
    ncol = max(len(r) for r in rows)
    W, H = f["size_in"]
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(rows), ncol, hspace=0.18, wspace=0.06, left=0.01, right=0.99, top=0.99,
                          bottom=0.09 if f["legend"] else 0.02)
    for i, r in enumerate(rows):
        j = 0
        while j < len(r):
            site = r[j]
            span = 1
            while j + span < len(r) and r[j + span] == site:
                span += 1
            ax = fig.add_subplot(gs[i, j:j + span])
            draw_site(ax, data[site], polys, colours, float(f["polygon_lw"]))
            ax.text(0.5, -0.03, cfg.site_name(site), transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
            j += span
    if f["legend"]:
        handles = [Patch(fc=colours[k], label=OUTCOME_LABEL[k]) for k in ("TP", "TN", "FN", "FP")]
        if mode == "test_only":
            handles.append(Patch(fc=NEUTRAL_BUILDING, label="training building (not tested)"))
        handles.append(Line2D([0], [0], color=BOUNDARY, lw=1.0, label="2013 favela polygon"))
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / C.FIGURE_MAPPING, dpi=int(f["raster_dpi"]))
    fig.savefig((out / C.FIGURE_MAPPING).with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(pd.DataFrame(counts).to_string(index=False))
    print(f"written {out / C.FIGURE_MAPPING} and {len(data)} site maps in {sdir}")


if __name__ == "__main__":
    main()
