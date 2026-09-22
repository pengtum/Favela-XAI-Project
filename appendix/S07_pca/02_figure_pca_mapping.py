"""SI Fig. S10: spatial mapping of the pooled attribution PC1, thresholded per site.

One map per site in ``appendix_S07_pca.map_sites``: every building of the
site, orange when favela-like (PC1 above the site's ROC-Youden threshold,
01_pc1_youden_thresholds.py) and grey otherwise, with the official 2013 favela
polygons outlined in black, so that favela-like buildings inside and outside
the polygons can be seen (SI Section S7). The per-site maps are the panels of
SI Fig. S10 (the overview map of Rio de Janeiro is not drawn here); a panel
grid with site names and a legend is also written.

Inputs
* outputs/appendix/S07_pca/pc1_favela_like.parquet (01_pc1_youden_thresholds.py);
* products.shap_gpkg (main/02_classification): building geometry, in the row
  order to which the ``row`` column of the PCA scores refers;
* official 2013 favela polygons (paths.raw.favelas_2013).
Outputs (outputs/appendix/S07_pca/)
* Figure_R1_PCA_mapping_<site>.pdf: one map per site (panels of SI Fig. S10);
* Figure_R1_PCA_mapping.pdf: all mapped sites in one panel grid with a legend.

Run from the repository root (after 01_pc1_youden_thresholds.py):
    python appendix/S07_pca/02_figure_pca_mapping.py
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from favela_xai.config import ConfigError, load_config
from favela_xai.io import read_favela_polygons, read_site_shap
from favela_xai.plotting_style import use_style

BUILDINGS = "pc1_favela_like.parquet"
GRID_FIGURE = "Figure_R1_PCA_mapping.pdf"
SITE_FIGURE = "Figure_R1_PCA_mapping_{site}.pdf"


def site_layers(cfg, site: str, flags: pd.DataFrame, polygons):
    """Buildings of one site with the favela-like flag, the 2013 polygons around them
    and the map extent."""
    g = read_site_shap(cfg, site, columns=["label"])
    f = flags[flags["site"] == site]
    like = np.zeros(len(g), dtype=bool)
    rows = f["row"].to_numpy(int)
    if len(rows) and rows.max() >= len(g):
        raise ConfigError(f"{site}: PCA scores refer to rows beyond the main/02 table; re-run main/05 and step 01")
    like[rows] = f["favela_like"].to_numpy(bool)
    g = g.assign(favela_like=like)
    if len(f) < len(g):
        print(f"  note: {site}: {len(g) - len(f)} buildings without PCA scores are drawn as other buildings")

    pad = float(cfg.get("appendix_S07_pca.map_pad_fraction"))
    xmin, ymin, xmax, ymax = g.total_bounds
    p = pad * max(xmax - xmin, ymax - ymin)
    extent = (xmin - p, ymin - p, xmax + p, ymax + p)
    near = polygons.cx[extent[0]:extent[2], extent[1]:extent[3]]
    return g, near, extent


def draw_map(ax, g, polygons, extent, style: dict) -> None:
    """One site map: other buildings grey, favela-like buildings orange, 2013 outlines black."""
    colours = style["map_colours"]
    g[~g["favela_like"]].plot(ax=ax, color=colours["other"], ec="none", rasterized=True)
    g[g["favela_like"]].plot(ax=ax, color=colours["favela_like"], ec="none", rasterized=True)
    if len(polygons):
        polygons.boundary.plot(ax=ax, color=colours["boundary"], lw=float(style["boundary_linewidth"]))
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():                       # framed panels
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(0.6)


def legend_handles(style: dict) -> list:
    colours = style["map_colours"]
    return [Patch(fc=colours["favela_like"], label="favela-like (PC1$_{\\mathrm{deep}}$ above the site's threshold)"),
            Patch(fc=colours["other"], label="other buildings"),
            Line2D([0], [0], color=colours["boundary"], lw=1.0, label="official 2013 favela boundary")]


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    style = cfg.section("appendix_S07_pca")
    path = out / BUILDINGS
    if not path.exists():
        raise ConfigError(f"{path} does not exist; run 01_pc1_youden_thresholds.py first")
    flags = pd.read_parquet(path)
    sites = [s for s in style["map_sites"] if s in cfg.sites]
    if not sites:
        raise ConfigError("none of appendix_S07_pca.map_sites is among the sites processed")
    polygons = read_favela_polygons(cfg, 2013)
    dpi = int(style["raster_dpi"])

    use_style()
    layers = {s: site_layers(cfg, s, flags, polygons) for s in sites}

    # ---- one map per site (panels of SI Fig. S10) ------------------------------
    for site in sites:
        g, near, extent = layers[site]
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        draw_map(ax, g, near, extent, style)
        fig.tight_layout()
        fig.savefig(out / SITE_FIGURE.format(site=site), dpi=dpi)      # buildings rasterised at dpi, outlines vector
        plt.close(fig)
        share = g["favela_like"].mean()
        print(f"  {cfg.site_name(site):24s} {len(g):>7d} buildings, {100 * share:5.1f}% favela-like")

    # ---- panel grid with site names and legend ----------------------------------
    ncols = int(style["grid_columns"])
    nrows = math.ceil((len(sites) + 1) / ncols)          # one extra cell for the legend
    width_in = 183 / 25.4                                  # Nature double column
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, width_in / ncols * nrows * 1.08), squeeze=False)
    axes = axes.ravel()
    for ax, site in zip(axes, sites):
        g, near, extent = layers[site]
        draw_map(ax, g, near, extent, style)
        ax.set_title(cfg.site_name(site), fontsize=7.5, pad=3)
    for ax in axes[len(sites):]:
        ax.axis("off")
    axes[len(sites)].legend(handles=legend_handles(style), loc="center", fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / GRID_FIGURE, dpi=dpi)
    plt.close(fig)
    print(f"[saved] {out / GRID_FIGURE} and {len(sites)} per-site maps")


if __name__ == "__main__":
    main()
