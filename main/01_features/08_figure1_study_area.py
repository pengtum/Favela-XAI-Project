"""Step 8: Fig. 1a, study-area map with one split disk per site (element map).

Manuscript items
  * Fig. 1a: the ten sites, each split disk giving the per-class building
    count (left half favela, right half formal settlement; area proportional
    to count). Fig. 1 is assembled by hand from this and the other element
    maps.

Drawing
  Basemap tiles (optional, 07_fig1_basemap.py) and the municipal boundary in
  Web Mercator; for each site a disk at the centroid of its analysed
  footprints, left half favela (blue), right half formal settlement (green),
  half-disk radius proportional to sqrt(count); site label positions from
  config; a size key for 5,000 / 15,000 / 30,000 buildings per class.

Inputs
  outputs/main/01_features/site_counts.csv   products.site_counts (04_site_summary.py)
  paths.raw.municipality                     municipal boundary (Data.Rio)
  basemap cache                              07_fig1_basemap.py (optional)
  config.yaml: main_01_features.figure1

Outputs (outputs/main/01_features/)
  fig1_map_sites.pdf, fig1_map_sites.svg

Run from the repository root:
  python main/01_features/08_figure1_study_area.py
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Patch, Wedge

from favela_xai.config import load_config
from favela_xai.plotting_style import FAVELA_MAP, FORMAL_MAP

import figure1_common as F1

OUTLINE_COLOUR = "#6f6f6f"
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72)   # legibility on the basemap
LABEL_OFFSET_M = 600      # gap between a disk and its label (EPSG:3857 metres); layout


def label_anchor(x, y, r_fav, r_formal, where: str):
    """Text position and alignment of a site label placed below/above/left/right of its disk."""
    r = max(r_fav, r_formal)
    if where == "above":
        return x, y + r + LABEL_OFFSET_M, "center", "bottom"
    if where == "left":
        return x - r_fav - LABEL_OFFSET_M, y, "right", "center"
    if where == "right":
        return x + r_formal + LABEL_OFFSET_M, y, "left", "center"
    return x, y - r - LABEL_OFFSET_M, "center", "top"


def main() -> None:
    cfg = load_config()
    f1 = F1.settings(cfg)
    out_dir = cfg.output_dir(__file__)
    F1.apply_style(f1)

    counts = pd.read_csv(cfg.require_product("site_counts"))
    points = gpd.GeoDataFrame(counts, geometry=gpd.points_from_xy(counts.lon, counts.lat), crs=4326
                              ).to_crs(F1.WEB_MERCATOR)
    outline = F1.municipality_outline(cfg)
    ext = F1.map_extent(outline, f1)
    basemap = F1.load_basemap(cfg)

    r_max = float(f1["disk_max_radius_m"])
    n_max = np.sqrt(max(counts.n_fav.max(), counts.n_non.max()))
    radius = lambda n: r_max * np.sqrt(n) / n_max                      # noqa: E731 (area proportional to count)

    width = float(f1["map_width_in"])
    fig, ax = plt.subplots(figsize=(width, width / ext["aspect"]))
    if basemap is not None:
        img, image_extent, meta = basemap
        ax.imshow(img, extent=image_extent, zorder=0, interpolation="bilinear")
    else:
        print("no basemap cache (run 07_fig1_basemap.py to add the tiles); drawing the outline only")
    outline.boundary.plot(ax=ax, color=OUTLINE_COLOUR, lw=0.9, zorder=1, alpha=0.85)

    positions = f1.get("label_positions", {}) or {}
    labels = f1.get("map_labels", {}) or {}
    for r in points.itertuples():
        x, y = r.geometry.x, r.geometry.y
        rf, rn = radius(r.n_fav), radius(r.n_non)
        ax.add_patch(Wedge((x, y), rf, 90, 270, fc=FAVELA_MAP, ec="white", lw=0.7, zorder=4))   # left half: favela
        ax.add_patch(Wedge((x, y), rn, 270, 90, fc=FORMAL_MAP, ec="white", lw=0.7, zorder=4))   # right half: formal
        tx, ty, ha, va = label_anchor(x, y, rf, rn, positions.get(r.site, "below"))
        ax.text(tx, ty, labels.get(r.site, r.name), ha=ha, va=va, fontsize=8.6, color="#1a1a1a", zorder=5,
                bbox=LABEL_BOX)

    map_x, map_y = ext["map_x"], ext["map_y"]
    ax.set_xlim(*map_x)
    ax.set_ylim(*map_y)
    ax.set_aspect("equal")
    ax.axis("off")

    # Size key: outlined circles of the reference counts, anchored at a common bottom point
    refs = list(f1["disk_key_counts"])
    kx = map_x[0] + (map_x[1] - map_x[0]) * 0.085
    ky = map_y[0] + (map_y[1] - map_y[0]) * 0.075
    r_big = radius(refs[-1])
    for c in refs:
        rr = radius(c)
        ax.add_patch(Circle((kx, ky + rr), rr, fill=False, ec=OUTLINE_COLOUR, lw=0.8, zorder=4))
        ax.plot([kx, kx + r_big + 900], [ky + 2 * rr, ky + 2 * rr], color="#9a9a9a", lw=0.5, zorder=4)
        ax.text(kx + r_big + 1050, ky + 2 * rr, f"{c:,}", ha="left", va="center", fontsize=6, color="#444444",
                zorder=5, bbox=LABEL_BOX)
    ax.text(kx, ky + 2 * r_big + 1500, "buildings per class", ha="center", va="bottom", fontsize=6.3,
            color="#222222", zorder=5, bbox=LABEL_BOX)

    # Class legend above the map
    ax.legend(handles=[Patch(fc=FAVELA_MAP, label="Favela"), Patch(fc=FORMAL_MAP, label="Formal settlement")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False,
              handlelength=1.4, columnspacing=1.6)
    if basemap is not None and f1.get("basemap_attribution_on_map", False):
        ax.text(0.995, 0.005, f1["basemap_attribution"], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.5, color="#666666", zorder=6)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for p in F1.save_element(fig, out_dir, "fig1_map_sites", f1):
        print(f"written {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
