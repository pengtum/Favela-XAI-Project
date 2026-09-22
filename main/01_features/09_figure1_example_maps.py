"""Step 9: Fig. 1a Rocinha input map and Fig. 1c indicator maps (element maps).

Manuscript items
  * Fig. 1a: footprints of the example district (Rocinha) labelled favela
    (blue) or formal settlement (green) by the official 2013 cadastre (dashed
    line).
  * Fig. 1c: one representative indicator per morphometric dimension
    mapped across the example district: building area, elongation, mean
    interbuilding distance, number of tessellation neighbours, average
    weighted distance.
  Fig. 1 is assembled by hand from these element maps.

Drawing
  Buildings whose centroid lies in the example window (squared and widened by
  example_window_pad); the 2013 favela boundary dashed on the input map and
  solid on the indicator maps. Indicator maps use a ramp to the colour of the
  indicator's dimension (as in Figs 3-4), clipped to the 2nd-98th
  percentiles, with a tag naming the dimension.

Inputs
  outputs/main/01_features/<site>_buildings_features.gpkg   products.features_gpkg (03_indicators.py)
  paths.raw.favelas_2013                                    Limite Favelas 2013 (Data.Rio)
  config.yaml: main_01_features.figure1 (example_site, example_window, feature_maps, ...)

Outputs (outputs/main/01_features/)
  fig1_map_input.pdf / .svg
  fig1_map_feat_<indicator>.pdf / .svg   one per indicator of feature_maps

Run from the repository root:
  python main/01_features/09_figure1_example_maps.py
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from favela_xai.config import load_config
from favela_xai.features import BY_KEY, check_registry, dimension_of
from favela_xai.io import read_favela_polygons, read_site_features
from favela_xai.labels import FAVELA
from favela_xai.plotting_style import BOUNDARY, DIMENSION_COLOURS, FAVELA_MAP, FORMAL_MAP

import figure1_common as F1

EDGE = dict(ec="white", lw=0.22)      # white outline of every footprint
BOUNDARY_LW = 1.1
MAP_SIZE_DEFAULT = (3.0, 3.3)


def square_bounds(window, pad: float):
    """Square around the centre of a window, side = longer window side x (1 + pad)."""
    cx, cy = (window[0] + window[2]) / 2, (window[1] + window[3]) / 2
    half = max(window[2] - window[0], window[3] - window[1]) / 2 * (1 + pad)
    return cx - half, cy - half, cx + half, cy + half


def dimension_ramp(colour: str) -> LinearSegmentedColormap:
    """Light tint (16%) -> mid tint (52%) -> full colour of a morphometric dimension."""
    r, g, b = mcolors.to_rgb(colour)
    tint = lambda a: (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)     # noqa: E731
    return LinearSegmentedColormap.from_list("dimension_ramp", [tint(0.16), tint(0.52), colour])


def draw_boundary(ax, line, bounds, dashed: bool) -> None:
    gpd.GeoSeries([line]).plot(ax=ax, color=BOUNDARY, lw=BOUNDARY_LW, zorder=6,
                               linestyle=(0, (4, 2.2)) if dashed else "-")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    cfg = load_config()
    f1 = F1.settings(cfg)
    out_dir = cfg.output_dir(__file__)
    F1.apply_style(f1)
    check_registry(cfg.get("indicators.dimension_sizes"))

    site = f1["example_site"]
    window = f1["example_window"]
    if site not in cfg.sites:                    # example site not in this run: first site, whole window
        site = cfg.sites[0]
        window = cfg.site_window(site)
        print(f"example site not processed in this mode; drawing {cfg.site_name(site)} instead")
    features = list(f1["feature_maps"])
    dims = [dimension_of(k) for k in features]
    if len(set(dims)) != len(dims):
        print(f"note: the indicator maps do not cover five different dimensions: {dims}")

    # Buildings of the example window (selected by centroid)
    bounds = square_bounds(window, float(f1["example_window_pad"]))
    g = read_site_features(cfg, site, columns=["label"] + features)
    c = g.geometry.centroid
    g = g[(c.x >= bounds[0]) & (c.x <= bounds[2]) & (c.y >= bounds[1]) & (c.y <= bounds[3])].reset_index(drop=True)
    fav = (g["label"] == FAVELA).to_numpy()
    print(f"{cfg.site_name(site)} example window: {len(g):,} buildings ({fav.sum():,} favela, "
          f"{(~fav).sum():,} formal settlement)")

    # Official 2013 boundary: edges of the polygons that intersect the window
    box = shapely.box(*bounds)
    polys = read_favela_polygons(cfg, 2013).to_crs(g.crs)
    near = np.asarray(polys.geometry.values[polys.intersects(box).to_numpy()], dtype=object)
    line = shapely.union_all(shapely.boundary(near)).intersection(box)

    size = tuple(f1.get("map_size_in", MAP_SIZE_DEFAULT))

    # ---- input map (Fig. 1a, right): labels from the 2013 cadastre, dashed boundary
    fig, ax = plt.subplots(figsize=size)
    g[~fav].plot(ax=ax, color=FORMAL_MAP, **EDGE)
    g[fav].plot(ax=ax, color=FAVELA_MAP, **EDGE)
    dashed = bool(f1.get("input_boundary_dashed", True))
    draw_boundary(ax, line, bounds, dashed=dashed)
    ax.legend(handles=[Patch(fc=FAVELA_MAP, label="Favela"), Patch(fc=FORMAL_MAP, label="Formal settlement"),
                       Line2D([0], [0], color=BOUNDARY, lw=BOUNDARY_LW, linestyle=(0, (4, 2.2)) if dashed else "-",
                              label="Official 2013 favela boundary")],
              loc="lower left", fontsize=6.0, frameon=False)
    for p in F1.save_element(fig, out_dir, "fig1_map_input", f1):
        print(f"written {p}")
    plt.close(fig)

    # ---- indicator maps (Fig. 1c): one per morphometric dimension
    lo_pct, hi_pct = f1["colour_percentiles"]
    for key, dim in zip(features, dims):
        colour = DIMENSION_COLOURS[dim]
        cmap = dimension_ramp(colour)
        v = g[key].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, [lo_pct, hi_pct])
        fig, ax = plt.subplots(figsize=size)
        g.plot(ax=ax, column=key, cmap=cmap, vmin=lo, vmax=hi, **EDGE)
        draw_boundary(ax, line, bounds, dashed=False)
        # dimension tag (dot and name, top left)
        ax.scatter([0.06], [0.94], s=60, c=[colour], transform=ax.transAxes, clip_on=False, zorder=10,
                   edgecolors="white", linewidths=0.6)
        ax.text(0.115, 0.94, dim, transform=ax.transAxes, fontsize=6.6, va="center", ha="left", color=colour,
                fontweight="bold", zorder=10, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cb = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.02, aspect=22)
        cb.set_label(f"{BY_KEY[key].full_name} (low to high)", fontsize=6.6)
        cb.ax.tick_params(labelsize=5.6)
        cb.outline.set_linewidth(0.5)
        cb.set_ticks([lo, hi])
        for p in F1.save_element(fig, out_dir, f"fig1_map_feat_{key}", f1):
            print(f"written {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
