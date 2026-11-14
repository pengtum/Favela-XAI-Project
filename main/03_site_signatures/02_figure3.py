"""Fig. 3: spatial and statistical anatomy of the building-scale favela signature.

Manuscript items
* Fig. 3 (Results 2.2):
  (a) every favela building of the eight other sites coloured by its dominant
      feature, shaded by the indicator's morphometric dimension; non-favela
      buildings grey;
  (b) Rocinha and Barão: dominant-feature map beside per-building Tree-SHAP maps
      of mean interbuilding distance and squareness (diverging colour, orange =
      toward favela); each SHAP map carries an inset with that favela's SHAP
      dependence (dots = buildings, bold line = this favela, dashed = pooled over
      all ten sites).
* Results 2.2: the dominant feature is mostly a spatial-distribution indicator,
  with squareness across much of Barão and the perimeter wall in Grotão de
  Costa Barros; the interbuilding-distance dependence reverses between Barão
  (toward favela) and Rocinha (toward formal). The binned dependence curves
  are written to figure3_dependence_curves.csv (the full per-site dependence
  is SI Figs S7-S8, appendix/S06_attribution_analyses).

The SVG output keeps the text editable for final layout. The published Fig. 3
was finished by hand from it; for example, the two panel headings, the legend
title, the 'other feature' and 'non-favela' legend entries and the inset titles
and key were removed, the two colour bars share the label 'Feature importance',
the first inset carries the axis labels 'feature value' and 'shap value', and
Grotão de Costa Barros is labelled 'Grotão' in panel a.

Settings (config.yaml)
* importance.dominant_feature: argmax of the 25 signed SHAP values.
* main_03_site_signatures: sites and indicators of panel b, panel-a order and
  titles, zoom window, number of colour-coded features, colour limit of the
  SHAP maps, the binning of the insets, and the line drawn on the SHAP maps
  (outline: official_2013 | label_hull).
* signed_distance.*: which 2013 polygons and which boundary are drawn (the same
  line as the signed distance of Methods 4.8).

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg
      (label, the two panel-b indicators, the 25 shap_<indicator> columns, geometry)
  data/raw/Limite_Favelas_2013/Limite_Favelas_2013.shp  (outline of panel b)

Outputs (outputs/main/03_site_signatures/)
  figure3.pdf, figure3.svg           the figure
  figure3_dependence_curves.csv      binned SHAP dependence of the panel-b
                                     indicators: pooled and per site
  figure3_legend_features.csv        the colour-coded dominant features and their
                                     favela-building counts over all sites

Run (from the repository root)
  python main/03_site_signatures/02_figure3.py
"""
from __future__ import annotations

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# plotting_style selects the non-interactive Agg backend before pyplot is imported
from favela_xai.plotting_style import DIMENSION_COLOURS, FAVELA_SHAP, FORMAL_SHAP, NEUTRAL_BUILDING, SHAP_CMAP

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from favela_xai.attribution import dominant_feature
from favela_xai.boundary import binned_mean, boundary_lines, nearby_polygons
from favela_xai.config import load_config
from favela_xai.features import SHAP_COLUMNS, dimension_of, label_of
from favela_xai.io import read_favela_polygons, read_site_shap, write_table
from favela_xai.labels import label_to_int

warnings.filterwarnings("ignore", message="The GeoDataFrame you are attempting to plot is empty")

# ----------------------------------------------------------------------------- style
# Figure-specific rcParams (large canvas); colours shared with the other
# figures come from plotting_style.
FIG_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 11, "axes.linewidth": 0.7,
    "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
}
# Per-indicator shades within the dimension hues (same palette as ED Fig. 2);
# indicators not listed are drawn in their dimension colour.
FEATURE_COLOURS = {
    "mean_interbuilding_distance": "#14746A", "adjacency": "#7FCBC0", "CAR": "#2A9D8F",
    "neighbour_dist": "#0E5249", "perimeter_wall": "#E2700E", "squareness": "#B5341F",
    "cwa": "#F2B84B", "compact_weighted_axis": "#9C5418", "shared_walls": "#8C4A28",
    "rectangularity": "#D99441", "avg_weighted_dist": "#4C9F70", "fractal": "#1E7A45",
    "num_neighbors": "#9A4D8E", "cell_alignment": "#C58EBC", "tes_area": "#6F3566",
    "area": "#0F4D92", "perimeter": "#5B8FC0",
}
# Legend order of the dimensions and their short names (legend band between a and b)
LEGEND_DIMENSION_ORDER = ["Spatial distribution", "Shape", "Configuration", "Connectivity", "Dimension"]
DIMENSION_ABBREV = {"Spatial distribution": "spatial", "Shape": "shape", "Connectivity": "connectivity",
                    "Configuration": "config", "Dimension": "dimension"}
NON_FAVELA = NEUTRAL_BUILDING      # '#e8e8e8', non-favela buildings
OTHER_FEATURE = "#c8c8c8"          # favela buildings whose dominant feature is not colour-coded
TREND = "#14746A"                  # inset: this favela
POOLED = "#9e9e9e"                 # inset: pooled over all sites
OUTLINE = "#111111"
HERO_COLOURS = ["#1f77b4", "#d62728"]   # row labels of panel b
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten"}


def feature_colour(feature: str) -> str:
    return FEATURE_COLOURS.get(feature, DIMENSION_COLOURS[dimension_of(feature)])


# ----------------------------------------------------------------------------- data
def load_site(cfg, site: str, hero_features: list[str]) -> gpd.GeoDataFrame:
    """Buildings of one site with label, panel-b indicators, 25 SHAP values and dominant feature."""
    g = read_site_shap(cfg, site, columns=["label"] + hero_features + SHAP_COLUMNS)
    g = g[g.geometry.notna() & g.geometry.is_valid & ~g.geometry.is_empty].copy()
    for c in hero_features + SHAP_COLUMNS:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["is_favela"] = label_to_int(g["label"]).astype(bool)
    g["driver"] = dominant_feature(g).to_numpy()
    return g


# ----------------------------------------------------------------------------- map helpers
def dominant_map(ax, g: gpd.GeoDataFrame, top: list[str]) -> None:
    """Non-favela grey, favela buildings coloured by their dominant feature."""
    fav = g[g["is_favela"]]
    g[~g["is_favela"]].plot(ax=ax, color=NON_FAVELA, ec="none", rasterized=True)
    fav[~fav["driver"].isin(top)].plot(ax=ax, color=OTHER_FEATURE, ec="none", rasterized=True)
    for f in top:
        sub = fav[fav["driver"] == f]
        if len(sub):
            sub.plot(ax=ax, color=feature_colour(f), ec="none", rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")


def square_bounds(bounds, pad: float = 0.04):
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) / 2 * (1 + pad)
    return (cx - half, cy - half, cx + half, cy + half)


def set_square(ax, sqb) -> None:
    ax.set_xlim(sqb[0], sqb[2])
    ax.set_ylim(sqb[1], sqb[3])


def frame_map(ax, colour: str = "#5a5a5a", lw: float = 0.9) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(colour)
        sp.set_linewidth(lw)


def outline_geometry(g: gpd.GeoDataFrame, polygons_2013, cfg, opts: dict):
    """Line drawn on the SHAP maps (config main_03_site_signatures.outline)."""
    if opts["outline"] == "official_2013":
        # The boundary from which the signed distance of Methods 4.8 is measured
        sd = cfg.section("signed_distance")
        near = nearby_polygons(polygons_2013.to_crs(g.crs), g.total_bounds, float(sd["polygon_search_pad_m"]))
        return boundary_lines(near, bool(sd["dissolve_polygons"])) if len(near) else None
    if opts["outline"] == "label_hull":
        # Hull of the favela-labelled footprints
        h = opts["label_hull"]
        fav = g.loc[g["is_favela"]].geometry
        if not len(fav):
            return None
        grown = fav.simplify(float(h["simplify_m"])).buffer(float(h["grow_m"]))
        region = shapely.union_all(np.asarray(grown.values, dtype=object))
        region = region.buffer(-float(h["shrink_m"])).simplify(float(h["final_simplify_m"]))
        return region.boundary
    raise ValueError(f"unknown outline '{opts['outline']}' (official_2013 or label_hull)")


def emptiest_corner(g: gpd.GeoDataFrame, sqb) -> list[float]:
    """Inset position in the map quadrant with the fewest building centroids."""
    minx, miny, maxx, maxy = sqb
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    c = g.geometry.centroid
    x, y = c.x.to_numpy(), c.y.to_numpy()
    m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    x, y = x[m], y[m]
    counts = {"tl": int(((x < cx) & (y > cy)).sum()), "tr": int(((x > cx) & (y > cy)).sum()),
              "bl": int(((x < cx) & (y < cy)).sum()), "br": int(((x > cx) & (y < cy)).sum())}
    boxes = {"tl": [0.02, 0.585, 0.43, 0.40], "tr": [0.55, 0.585, 0.43, 0.40],
             "bl": [0.02, 0.02, 0.43, 0.40], "br": [0.55, 0.02, 0.43, 0.40]}
    return boxes[min(counts, key=counts.get)]


def dependence_inset(ax, g: gpd.GeoDataFrame, name: str, f: str, dep: dict, sqb, rng, n_points: int,
                     min_n: int) -> None:
    """SHAP dependence of indicator f in one favela: dots, this favela's binned mean, pooled binned mean."""
    D = dep[f]
    iax = ax.inset_axes(emptiest_corner(g, sqb), zorder=20)
    iax.set_facecolor("white")
    iax.patch.set_alpha(0.45)
    iax.axhline(0, color="#cfcfcf", lw=0.6, ls="--", zorder=1)
    iax.plot(D["centres"], D["pooled"], color=POOLED, lw=1.3, ls=(0, (3, 2)), zorder=3)
    # Random draws in a fixed order (formal, then favela) so that the dots are reproducible
    for is_fav, colour in [(False, FORMAL_SHAP), (True, FAVELA_SHAP)]:
        sub = g[g["is_favela"] == is_fav]
        xs, ys = sub[f].to_numpy(), sub["shap_" + f].to_numpy()
        m = np.isfinite(xs) & np.isfinite(ys) & (xs >= D["lo"]) & (xs <= D["hi"])
        xs, ys = xs[m], ys[m]
        if len(xs) > n_points:
            ii = rng.choice(len(xs), n_points, replace=False)
            xs, ys = xs[ii], ys[ii]
        iax.scatter(xs, ys, s=2.4, c=colour, alpha=0.20, lw=0, rasterized=True, zorder=2)
    yv = binned_mean(g[f].to_numpy(), g["shap_" + f].to_numpy(), D["edges"], min_n)
    ok = np.isfinite(yv)
    iax.plot(D["centres"][ok], yv[ok], color=TREND, lw=1.9, marker="o", ms=2.4, zorder=5)
    iax.set_xlim(D["lo"], D["hi"])
    iax.tick_params(labelsize=6.5, length=2, pad=1)
    iax.locator_params(axis="both", nbins=3)
    for sp in iax.spines.values():
        sp.set_color("#777")
        sp.set_linewidth(0.6)
    iax.set_title(f"{name} · SHAP dep.", fontsize=7, pad=1.5, color=feature_colour(f))


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    opts = cfg.section("main_03_site_signatures")
    ins = opts["inset"]
    hero_features = list(opts["hero_features"])
    sites = list(cfg.sites)

    heroes = [s for s in opts["hero_sites"] if s in sites]
    if not heroes:                      # the panel-b sites are not among the processed sites
        heroes = sites[:1]
        print(f"note: panel-b sites not available; drawing {heroes} instead")
    thumbs = [s for s in opts["panel_a_order"] if s in sites and s not in heroes]
    thumbs += [s for s in sites if s not in thumbs and s not in heroes]

    print("loading the buildings, the 25 SHAP values and the dominant feature ...")
    data = {}
    for s in sites:
        data[s] = load_site(cfg, s, hero_features)
        print(f"  {cfg.site_name(s):24s} {len(data[s]):>7d}")

    # Colour-coded dominant features: the most frequent over all favela buildings,
    # grouped by dimension for the legend
    allfav = pd.concat([data[s].loc[data[s]["is_favela"], "driver"] for s in sites])
    counts = allfav.value_counts()
    top = sorted(counts.head(int(opts["n_legend_features"])).index,
                 key=lambda f: (LEGEND_DIMENSION_ORDER.index(dimension_of(f)), -int(counts[f])))
    print("colour-coded dominant features:", [label_of(f) for f in top])
    write_table(pd.DataFrame({"feature": top, "label": [label_of(f) for f in top],
                              "dimension": [dimension_of(f) for f in top],
                              "n_favela_buildings": [int(counts[f]) for f in top],
                              "share_of_favela_buildings": [float(counts[f]) / len(allfav) for f in top]}),
                out / "figure3_legend_features.csv")

    # Symmetric colour limits of the SHAP maps over the panel-b sites
    pct = float(opts["shap_colour_percentile"])
    norms = {}
    for f in hero_features:
        v = float(np.nanpercentile(np.abs(np.concatenate([data[s]["shap_" + f].to_numpy() for s in heroes])), pct))
        norms[f] = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    # Dependence curves: equal bins between pooled percentiles of the indicator value
    dep, curve_rows = {}, []
    for f in hero_features:
        allx = np.concatenate([data[s][f].to_numpy() for s in sites])
        ally = np.concatenate([data[s]["shap_" + f].to_numpy() for s in sites])
        lo, hi = np.nanpercentile(allx, ins["percentile_range"])
        edges = np.linspace(lo, hi, int(ins["n_bins"]) + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        pooled = binned_mean(allx, ally, edges, int(ins["min_buildings_per_bin"]))
        dep[f] = dict(lo=lo, hi=hi, edges=edges, centres=centres, pooled=pooled)
        per_site = {"pooled": pooled}
        per_site.update({s: binned_mean(data[s][f].to_numpy(), data[s]["shap_" + f].to_numpy(), edges,
                                        int(ins["min_buildings_per_bin"])) for s in sites})
        for who, curve in per_site.items():
            for b in range(len(centres)):
                curve_rows.append(dict(feature=f, label=label_of(f), curve=who, bin_lo=edges[b], bin_hi=edges[b + 1],
                                       bin_centre=centres[b], mean_shap=curve[b]))
    write_table(pd.DataFrame(curve_rows), out / "figure3_dependence_curves.csv")

    polygons_2013 = read_favela_polygons(cfg, 2013) if opts["outline"] == "official_2013" else None
    rng = np.random.default_rng(int(ins["seed"]))
    zoom = opts.get("zoom_windows") or {}
    short_names = opts.get("short_names") or {}        # panel-a titles (layout)

    plt.rcParams.update(FIG_RC)
    fig = plt.figure(figsize=(17.5, 21.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.60, 0.14, 1.0], hspace=0.02)

    # ===== (a) dominant feature of every favela building, the other sites =====
    n_rows_a = max(2, math.ceil(len(thumbs) / 4))
    gtop = outer[0].subgridspec(n_rows_a, 4, hspace=0.10, wspace=0.02)
    for k, s in enumerate(thumbs):
        axm = fig.add_subplot(gtop[k // 4, k % 4])
        dominant_map(axm, data[s], top)
        set_square(axm, square_bounds(tuple(data[s].total_bounds)))
        frame_map(axm)
        axm.set_title(short_names.get(s, cfg.site_name(s)), fontsize=13, pad=4)
    fig.text(0.012, 0.978, "a", fontsize=20, fontweight="bold")
    n_words = NUMBER_WORDS.get(len(thumbs), str(len(thumbs)))
    fig.text(0.5, 0.978, f"Signature anatomy across the other {n_words} favelas: dominant feature per building",
             ha="center", fontsize=13.5, fontweight="bold")

    # ===== legend band between a and b =====
    axl = fig.add_subplot(outer[1])
    axl.axis("off")
    handles = [Patch(fc=feature_colour(f), label=f"{label_of(f)} · {DIMENSION_ABBREV[dimension_of(f)]}") for f in top]
    handles += [Patch(fc=OTHER_FEATURE, label="other feature"), Patch(fc=NON_FAVELA, label="non-favela")]
    axl.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.24), ncol=8, fontsize=11.5,
               title="dominant feature  ·  morphometric dimension", title_fontsize=12.5,
               handlelength=1.2, columnspacing=1.3, handletextpad=0.5)

    # ===== (b) the panel-b favelas in detail, with dependence insets =====
    ncol = 1 + len(hero_features)
    gb = outer[2].subgridspec(len(heroes) + 1, ncol, height_ratios=[1] * len(heroes) + [0.085],
                              hspace=0.045, wspace=0.025)
    col_titles = ["dominant feature"] + [f"SHAP: {label_of(f)}" for f in hero_features]
    for i, s in enumerate(heroes):
        g = data[s]
        name = cfg.site_name(s)
        if s in zoom:
            xmin, ymin, xmax, ymax = zoom[s]
            bnd = (xmin, ymin, xmax, ymax)
        else:
            bnd = tuple(g.total_bounds)
        sqb = square_bounds(bnd)

        ax = fig.add_subplot(gb[i, 0])
        dominant_map(ax, g, top)
        set_square(ax, sqb)
        frame_map(ax)
        ax.text(-0.045, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=17,
                fontweight="bold", color=HERO_COLOURS[i % len(HERO_COLOURS)])
        if i == 0:
            ax.set_title(col_titles[0], fontsize=15, fontweight="bold", pad=6)

        line = outline_geometry(g, polygons_2013, cfg, opts)
        for j, f in enumerate(hero_features):
            ax = fig.add_subplot(gb[i, 1 + j])
            g.plot(ax=ax, column="shap_" + f, cmap=SHAP_CMAP, norm=norms[f], ec="none", rasterized=True)
            if line is not None and not line.is_empty:
                gpd.GeoSeries([line], crs=g.crs).plot(ax=ax, color=OUTLINE, lw=0.6, zorder=5)
            ax.set_aspect("equal", adjustable="datalim")
            set_square(ax, sqb)
            frame_map(ax)
            if i == 0:
                ax.set_title(col_titles[1 + j], fontsize=15, fontweight="bold", color=feature_colour(f), pad=6)
            dependence_inset(ax, g, name, f, dep, sqb, rng, int(ins["points_per_class"]),
                             int(ins["min_buildings_per_bin"]))

    # bottom strip: colour bars under the SHAP-map columns, inset key under the first column
    for j, f in enumerate(hero_features):
        axc = fig.add_subplot(gb[len(heroes), 1 + j])
        axc.axis("off")
        cax = axc.inset_axes([0.16, 0.45, 0.68, 0.32])
        sm = mpl.cm.ScalarMappable(cmap=SHAP_CMAP, norm=norms[f])
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_visible(False)
        cb.set_label(f"SHAP of {label_of(f)}  ($\\leftarrow$ formal  |  favela $\\rightarrow$)", fontsize=10.5)
    axn = fig.add_subplot(gb[len(heroes), 0])
    axn.axis("off")
    axn.text(0.0, 0.6, "inset = SHAP dependence\ndot = building (orange favela / blue non-favela)\n"
             "bold = this favela · dashed = pooled, all sites", ha="left", va="center", fontsize=8.5, color="#555")
    fig.text(0.012, 0.604, "b", fontsize=20, fontweight="bold")
    lead = "Two contrasting favelas" if len(heroes) == 2 else "Favela"
    fig.text(0.5, 0.606, f"{lead} in detail: spatial and statistical heterogeneity of feature importance",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.996, top=0.952, bottom=0.008)

    for ext in ("pdf", "svg"):
        fig.savefig(out / f"figure3.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)
    print(f"[saved] {out / 'figure3.pdf'} (+ .svg)")


if __name__ == "__main__":
    main()
