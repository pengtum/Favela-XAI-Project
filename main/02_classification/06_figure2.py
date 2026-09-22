"""Step 6: Fig. 2 (favela-detection performance), panels a, b and c.

Manuscript items
  * Fig. 2a: building-level classification outcome of every building (training and
    held-out test)
    for two representative favelas (Faz Quem Quer, Morro da Guaiba), with the
    official 2013 favela boundary.
  * Fig. 2b: per-site held-out test accuracy, ordered high to low, and the
    ten-site mean (0.933).
  * Fig. 2c: misclassification rate by signed distance to the 2013 favela
    boundary (negative inside, positive outside; 6 m bins) from stratified
    five-fold cross-validation (XGBoost, 100 trees, seed 432): all sites with
    Wilson 95% confidence interval and the 10th-90th percentile of the site
    rates (site bins with fewer than 30 buildings omitted).
  The three panels are drawn in one matplotlib figure (Arial); the final
  typesetting of the manuscript figure is done by hand.

Panel a (config main_02_classification.figure2)
  panel_a_buildings (all_buildings | test_only): every building coloured by its prediction
  (default, as published), or held-out test buildings coloured and training buildings grey.
  misclassified_colours (by_class | single): orange for favela buildings predicted formal and
  red for formal-settlement buildings predicted favela, or orange for both.
  Outline: edges of the 2013 favela polygons, clipped to the site extent.
Panel b: accuracy column of table_S3_site_metrics.csv (02_table_S3.py); dashed line = mean.
Panel c: 6 m bins from -90 to +90 m (boundary_errors.bin_width_m, half_range_m); bins with
  fewer than 30 buildings are left empty, per site and pooled (min_buildings_per_site_bin);
  Wilson interval (wilson_z); grey band = 10th-90th percentile of the site rates
  (site_band_percentiles).

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg   (01_train_models.py): label,
                                                              predicted_label, is_test
  outputs/main/02_classification/table_S3_site_metrics.csv    (02_table_S3.py)
  outputs/main/02_classification/oof_predictions.parquet      (03_oof_predictions.py)
  paths.raw.favelas_2013                                      Limite Favelas 2013 (Data.Rio)

Outputs (outputs/main/02_classification/)
  figure2.pdf            183 mm wide, panels a-c
  figure2c_values.csv    per 6 m bin: buildings, pooled rate, Wilson interval, site 10th/90th
                         percentiles, number of sites with the bin

Run from the repository root:
  python main/02_classification/06_figure2.py
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from favela_xai.config import load_config
from favela_xai.io import read_favela_polygons, read_site_shap, write_table
from favela_xai.labels import FAVELA
from favela_xai.plotting_style import (BOUNDARY, FAVELA_MAP, FORMAL_MAP, MISCLASSIFIED, MISCLASSIFIED_FORMAL,
                                       NEUTRAL_BUILDING, figure_size, save_figure, use_style)
from favela_xai.stats import wilson_interval

# Colours of panels b and c
DOT = "#2F4B5E"
STEM = "#dfe2e6"
MEAN_LINE = "#9aa0a6"
RATE = "#E8752A"
SITE_BAND = "#DDE4E9"
MUTED_TEXT = "#5A6875"


# -----------------------------------------------------------------------------
# Panel a: classification outcome maps
# -----------------------------------------------------------------------------
def outcome_map(ax, cfg, site: str, polygons: gpd.GeoDataFrame, f2: dict) -> None:
    g = read_site_shap(cfg, site, columns=["label", "predicted_label", "is_test"])
    fav = (g["label"] == FAVELA).to_numpy()
    pred_fav = (g["predicted_label"] == FAVELA).to_numpy()
    shown = g["is_test"].astype(bool).to_numpy() if f2["panel_a_buildings"] == "test_only" else np.ones(len(g), bool)
    by_class = f2.get("misclassified_colours", "by_class") == "by_class"

    layers = [(~shown, NEUTRAL_BUILDING),
              (shown & fav & pred_fav, FAVELA_MAP),
              (shown & ~fav & ~pred_fav, FORMAL_MAP),
              (shown & fav & ~pred_fav, MISCLASSIFIED),
              (shown & ~fav & pred_fav, MISCLASSIFIED_FORMAL if by_class else MISCLASSIFIED)]
    for mask, colour in layers:                       # misclassified buildings drawn last (on top)
        if mask.any():
            g[mask].plot(ax=ax, color=colour, edgecolor="none", linewidth=0)

    # Official 2013 boundary: polygon edges, clipped to the site extent
    xmin, ymin, xmax, ymax = g.total_bounds
    box = shapely.box(xmin, ymin, xmax, ymax)
    polys = polygons.to_crs(g.crs)
    near = np.asarray(polys.geometry.values[polys.intersects(box).to_numpy()], dtype=object)
    if len(near):
        line = shapely.union_all(shapely.boundary(near)).intersection(box)
        gpd.GeoSeries([line], crs=g.crs).plot(ax=ax, color=BOUNDARY, linewidth=0.6, zorder=5)

    pad = 0.02 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#222222")
    ax.text(0.985, 0.015, cfg.site_name(site), transform=ax.transAxes, ha="right", va="bottom", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


# -----------------------------------------------------------------------------
# Panel b: per-site test accuracy
# -----------------------------------------------------------------------------
def accuracy_panel(ax, cfg, f2: dict) -> None:
    m = pd.read_csv(cfg.require_product("site_metrics")).set_index("site")
    acc = m["accuracy"].sort_values(ascending=False)          # high to low, top to bottom
    names = [f2.get("short_names", {}).get(s, cfg.site_name(s)) for s in acc.index]
    mean = float(m["accuracy"].mean())
    x_min = min(float(f2["panel_b_x_min"]), np.floor(acc.min() * 100) / 100 - 0.01)
    y = np.arange(len(acc))[::-1]
    ax.hlines(y, x_min, acc.to_numpy(), color=STEM, lw=1.7, zorder=1)
    ax.scatter(acc.to_numpy(), y, s=70, c=DOT, edgecolors="white", linewidths=0.7, zorder=3)
    for yi, a in zip(y, acc.to_numpy()):
        ax.text(a + 0.004, yi, f"{a:.3f}", va="center", ha="left", color="#555555", zorder=4)
    ax.axvline(mean, color=MEAN_LINE, ls="--", lw=0.9, zorder=2)
    ax.text(mean, len(acc) - 0.1, f"mean {mean:.3f}", color="#777777", ha="center", va="bottom")
    ax.set_xlim(x_min, 1.012)
    ax.set_ylim(-0.6, len(acc) - 0.05)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ticks = np.arange(np.ceil(x_min * 20) / 20, 1.0001, 0.05)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_xlabel("Test accuracy", labelpad=1)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)                      # boxed axes


# -----------------------------------------------------------------------------
# Panel c: misclassification rate by signed distance
# -----------------------------------------------------------------------------
def rate_table(cfg) -> pd.DataFrame:
    be = cfg.section("boundary_errors")
    width, half = float(be["bin_width_m"]), float(be["half_range_m"])
    min_n = int(be["min_buildings_per_site_bin"])
    z = float(be["wilson_z"])
    edges = np.arange(-half, half + width / 2, width)
    D = pd.read_parquet(cfg.require_product("oof_predictions"), columns=["site", "signed_dist", "error"])

    def binned(df):
        sd = df["signed_dist"].to_numpy(float)
        er = df["error"].to_numpy().astype(float)
        inr = (sd >= edges[0]) & (sd < edges[-1])
        idx = np.digitize(sd[inr], edges) - 1
        n = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        k = np.bincount(idx, weights=er[inr], minlength=len(edges) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(n >= min_n, k / n, np.nan)
        lo, hi = wilson_interval(k, n, z)
        return n, rate, np.where(n >= min_n, lo, np.nan), np.where(n >= min_n, hi, np.nan)

    n, rate, lo, hi = binned(D)
    per_site = np.array([binned(D[D.site == s])[1] for s in cfg.sites if (D.site == s).any()])
    p_lo, p_hi = be["site_band_percentiles"]
    with np.errstate(invalid="ignore"):
        s_lo, s_hi = np.nanpercentile(per_site, [p_lo, p_hi], axis=0)
    return pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "bin_centre": 0.5 * (edges[:-1] + edges[1:]),
                         "n_buildings": n.astype(int), "rate_pooled": rate, "ci_lo": lo, "ci_hi": hi,
                         f"site_p{p_lo:g}": s_lo, f"site_p{p_hi:g}": s_hi,
                         "n_sites_with_bin": np.isfinite(per_site).sum(axis=0)})


def rate_panel(ax, t: pd.DataFrame, cfg, f2: dict) -> None:
    be = cfg.section("boundary_errors")
    p_lo, p_hi = be["site_band_percentiles"]
    half = float(be["half_range_m"])
    x = t["bin_centre"].to_numpy()
    ax.fill_between(x, 100 * t[f"site_p{p_lo:g}"], 100 * t[f"site_p{p_hi:g}"], color=SITE_BAND, lw=0)
    ax.fill_between(x, 100 * t["ci_lo"], 100 * t["ci_hi"], color=RATE, alpha=0.25, lw=0)
    ax.plot(x, 100 * t["rate_pooled"], color=RATE, lw=2.0, solid_joinstyle="round")
    ax.axvline(0, color="#444444", lw=1.0, dashes=(3.7, 1.6))
    y_max = max(float(f2["panel_c_y_max_pct"]), float(np.nanmax(100 * t[f"site_p{p_hi:g}"])) + 2)
    ax.set_xlim(-half, half)
    ax.set_ylim(0, y_max)
    ax.set_xticks(np.arange(-half, half + 1, 30))
    ax.set_yticks(np.arange(0, y_max, 20))
    ax.set_xlabel("Signed distance to favela boundary (m)", labelpad=1)
    ax.set_ylabel("Buildings misclassified (%)")
    ax.text(-half + 4, y_max * 0.99, "Inside favela", color=MUTED_TEXT, ha="left", va="top")
    ax.text(half - 4, y_max * 0.99, "Outside favela", color=MUTED_TEXT, ha="right", va="top")
    handles = [Patch(fc=SITE_BAND, lw=0), (Patch(fc=RATE, alpha=0.25, lw=0), Line2D([], [], color=RATE, lw=2.0))]
    ax.legend(handles, [f"per-site {p_lo:g}–{p_hi:g}%", "all sites"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.15, handlelength=1.6, handletextpad=0.45, columnspacing=1.4)


# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    f2 = cfg.get("main_02_classification.figure2")
    fs = float(f2["font_size_pt"])
    use_style(**{"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs,
                 "legend.fontsize": fs, "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=figure_size(float(f2["width_mm"]), float(f2["height_mm"])))
    label_kw = dict(fontsize=float(f2["panel_label_size_pt"]), ha="left", va="top")

    # (a) two outcome maps side by side, legend underneath
    polygons = read_favela_polygons(cfg, 2013)
    # the two sites of the caption, or the first processed sites if they are not in the run
    sites_a = [s for s in f2["panel_a_sites"] if s in cfg.sites] or list(cfg.sites[:2])
    map_axes = [fig.add_axes([0.035, 0.575, 0.445, 0.40]), fig.add_axes([0.52, 0.575, 0.445, 0.40])]
    for i, ax in enumerate(map_axes):
        if i < len(sites_a):
            outcome_map(ax, cfg, sites_a[i], polygons, f2)
        else:
            ax.axis("off")
    by_class = f2.get("misclassified_colours", "by_class") == "by_class"
    handles = [Patch(fc=FAVELA_MAP), Patch(fc=FORMAL_MAP),
               (Patch(fc=MISCLASSIFIED), Patch(fc=MISCLASSIFIED_FORMAL)) if by_class else Patch(fc=MISCLASSIFIED)]
    labels = ["Favela", "Formal settlement", "Misclassified"]
    if f2["panel_a_buildings"] == "test_only":
        handles.append(Patch(fc=NEUTRAL_BUILDING))
        labels.append("Training set (not evaluated)")
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="center",
               bbox_to_anchor=(0.5, 0.545), ncol=len(labels), frameon=False, handlelength=1.4, columnspacing=1.6)
    fig.text(0.005, 0.995, "(a)", **label_kw)

    # (b) per-site test accuracy
    ax_b = fig.add_axes([0.215, 0.085, 0.29, 0.38])
    accuracy_panel(ax_b, cfg, f2)
    fig.text(0.005, 0.505, "(b)", **label_kw)

    # (c) misclassification rate by signed distance
    t = rate_table(cfg)
    write_table(t, out_dir / "figure2c_values.csv")
    ax_c = fig.add_axes([0.635, 0.085, 0.345, 0.38])
    rate_panel(ax_c, t, cfg, f2)
    fig.text(0.545, 0.505, "(c)", **label_kw)

    path = save_figure(fig, out_dir / "figure2.pdf")
    plt.close(fig)
    pk = int(np.nanargmax(t["rate_pooled"].to_numpy()))
    print(f"pooled rate peaks at {100 * t['rate_pooled'].iloc[pk]:.1f}% in [{t['bin_lo'].iloc[pk]:g}, "
          f"{t['bin_hi'].iloc[pk]:g}) m")
    print(f"written {path} and {out_dir / 'figure2c_values.csv'}")


if __name__ == "__main__":
    main()
