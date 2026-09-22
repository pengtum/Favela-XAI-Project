"""Step 5: Supplementary Fig. S1, study favelas compared with all favelas in the official 2013 boundaries.

Manuscript items
* Supplementary Fig. S1:
  a, the 1,076 favela polygons of the 2013 layer, study favelas coloured by
  site, the others in grey, Praca XV (star), the municipal boundary and the
  planning areas AP1-AP5;
  b-e, cumulative distributions for study (black) and other (grey) favelas of
  polygon area, compactness 4 pi A / P^2, distance to Praca XV and footprints
  per hectare, with the two-sample Kolmogorov-Smirnov D and p; ticks mark the
  study favelas, coloured by site.

Inputs (outputs of steps 01, 03 and 04 of this folder)
* context_layers/rio_planning_areas.gpkg, context_layers/rio_municipality.gpkg (01);
  polygons_base.parquet (03); city_polygons.csv (04).

Outputs (outputs/appendix/S02_site_selection/)
* Figure_R1_site_selection.pdf (and a PNG preview).

Settings
* ``site_context.map_min_part_area_ha``: islets below this area are dropped
  from the map's context layers; ``site_context.city_centre_lonlat``.

Run from the repository root:
    python appendix/S02_site_selection/05_figure_S1.py
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

from favela_xai.config import load_config  # noqa: E402

import s02_lib as L  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False})

# Site colours of SI Fig. S1 (config site order)
SITE_COLOURS = {"barao": "#1b9e77", "catumbi": "#d95f02", "faz_quem_quer": "#7570b3", "fazena_coqueiro": "#e7298a",
                "grotao_de_costa_barros": "#66a61e", "jacarezinho": "#e6ab02", "morro_da_guaiba": "#a6761d",
                "nova_cidade": "#1f78b4", "palmeras": "#b15928", "rocinha": "#6a3d9a"}
OTHER = "#9a9a9a"
# Context layers: the ground, not the subject
C_LAND, C_COAST, C_AP, C_APTXT = "#f3f2ef", "#adadad", "#cfcfcf", "#8a8a8a"
# Planning-area label positions (EPSG:31983); AP1 gets a short leader into Guanabara Bay
AP_LABELS = {"AP2": (678138, 7459074), "AP3": (678381, 7474101), "AP4": (658001, 7460797), "AP5": (639940, 7461364)}
AP1_LABEL, AP1_ANCHOR = (691900, 7463150), (688400, 7464700)
SCALE_BAR = (624000.0, 7449400.0)     # 10 km bar over Sepetiba Bay
NORTH_ARROW = (628200.0, 7477600.0)


def drop_islets(geom, min_ha: float):
    """Keep only polygon parts of at least min_ha (map generalisation of the context layers)."""
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    keep = [p for p in parts if p.area / 1e4 >= min_ha]
    return shapely.union_all(keep), len(parts) - len(keep)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    sites = cfg.sites
    names = L.site_display_names(cfg)
    crs = cfg.get("crs")
    min_ha = float(cfg.get("site_context.map_min_part_area_ha"))

    pol = L.read_polygons(out, cfg)
    city = pd.read_csv(L.require(out / L.CITY_POLYGONS_CSV, "04_compare_study_vs_city.py")).set_index(
        "poly_index").sort_index()
    pol = pol[["geometry"]].join(city[["study", "site"]])
    ap = gpd.read_file(L.require(out / L.PLANNING_AREAS_GPKG, "01_context_layers.py"),
                       layer="planning_areas_simplified").set_index("ap")
    mu = gpd.read_file(out / L.MUNICIPALITY_GPKG, layer="municipality_simplified")
    ap_draw = {a: drop_islets(ap.geometry[a], min_ha)[0] for a in ap.index}
    mu_draw, n_drop = drop_islets(mu.geometry.iloc[0], min_ha)
    print(f"context layers: municipal islets < {min_ha:g} ha dropped from the map: {n_drop}")

    fig = plt.figure(figsize=(7.1, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=0.38, wspace=0.42,
                          left=0.07, right=0.985, top=0.975, bottom=0.095)
    axm = fig.add_subplot(gs[0, :3])
    axl = fig.add_subplot(gs[0, 3])
    axl.axis("off")
    # the map box reaches a little into the row gap so the whole municipality fits; panels b-e keep their size
    p = axm.get_position()
    grow = 0.22 / fig.get_figheight()
    axm.set_position([p.x0, p.y0 - grow, p.width, p.height + grow])
    p = axm.get_position()
    box_ar = (p.width * fig.get_figwidth()) / (p.height * fig.get_figheight())

    # ---- a: context (municipal land, planning areas, coastline), then the favela polygons
    gpd.GeoSeries([mu_draw], crs=crs).plot(ax=axm, color=C_LAND, linewidth=0, zorder=0)
    gpd.GeoSeries(list(ap_draw.values()), crs=crs).boundary.plot(ax=axm, color=C_AP, linewidth=0.4, zorder=1)
    gpd.GeoSeries([mu_draw], crs=crs).boundary.plot(ax=axm, color=C_COAST, linewidth=0.6, zorder=2)
    pol[~pol.study.astype(bool)].plot(ax=axm, color=OTHER, linewidth=0, zorder=3)
    for s in sites:
        sub = pol[pol.site == s]
        if len(sub):
            sub.plot(ax=axm, color=SITE_COLOURS[s], linewidth=0.25, edgecolor=SITE_COLOURS[s], zorder=4)
    cx0, cy0 = L.city_centre_xy(cfg)
    axm.plot(cx0, cy0, marker="*", ms=8, color="k", mec="w", mew=0.4, zorder=6)
    axm.annotate("Praça XV", (cx0, cy0), xytext=(4, 5), textcoords="offset points", fontsize=6.5, zorder=6)
    halo = [pe.withStroke(linewidth=1.3, foreground="white")]
    for a, (lx, ly) in AP_LABELS.items():
        axm.text(lx, ly, a, fontsize=6, color=C_APTXT, ha="center", va="center", zorder=5, path_effects=halo)
    axm.plot([AP1_LABEL[0] - 1500, AP1_ANCHOR[0]], [AP1_LABEL[1] + 150, AP1_ANCHOR[1]],
             color=C_APTXT, lw=0.35, solid_capstyle="round", zorder=5)
    axm.text(*AP1_LABEL, "AP1", fontsize=6, color=C_APTXT, ha="center", va="center", zorder=5, path_effects=halo)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    # extent from the municipal boundary; x padded with open water so the equal-aspect box is filled
    bx0, by0, bx1, by1 = mu_draw.bounds
    ymin, ymax = by0 - 700, by1 + 700
    pad = (ymax - ymin) * box_ar - (bx1 - bx0)
    axm.set_xlim(bx0 - 0.40 * pad, bx1 + 0.60 * pad)
    axm.set_ylim(ymin, ymax)
    sbx, sby = SCALE_BAR
    axm.plot([sbx, sbx + 10000], [sby, sby], color="k", lw=1.2, solid_capstyle="butt", zorder=6)
    axm.text(sbx + 5000, sby + 500, "10 km", ha="center", va="bottom", fontsize=6.5, zorder=6)
    nax, nay = NORTH_ARROW
    axm.annotate("", xy=(nax, nay + 2600), xytext=(nax, nay), zorder=6,
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=0.7, mutation_scale=5.5, shrinkA=0, shrinkB=0))
    axm.text(nax, nay + 3000, "N", ha="center", va="bottom", fontsize=6.5, zorder=6)

    n_study, n_city = int(pol.study.astype(bool).sum()), len(pol)
    handles = [Patch(color=SITE_COLOURS[s], label=f"{names[s]} ({int((pol.site == s).sum())})") for s in sites]
    leg = axl.legend(handles=handles, loc="upper left", ncol=1, frameon=False, fontsize=6.3, handlelength=1.0,
                     labelspacing=0.35, title=f"Study favelas by site (n = {n_study})", title_fontsize=6.5,
                     bbox_to_anchor=(-0.05, 1.0), alignment="left")
    axl.add_artist(leg)
    axl.legend(handles=[Patch(color=OTHER, label=f"Other 2013 favelas\n(n = {n_city - n_study})"),
                        Patch(facecolor=C_LAND, edgecolor=C_COAST, linewidth=0.5, label="Rio de Janeiro municipality"),
                        Line2D([0], [0], color=C_AP, lw=0.7, label="Planning areas (AP1–AP5)")],
               loc="lower left", frameon=False, fontsize=6.3, handlelength=1.0, bbox_to_anchor=(-0.05, 0.0))
    axm.text(0.0, 1.0, "a", transform=axm.transAxes, fontsize=9, fontweight="bold", va="top")

    # ---- b-e: cumulative distributions, study versus other favelas
    panels = [("area_ha", "Polygon area (ha)", True), ("compactness", "Compactness 4πA/P²", False),
              ("dist_centre_km", "Distance to Praça XV (km)", False),
              ("bldg_density_per_ha", "Building footprints per ha", True)]
    st_all, ot_all = city[city.study.astype(bool)], city[~city.study.astype(bool)]
    for i, (v, lab, logx) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        a, b = st_all[v].dropna(), ot_all[v].dropna()
        if logx:
            a, b = a[a > 0], b[b > 0]
        for arr, col, name in [(b, "#7f7f7f", "Other"), (a, "k", "Study")]:
            xs = np.sort(arr.to_numpy())
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=col, lw=1.1,
                    label=f"{name} (n = {len(xs)})")
        for s in sites:          # study favelas as ticks below the curves
            vv = st_all.loc[st_all.site == s, v].dropna()
            if logx:
                vv = vv[vv > 0]
            ax.scatter(vv, np.full(len(vv), -0.06), marker="|", s=18, lw=0.7, color=SITE_COLOURS[s], clip_on=False,
                       zorder=4)
        ks = stats.ks_2samp(st_all[v].dropna(), ot_all[v].dropna())
        ptxt = "p < 0.001" if ks.pvalue < 0.001 else f"p = {ks.pvalue:.2f}"
        ax.text(0.03, 0.97, f"KS D = {ks.statistic:.2f}\n{ptxt}", transform=ax.transAxes, va="top", ha="left",
                fontsize=6.3)
        print(f"  {v}: KS D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(lab)
        if i == 0:
            ax.set_ylabel("Cumulative share of polygons")
        if i == 2:
            ax.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.2, borderaxespad=0.1,
                      bbox_to_anchor=(1.0, 0.05))
        ax.text(-0.3 if i == 0 else -0.18, 1.1, "bcde"[i], transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top")

    fig.savefig(out / L.FIGURE_S1)
    fig.savefig((out / L.FIGURE_S1).with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"written {out / L.FIGURE_S1}")


if __name__ == "__main__":
    main()
