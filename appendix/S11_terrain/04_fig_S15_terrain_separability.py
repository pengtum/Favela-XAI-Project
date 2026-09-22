"""SI Fig. S15: terrain distribution overlap (OVL) of favela and formal-settlement buildings versus accuracy.

Manuscript items
* SI Fig. S15 (an editable .svg is written as well).
* SI S11 'Terrain distribution overlap versus classification accuracy': the
  overlap coefficient OVL = integral of min(f_fav, f_nonfav) of the
  kernel-density estimates of mean slope and of mean elevation at each site,
  its range across sites, and its Spearman correlation with per-site accuracy
  (n = 10).

Computation
* Buildings (``appendix_S11_terrain.ovl_population``): ``all_buildings`` (all
  buildings of each site) or ``test_set`` (the test buildings of the ablation
  split).
* Gaussian kernel-density estimates (scipy gaussian_kde, bandwidth factor
  ``terrain.ovl_kde_bandwidth`` = 0.3) on a grid of ``terrain.ovl_grid_points``
  = 500 points spanning the pooled range of both classes padded by 5% on each
  side; OVL by the trapezoidal rule.
* Accuracy (``appendix_S11_terrain.figure_accuracy``): SI Table S3
  (``table_S3``) or the Morph column of the ablation (``ablation_morph``).
* Sites are ordered top to bottom by accuracy.
* Colours (``appendix_S11_terrain.fig_S15_colours``): favela, formal
  settlement and overlap zone.

Inputs
* outputs/appendix/S11_terrain/<site>_buildings_with_terrain.gpkg (01), or
  exp1_pooled_test_buildings.csv (02) with ovl_population = test_set.
* products.site_metrics (main/02_classification), or the ablation Morph
  accuracy (02).

Outputs (outputs/appendix/S11_terrain/)
* ``terrain_overlap_by_site.csv``: accuracy, slope OVL and elevation OVL per site.
* ``terrain_overlap_correlations.csv``: Spearman rho and p of OVL against accuracy.
* ``terrain_separability_vs_accuracy.pdf`` / ``.svg``: SI Fig. S15.

Run from the repository root:
    python appendix/S11_terrain/04_fig_S15_terrain_separability.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde, spearmanr  # noqa: E402

from favela_xai.config import ConfigError, load_config  # noqa: E402
from favela_xai.io import write_table  # noqa: E402

import terrain_common as tc  # noqa: E402

# Figure style. The class colours come from appendix_S11_terrain.fig_S15_colours.
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
      "axes.linewidth": 0.6, "legend.frameon": False,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6}
REF = "#767676"      # annotation text
YLIM_C, XLIM_C = (82.0, 100.0), (0.05, 1.0)   # default ranges of panel c


def kde_overlap(a, b, bandwidth: float, n_grid: int, pad: float, min_n: int):
    """Overlap coefficient of two kernel-density estimates.

    Returns (OVL, grid, density a, density b); OVL is NaN when either sample
    has fewer than ``min_n`` finite values or both samples are constant.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < min_n or len(b) < min_n:
        return np.nan, None, None, None
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    span = hi - lo
    if span == 0:
        return np.nan, None, None, None
    x = np.linspace(lo - pad * span, hi + pad * span, n_grid)
    try:
        ka = gaussian_kde(a, bw_method=bandwidth)(x)
        kb = gaussian_kde(b, bw_method=bandwidth)(x)
    except (np.linalg.LinAlgError, ValueError):   # degenerate sample (e.g. all values identical in one class)
        return np.nan, None, None, None
    return float(np.trapz(np.minimum(ka, kb), x)), x, ka, kb


def load_buildings(cfg, out, variables) -> pd.DataFrame:
    """site, label and the OVL variables of the buildings used for the OVL."""
    population = cfg.get("appendix_S11_terrain.ovl_population")
    if population == "all_buildings":
        frames = []
        for site in cfg.sites:
            path = tc.require(out / tc.BUILDINGS_GPKG.format(site=site), "01_building_terrain.py")
            df = pyogrio.read_dataframe(path, columns=["label"] + list(variables), read_geometry=False)
            df.insert(0, "site", site)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    if population == "test_set":
        df = pd.read_csv(tc.require(out / tc.POOLED_TEST_CSV, "02_terrain_ablation.py"))
        return df[["site", "label"] + list(variables)]
    raise ConfigError(f"appendix_S11_terrain.ovl_population must be all_buildings or test_set, not '{population}'")


def draw_ridge(ax, groups, ordered, var, xlabel, show_labels, kde_kw, ridge_min, colours):
    """One ridge per site: favela and formal densities, their overlap zone and the OVL value.

    ``colours``: dict with the keys favela, formal and overlap.
    """
    n = len(ordered)
    sp = 1.0
    for i, (site, name, acc, ovl) in enumerate(ordered):
        fav, nfav = groups[site][var]
        if len(fav) < ridge_min or len(nfav) < ridge_min:
            continue
        _, x, kf, kn = kde_overlap(fav, nfav, **kde_kw)
        if x is None:
            continue
        yb = (n - 1 - i) * sp
        mx = max(kf.max(), kn.max())
        sc = sp * 0.85 / mx if mx > 0 else 1.0
        kf_s, kn_s = kf * sc, kn * sc
        ax.fill_between(x, yb, yb + kf_s, alpha=0.40, color=colours["favela"], lw=0)
        ax.fill_between(x, yb, yb + kn_s, alpha=0.40, color=colours["formal"], lw=0)
        ax.fill_between(x, yb, yb + np.minimum(kf_s, kn_s), alpha=0.55, color=colours["overlap"], lw=0)
        ax.plot(x, yb + kf_s, color=colours["favela"], lw=0.8)
        ax.plot(x, yb + kn_s, color=colours["formal"], lw=0.8)
        ax.axhline(yb, color="#cccccc", lw=0.3)
        ax.text(0.985, yb + sp * 0.42, f"OVL {ovl:.2f}" if not np.isnan(ovl) else "OVL n/a",
                fontsize=5.3, color=colours["overlap"], ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks([(n - 1 - i) * sp + sp * 0.3 for i in range(n)])
    ax.set_yticklabels([f"{name} ({acc:.1f}%)" for _, name, acc, _ in ordered] if show_labels else [""] * n,
                       fontsize=6)
    ax.set_ylim(-0.3, n * sp)
    ax.set_xlabel(xlabel, fontsize=7)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)


def panel(ax, title):
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=4)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    plt.rcParams.update(RC)
    colours = cfg.get("appendix_S11_terrain.fig_S15_colours")
    fav, nfv, ovl_c = colours["favela"], colours["formal"], colours["overlap"]
    fav_label = cfg.get("labels.favela")
    slope_var, elev_var = cfg.get("appendix_S11_terrain.ovl_variables")
    kde_kw = dict(bandwidth=float(cfg.get("terrain.ovl_kde_bandwidth")),
                  n_grid=int(cfg.get("terrain.ovl_grid_points")),
                  pad=float(cfg.get("appendix_S11_terrain.ovl_grid_pad_fraction")),
                  min_n=int(cfg.get("appendix_S11_terrain.ovl_min_buildings")))
    ridge_min = int(cfg.get("appendix_S11_terrain.ridge_min_buildings"))

    buildings = load_buildings(cfg, out, [slope_var, elev_var])
    accuracy = tc.figure_accuracy(cfg, out) * 100.0              # percent, as drawn

    # Per-site favela / formal samples and OVL values
    groups, rows = {}, []
    for site in cfg.sites:
        b = buildings[buildings["site"] == site]
        is_fav = b["label"].astype(str) == fav_label
        groups[site] = {v: (b.loc[is_fav, v].dropna().to_numpy(), b.loc[~is_fav, v].dropna().to_numpy())
                        for v in (slope_var, elev_var)}
        rows.append(dict(site=site, name=tc.display_name(cfg, site), accuracy=accuracy[site],
                         n_favela=int(is_fav.sum()), n_formal=int((~is_fav).sum()),
                         ovl_slope=kde_overlap(*groups[site][slope_var], **kde_kw)[0],
                         ovl_elevation=kde_overlap(*groups[site][elev_var], **kde_kw)[0]))
    stats = pd.DataFrame(rows).sort_values("accuracy", ascending=False, kind="mergesort").reset_index(drop=True)
    stats["ovl_population"] = cfg.get("appendix_S11_terrain.ovl_population")
    write_table(stats, out / tc.OVERLAP_CSV)

    rho_s, p_s = spearmanr(stats["ovl_slope"], stats["accuracy"], nan_policy="omit")
    rho_e, p_e = spearmanr(stats["ovl_elevation"], stats["accuracy"], nan_policy="omit")
    write_table(pd.DataFrame([
        dict(comparison="slope OVL vs accuracy", spearman_rho=rho_s, p_value=p_s, n=int(stats["ovl_slope"].notna().sum())),
        dict(comparison="elevation OVL vs accuracy", spearman_rho=rho_e, p_value=p_e,
             n=int(stats["ovl_elevation"].notna().sum())),
    ]), out / tc.OVERLAP_CORR_CSV)

    o_s = [(r.site, r.name, r.accuracy, r.ovl_slope) for r in stats.itertuples()]
    o_e = [(r.site, r.name, r.accuracy, r.ovl_elevation) for r in stats.itertuples()]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    draw_ridge(axes[0], groups, o_s, slope_var, "mean slope (°)", True, kde_kw, ridge_min, colours)
    panel(axes[0], "a   Slope distribution")
    draw_ridge(axes[1], groups, o_e, elev_var, "mean elevation (m)", False, kde_kw, ridge_min, colours)
    panel(axes[1], "b   Elevation distribution")

    ax = axes[2]
    ax.scatter(stats["ovl_slope"], stats["accuracy"], s=26, c=fav, edgecolors="black", lw=0.4, marker="o",
               zorder=5, label="slope OVL")
    ax.scatter(stats["ovl_elevation"], stats["accuracy"], s=26, c=nfv, edgecolors="black", lw=0.4, marker="s",
               zorder=5, label="elevation OVL")
    xfit = np.linspace(0, 1, 50)
    for col, colour in [("ovl_slope", fav), ("ovl_elevation", nfv)]:
        ok = stats[col].notna()
        if ok.sum() >= 2:       # least-squares trend line
            ax.plot(xfit, np.poly1d(np.polyfit(stats.loc[ok, col], stats.loc[ok, "accuracy"], 1))(xfit),
                    "--", color=colour, lw=1.0, alpha=0.6)
    ax.text(0.04, 0.04, f"slope: $\\rho$={rho_s:.2f}, p={p_s:.2f}\nelev:  $\\rho$={rho_e:.2f}, p={p_e:.2f}",
            transform=ax.transAxes, fontsize=6, va="bottom", color=REF)
    ax.set_xlabel("terrain distribution overlap (OVL)", fontsize=7)
    ax.set_ylabel("accuracy (%)", fontsize=7)
    ovl_all = pd.concat([stats["ovl_slope"], stats["ovl_elevation"]]).dropna()
    ax.set_xlim(min(XLIM_C[0], ovl_all.min() - 0.05) if len(ovl_all) else XLIM_C[0], XLIM_C[1])
    ax.set_ylim(min(YLIM_C[0], stats["accuracy"].min() - 1.0), max(YLIM_C[1], stats["accuracy"].max() + 1.0))
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)
    panel(ax, "c   Overlap vs accuracy")

    fig.legend(handles=[Patch(fc=fav, alpha=0.6, label="favela"), Patch(fc=nfv, alpha=0.6, label="non-favela"),
                        Patch(fc=ovl_c, alpha=0.7, label="overlap zone")],
               loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.28)
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{tc.FIG_S15}.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(stats[["name", "accuracy", "ovl_slope", "ovl_elevation"]].round(2).to_string(index=False))
    print(f"  slope OVL {stats['ovl_slope'].min():.2f}-{stats['ovl_slope'].max():.2f}; "
          f"elevation OVL {stats['ovl_elevation'].min():.2f}-{stats['ovl_elevation'].max():.2f}")
    print(f"  slope rho = {rho_s:.2f}, p = {p_s:.2f} | elevation rho = {rho_e:.2f}, p = {p_e:.2f}")
    # SI S11: sites with slope OVL above 0.90 and accuracy above 94%
    high = stats[(stats["ovl_slope"] > 0.90) & (stats["accuracy"] > 94.0)]
    print("  sites with slope OVL > 0.90 and accuracy > 94%: " + (", ".join(high["name"]) if len(high) else "none"))
    print(f"  saved {out / (tc.FIG_S15 + '.pdf')} (+ .svg)")


if __name__ == "__main__":
    main()
