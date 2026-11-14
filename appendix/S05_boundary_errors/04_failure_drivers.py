"""Step 4: Supplementary Fig. S3, Tree-SHAP drivers of the false positives in each site.

Manuscript items
* Supplementary Fig. S3: for each site, the mean Tree-SHAP contribution among
  false positives of the features that rank among the top drivers in any site;
  a black box marks each site's dominant driver.
* SI S5 text: the dominant driver is site-dependent (the adjacency ratio in
  Rocinha and Catumbi, perimeter wall in Grotao de Costa Barros, mean
  interbuilding distance in most other sites), with the median interbuilding distance of
  false positives and correctly classified formal buildings (for example
  3.1 m versus 4.7 m in Jacarezinho). Written to failure_drivers.csv and
  failure_drivers_numbers.txt.

Definitions
* False positive: a building labelled formal settlement that the main per-site
  classifier (Methods 4.4) assigns to the favela class; true negative: a formal
  building classified formal. ``failure_drivers_figure.population``:
  ``all_buildings``: the classifier's prediction for every building of the site;
  ``test_split``: held-out test buildings only.
* Mean SHAP among false positives: mean of each indicator's Tree-SHAP value
  (positive toward favela) over the site's false positives.
* Features shown: the union of each site's three largest mean values; rows
  ordered by their mean over the sites. Dominant driver: the shown feature with
  the largest mean value in the site (black box).
* Interbuilding distance: median of the indicator 'mean interbuilding
  distance' over the false positives and over the true negatives of each site.

Inputs:  ``products.shap_gpkg`` (main/02_classification); ``products.site_metrics``
         for the site order when ``failure_drivers_figure.site_order: by_test_accuracy``.
Outputs (outputs/appendix/S05_boundary_errors/): failure_drivers.csv,
         failure_drivers_numbers.txt, Figure_R1_failure_shap.pdf (+ PNG preview).

Run from the repository root:
    python appendix/S05_boundary_errors/04_failure_drivers.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from favela_xai.config import ConfigError, load_config  # noqa: E402
from favela_xai.features import MODEL_FEATURE_ORDER, label_of  # noqa: E402
from favela_xai.io import write_table, write_text  # noqa: E402
from favela_xai.plotting_style import SHAP_CMAP  # noqa: E402

import s05_common as C  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                     "legend.frameon": False})
IBD = "mean_interbuilding_distance"


def site_order(cfg, rule) -> list[str]:
    """Column order: by the SI Table S3 test accuracy (high to low), or a fixed list."""
    if rule == "by_test_accuracy":
        path = cfg.product("site_metrics")
        if path.exists():
            df = pd.read_csv(path)
            col = next((c for c in ("site", "site_key") if c in df.columns), None)
            if col is not None and "accuracy" in df.columns:
                acc = df.set_index(col)["accuracy"].astype(float)
                return sorted(cfg.sites, key=lambda s: -acc.get(s, -np.inf))
        print(f"NOTE {path} not found or without site/accuracy columns; using the fixed order failure_drivers_figure.fixed_order")
        rule = "fixed"
    if rule == "fixed":
        fixed = C.settings(cfg)["failure_drivers_figure"]["fixed_order"]
        return [s for s in fixed if s in cfg.sites] + [s for s in cfg.sites if s not in fixed]
    raise ConfigError("appendix_S05_boundary_errors.failure_drivers_figure.site_order must be by_test_accuracy or fixed")


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    f = C.settings(cfg)["failure_drivers_figure"]
    if f["population"] not in ("all_buildings", "test_split"):
        raise ConfigError("appendix_S05_boundary_errors.failure_drivers_figure.population must be all_buildings or test_split")
    top_n = int(f["top_per_site"])

    rows, fp_mean = [], {}
    for site in cfg.sites:
        t = C.main_outputs(cfg, site, extra=(IBD,))
        if f["population"] == "test_split":
            t = t[t["is_test"]]
        fp = (t.is_favela == 0) & (t.pred_favela == 1)
        tn = (t.is_favela == 0) & (t.pred_favela == 0)
        m = t.loc[fp, ["shap_" + k for k in MODEL_FEATURE_ORDER]].mean()
        m.index = MODEL_FEATURE_ORDER
        fp_mean[site] = m
        ibd = pd.to_numeric(t[IBD], errors="coerce")
        rows.append(dict(site=site, site_name=cfg.site_name(site), population=f["population"],
                         n_formal=int((t.is_favela == 0).sum()), n_false_positive=int(fp.sum()),
                         n_true_negative=int(tn.sum()),
                         dominant_driver_all25=m.idxmax() if fp.any() else "",
                         median_interbuilding_distance_fp=float(ibd[fp].median()) if fp.any() else np.nan,
                         median_interbuilding_distance_tn=float(ibd[tn].median()) if tn.any() else np.nan,
                         **{f"fp_mean_shap_{k}": float(m[k]) for k in MODEL_FEATURE_ORDER}))

    order = site_order(cfg, f["site_order"])
    top = set()
    for s in order:
        if fp_mean[s].notna().any():
            top.update(fp_mean[s].nlargest(top_n).index)
    feats = sorted(top, key=lambda k: -np.nanmean([fp_mean[s].get(k, np.nan) for s in order]))
    M = np.array([[fp_mean[s].get(k, np.nan) for s in order] for k in feats])
    dominant = {s: feats[int(np.nanargmax(M[:, j]))] if np.isfinite(M[:, j]).any() else "" for j, s in enumerate(order)}
    R = pd.DataFrame(rows)
    R["dominant_driver_shown"] = R.site.map(dominant)
    R["features_shown"] = ";".join(feats)
    write_table(R, out / C.FAILURE_CSV)
    lines = [f"false positives from: {f['population']}"]
    for s in order:
        r = R.set_index("site").loc[s]
        lines.append(f"{cfg.site_name(s):<24} FP {int(r.n_false_positive):>5,}  dominant driver {r.dominant_driver_shown:<28}"
                     f" median interbuilding distance FP {r.median_interbuilding_distance_fp:.1f} m, "
                     f"TN {r.median_interbuilding_distance_tn:.1f} m")
    write_text("\n".join(lines) + "\n", out / "failure_drivers_numbers.txt")
    print("\n".join(lines))

    # ---- heatmap (vector cells)
    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=tuple(f["size_in"]))
    im = ax.pcolormesh(np.arange(len(order) + 1) - .5, np.arange(len(feats) + 1) - .5, np.ma.masked_invalid(M),
                       cmap=SHAP_CMAP, vmin=-vmax, vmax=vmax, shading="flat", edgecolors="face", linewidth=0.3)
    ax.set_xlim(-.5, len(order) - .5)
    ax.set_ylim(len(feats) - .5, -.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([C.site_label(cfg, s, short=True) for s in order], rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([label_of(k) for k in feats], fontsize=6.5)
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j, s in enumerate(order):
        if dominant[s]:
            ax.add_patch(plt.Rectangle((j - .5, feats.index(dominant[s]) - .5), 1, 1, fill=False, ec="#272727", lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("mean SHAP among false positives\n(positive → favela)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    fig.tight_layout()
    fig.savefig(out / C.FIGURE_FAILURE, bbox_inches="tight", dpi=600)
    fig.savefig((out / C.FIGURE_FAILURE).with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"written {out / C.FIGURE_FAILURE}")


if __name__ == "__main__":
    main()
