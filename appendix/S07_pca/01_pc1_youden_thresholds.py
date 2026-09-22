"""SI Section S7: per-site ROC-Youden thresholds of the pooled attribution PC1.

Thresholds PC1 of the pooled, standardised attribution PCA (Fig. 5) within
each site at its ROC-Youden optimum, with the official 2013 favela labels as
reference, and flags the buildings above the threshold as 'favela-like'
(SI Section S7, second paragraph; Results 2.4). The maps of SI Fig. S10 are
drawn from the table written here by 02_figure_pca_mapping.py.

Method (config.yaml ``pca`` and ``appendix_S07_pca``)
* PC1: first component of the pooled, standardised PCA of the 25 Tree-SHAP
  columns of all ten sites (Methods 4.6), computed by
  main/05_attribution_plane/01_attribution_pca.py and oriented so that favela
  buildings score higher.
* Threshold: for each site, the point of the ROC curve of PC1 (favela =
  positive class) that maximises Youden's J = TPR - FPR
  (favela_xai.attribution.youden_threshold).
* ``favela_like_rule``: '>' (PC1 above the threshold) | '>=' (at or above).
* Agreement with the official polygon is summarised per site by the AUC of PC1
  and by J; the weakest site is printed over all sites and over the sites of
  SI Fig. S10.

Inputs
* products.attribution_pca_scores (main/05_attribution_plane): site, row,
  uID, is_favela, PC1 per building.
Outputs (outputs/appendix/S07_pca/)
* pc1_youden_thresholds.csv: per site the threshold, J, TPR, FPR, AUC,
  numbers of buildings, class means of PC1, share of favela-like buildings
  inside and outside the 2013 polygons, and whether the site is mapped;
* pc1_favela_like.parquet: per building site, row, uID, is_favela, PC1 and
  favela_like (input of 02_figure_pca_mapping.py).

Run from the repository root (after main/05_attribution_plane):
    python appendix/S07_pca/01_pc1_youden_thresholds.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from favela_xai.attribution import youden_threshold
from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_table

THRESHOLDS = "pc1_youden_thresholds.csv"
BUILDINGS = "pc1_favela_like.parquet"


def favela_like(pc1: np.ndarray, threshold: float, rule: str) -> np.ndarray:
    """Buildings called favela-like: PC1 above (``>``) or at/above (``>=``) the threshold."""
    if rule == ">":
        return pc1 > threshold
    if rule == ">=":
        return pc1 >= threshold
    raise ConfigError("appendix_S07_pca.favela_like_rule must be '>' or '>='")


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    s = cfg.section("appendix_S07_pca")
    if cfg.get("pca.youden_threshold") != "per_site":
        raise ConfigError("SI S7: the threshold is fitted within each site (pca.youden_threshold: per_site)")
    rule = s["favela_like_rule"]
    shown = [x for x in s["map_sites"] if x in cfg.sites]

    scores = pd.read_parquet(cfg.require_product("attribution_pca_scores"))
    for col in ("site", "row", "is_favela", "PC1"):
        if col not in scores.columns:
            raise ConfigError(f"products.attribution_pca_scores has no column '{col}'; "
                              "re-run main/05_attribution_plane/01_attribution_pca.py")

    rows, parts = [], []
    for site in cfg.sites:
        t = scores[scores["site"] == site].copy()
        if t.empty:
            raise ConfigError(f"no PCA scores for {site}; re-run main/05_attribution_plane")
        fav = t["is_favela"].to_numpy(bool)
        pc1 = t["PC1"].to_numpy(float)
        if fav.all() or not fav.any():
            raise ConfigError(f"{site}: both classes are needed for a ROC-Youden threshold")
        y = youden_threshold(pc1, fav)
        like = favela_like(pc1, y["threshold"], rule)
        t["favela_like"] = like
        parts.append(t[[c for c in ("site", "row", "uID", "is_favela", "PC1", "favela_like") if c in t.columns]])
        rows.append(dict(site=site, site_name=cfg.site_name(site), shown_in_figure=site in shown,
                         threshold=y["threshold"], youden_j=y["youden_j"], tpr=y["tpr"], fpr=y["fpr"],
                         auc=y["auc"], n=y["n"], n_favela=int(fav.sum()), n_formal=int((~fav).sum()),
                         mean_pc1_favela=float(pc1[fav].mean()), mean_pc1_formal=float(pc1[~fav].mean()),
                         share_favela_like_inside=float(like[fav].mean()),
                         share_favela_like_outside=float(like[~fav].mean())))
    table = pd.DataFrame(rows)
    write_table(table, out / THRESHOLDS)
    write_table(pd.concat(parts, ignore_index=True), out / BUILDINGS)

    # ---- numbers behind the SI S7 text -----------------------------------------
    print(table[["site_name", "threshold", "youden_j", "auc", "share_favela_like_inside",
                 "share_favela_like_outside"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nthresholds {table['threshold'].min():.3f}-{table['threshold'].max():.3f}")
    for label, sub in (("all sites", table), ("sites shown in SI Fig. S10", table[table["shown_in_figure"]])):
        if len(sub):
            a = sub.loc[sub["auc"].idxmin()]
            j = sub.loc[sub["youden_j"].idxmin()]
            print(f"weakest agreement, {label}: lowest AUC {a['site_name']} ({a['auc']:.4f}); "
                  f"lowest J {j['site_name']} ({j['youden_j']:.3f})")
    print(f"[saved] {out / THRESHOLDS}\n[saved] {out / BUILDINGS}")


if __name__ == "__main__":
    main()
