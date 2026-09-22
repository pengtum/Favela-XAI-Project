"""Site-level importance of the 25 indicators and dominant-feature counts.

Manuscript items
* Results 2.2: cross-site importance ranking (mean interbuilding distance and
  adjacency lead) and the per-site leaders (squareness in Barão, perimeter wall
  in Grotão de Costa Barros, interbuilding distance and adjacency in Rocinha
  and Catumbi).
* Extended Data Fig. 2: values behind the per-site importance for the favela
  class, with one shared indicator order (the cross-site mean). The same shares
  are the importance measure of Extended Data Figs 3-4 and SI Table S16.
* Fig. 3a,b: number of favela buildings per dominant feature.
* SI Section S6, first paragraph: leading indicators and secondary tier.

Definitions (config.yaml, section 'importance')
* Importance of an indicator in a site: mean |SHAP| over the site's favela
  buildings, and its share of the site total.
* Dominant feature of a building: the indicator with the largest signed
  contribution toward favela.

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg
      (label and the 25 shap_<indicator> columns; positive = toward favela)

Outputs (outputs/main/03_site_signatures/)
  persite_importance_favela.csv  one row per site x indicator: mean_abs_shap,
                                 share, rank within the site (products.importance_favela)
  crosssite_importance.csv       one row per indicator: cross-site mean of the
                                 per-site values (ED Fig. 2 shared order) and the
                                 mean |SHAP| pooled over all buildings of all sites
                                 (the panel order of Fig. 4)
  persite_top3.csv               the three leading indicators of every site
  dominant_feature_counts.csv    one row per site x indicator: favela buildings
                                 whose dominant feature is that indicator

Run (from the repository root)
  python main/03_site_signatures/01_importance_tables.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.attribution import dominant_feature, favela_importance
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, dimension_of, label_of
from favela_xai.io import write_table
from favela_xai.labels import label_to_int


def read_shap_table(cfg, site: str) -> pd.DataFrame:
    """Label and the 25 SHAP columns of one site (no geometry); rows without SHAP dropped."""
    path = cfg.require_product("shap_gpkg", site=site)
    df = pyogrio.read_dataframe(path, columns=["label"] + SHAP_COLUMNS, read_geometry=False)
    df[SHAP_COLUMNS] = df[SHAP_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["label"] + SHAP_COLUMNS).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    if cfg.get("importance.population") != "favela_buildings":
        raise ValueError("importance.population must be 'favela_buildings' (ED Fig. 2 caption)")
    if cfg.get("importance.dominant_feature") != "argmax_signed":
        raise ValueError("importance.dominant_feature must be 'argmax_signed' (Fig. 3 caption)")

    imp_rows, dom_rows = [], []
    pooled_abs_sum = np.zeros(len(SHAP_COLUMNS))
    pooled_n = 0
    for site in cfg.sites:
        df = read_shap_table(cfg, site)
        fav = label_to_int(df["label"]).astype(bool)

        # Mean |SHAP| over the favela buildings and share of the site total
        imp = favela_importance(df, fav)
        imp["rank"] = imp["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
        for feat, r in imp.iterrows():
            imp_rows.append(dict(site=site, site_name=cfg.site_name(site), feature=feat,
                                 label=label_of(feat), dimension=dimension_of(feat),
                                 n_favela_buildings=int(fav.sum()), mean_abs_shap=r["mean_abs_shap"],
                                 share=r["share"], rank=int(r["rank"])))

        # Mean |SHAP| over all buildings, pooled over sites (Fig. 4 panel order)
        pooled_abs_sum += df[SHAP_COLUMNS].abs().sum().to_numpy()
        pooled_n += len(df)

        # Dominant feature of every favela building
        dom = dominant_feature(df.loc[fav])
        counts = dom.value_counts().reindex(MODEL_FEATURE_ORDER, fill_value=0)
        for feat, n in counts.items():
            dom_rows.append(dict(site=site, site_name=cfg.site_name(site), feature=feat,
                                 label=label_of(feat), dimension=dimension_of(feat),
                                 n_favela_buildings=int(n),
                                 share_of_favela_buildings=float(n) / max(int(fav.sum()), 1)))

        top3 = imp.sort_values("rank").head(3)
        print(f"  {cfg.site_name(site):24s} n = {len(df):>6d} ({int(fav.sum()):>6d} favela)  top 3: "
              + ", ".join(f"{label_of(f)} ({v:.2f})" for f, v in top3["mean_abs_shap"].items()))

    persite = pd.DataFrame(imp_rows)
    write_table(persite, cfg.product("importance_favela", create_parent=True))

    top3 = persite[persite["rank"] <= 3].sort_values(["site", "rank"])
    write_table(top3[["site", "site_name", "rank", "feature", "label", "dimension", "mean_abs_shap", "share"]],
                out / "persite_top3.csv")

    # Cross-site aggregates. ED Fig. 2 orders the indicators by the cross-site
    # mean of the per-site favela-building means; Fig. 4 groups and orders
    # its panels by the mean |SHAP| pooled over all buildings.
    wide = persite.pivot(index="feature", columns="site", values="mean_abs_shap")
    share_wide = persite.pivot(index="feature", columns="site", values="share")
    cross = pd.DataFrame({
        "feature": MODEL_FEATURE_ORDER,
        "label": [label_of(f) for f in MODEL_FEATURE_ORDER],
        "dimension": [dimension_of(f) for f in MODEL_FEATURE_ORDER],
        "crosssite_mean_abs_shap_favela": wide.reindex(MODEL_FEATURE_ORDER).mean(axis=1).to_numpy(),
        "crosssite_mean_share_favela": share_wide.reindex(MODEL_FEATURE_ORDER).mean(axis=1).to_numpy(),
        "pooled_mean_abs_shap_all_buildings": pooled_abs_sum / max(pooled_n, 1),
    })
    cross["crosssite_rank"] = cross["crosssite_mean_abs_shap_favela"].rank(ascending=False, method="first").astype(int)
    cross["pooled_rank"] = cross["pooled_mean_abs_shap_all_buildings"].rank(ascending=False, method="first").astype(int)
    cross = cross.sort_values("crosssite_rank").reset_index(drop=True)
    write_table(cross, out / "crosssite_importance.csv")

    write_table(pd.DataFrame(dom_rows), out / "dominant_feature_counts.csv")

    print("\nCross-site order (ED Fig. 2), top 5: "
          + ", ".join(f"{r.label} ({r.crosssite_mean_abs_shap_favela:.2f})" for r in cross.head(5).itertuples()))
    pooled = cross.sort_values("pooled_rank")
    print("Pooled over all buildings, top 5:  "
          + ", ".join(f"{r.label} ({r.pooled_mean_abs_shap_all_buildings:.2f})" for r in pooled.head(5).itertuples()))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
