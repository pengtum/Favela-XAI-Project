"""Pooled, standardised PCA of the Tree-SHAP attributions (the attribution plane).

Manuscript items
* Methods 4.6: each of the 25 attribution columns standardised over the
  buildings of all ten sites, one PCA fitted to the pooled matrix, first two
  components kept.
* Fig. 5: explained variance of PC1 and PC2 (17.9% and 8.5%, printed at the
  end of the run); the scores are drawn by 02_figure5.py.
* Results 2.4: within every site the favela and formal-settlement clouds are
  displaced in their centres of mass (per-site class means of PC1 and PC2 in
  persite_summary.csv).
* SI Section S7 and SI Fig. S10: the PC1 scores written here are thresholded
  within each site at the ROC-Youden optimum in appendix/S07_pca.

Method (config.yaml, section 'pca')
* Input: the 25 Tree-SHAP columns of every building of the ten sites (per-site
  models of Methods 4.4; positive = toward favela), stacked in the site order
  of config.yaml, columns in favela_xai.features.MODEL_FEATURE_ORDER.
  Buildings with a missing attribution are left out.
* Standardisation: z-score of each column over the pooled buildings with the
  population standard deviation (scikit-learn StandardScaler), then one PCA
  with all 25 components (svd_solver 'full'); PC1 and PC2 are kept.
* Component signs are fixed explicitly: pca.orient_pc1 = favela_positive (the
  pooled favela mean of PC1 above the formal mean); pca.orient_pc2 = largest
  loading positive.

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg
      (label, the 25 shap_<indicator> columns, uID when present)

Outputs (outputs/main/05_attribution_plane/)
  attribution_pca_scores.parquet  one row per building: site, row (position in
                                  the main/02 table), uID (when present), label,
                                  is_favela, PC1, PC2 (products.attribution_pca_scores)
  explained_variance.csv          25 components: explained variance ratio,
                                  cumulative ratio, eigenvalue, sign flipped
  loadings.csv                    component x indicator: eigenvector and loading
                                  (eigenvector x sqrt(eigenvalue))
  scaler_params.csv               per indicator: pooled mean and standard deviation
                                  of its SHAP column (the standardisation)
  persite_summary.csv             per site: buildings, favela buildings, class
                                  means of PC1 and PC2, PC1-only AUC

Run (from the repository root)
  python main/05_attribution_plane/01_attribution_pca.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio

from sklearn.metrics import roc_auc_score

from favela_xai.attribution import attribution_pca
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, dimension_of, label_of
from favela_xai.io import write_table
from favela_xai.labels import label_to_int


def read_site_attributions(cfg, site: str) -> pd.DataFrame:
    """Label, building id and the 25 SHAP columns of one site, in the row order of main/02."""
    path = cfg.require_product("shap_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    id_cols = ["uID"] if "uID" in fields else []
    df = pyogrio.read_dataframe(path, columns=id_cols + ["label"] + SHAP_COLUMNS, read_geometry=False)
    df[SHAP_COLUMNS] = df[SHAP_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df.insert(0, "row", np.arange(len(df)))
    df.insert(0, "site", site)
    n_missing = int(df[SHAP_COLUMNS].isna().any(axis=1).sum())
    if n_missing:
        print(f"  note: {site}: {n_missing} buildings without complete attributions left out")
    return df.dropna(subset=["label"] + SHAP_COLUMNS).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    p = cfg.section("pca")
    if not p["pooled_over_sites"]:
        raise ValueError("Methods 4.6: a single PCA fitted to the pooled matrix (pca.pooled_over_sites: true)")
    if not p["standardise"]:
        print("warning: pca.standardise is false; Methods 4.6 standardises the 25 attribution columns")
    n_keep = int(p["components_kept"])

    # ---- pooled attribution matrix of the ten sites --------------------------
    parts = []
    for site in cfg.sites:
        df = read_site_attributions(cfg, site)
        parts.append(df)
        print(f"  {cfg.site_name(site):24s} {len(df):>7d} buildings")
    pooled = pd.concat(parts, ignore_index=True)
    is_favela = label_to_int(pooled["label"]).astype(bool)
    X = pooled[SHAP_COLUMNS].to_numpy(float)
    print(f"pooled matrix: {X.shape[0]:,} buildings x {X.shape[1]} attribution columns")

    # ---- standardise + PCA (favela_xai.attribution.attribution_pca) ----------
    res = attribution_pca(X, is_favela, cfg)
    scores = res["scores"]
    evr = res["explained_variance_ratio"]
    # Eigenvalues (variance of each component, ddof 1 as in scikit-learn)
    eigen = scores.var(axis=0, ddof=1)

    # ---- per-building scores (the first components_kept components) ---------
    cols = ["site", "row"] + (["uID"] if "uID" in pooled.columns else []) + ["label"]
    table = pooled[cols].copy()
    table["is_favela"] = is_favela
    for k in range(n_keep):
        table[f"PC{k + 1}"] = scores[:, k]
    write_table(table, cfg.product("attribution_pca_scores", create_parent=True))

    # ---- explained variance, loadings, standardisation -----------------------
    comps = pd.DataFrame({
        "component": evr.index,
        "explained_variance_ratio": evr.to_numpy(),
        "cumulative_ratio": np.cumsum(evr.to_numpy()),
        "eigenvalue": eigen,
        "sign_flipped": [int(c[2:]) in res["flipped"] for c in evr.index],
    })
    write_table(comps, out / "explained_variance.csv")

    vec = res["loadings"]                                   # components x indicators (eigenvectors)
    rows = []
    for k, comp in enumerate(vec.index):
        for f in MODEL_FEATURE_ORDER:
            rows.append(dict(component=comp, feature=f, label=label_of(f), dimension=dimension_of(f),
                             eigenvector=float(vec.loc[comp, f]),
                             loading=float(vec.loc[comp, f] * np.sqrt(eigen[k]))))
    write_table(pd.DataFrame(rows), out / "loadings.csv")

    write_table(pd.DataFrame({"feature": MODEL_FEATURE_ORDER,
                              "label": [label_of(f) for f in MODEL_FEATURE_ORDER],
                              "mean": res["scaler_mean"].to_numpy(),
                              "scale": res["scaler_scale"].to_numpy()}), out / "scaler_params.csv")

    # ---- per site: class means of PC1 and PC2 and the PC1-only AUC -----------
    # (the per-site ROC-Youden thresholds of SI Fig. S10 are computed in appendix/S07_pca)
    summary = []
    for site in cfg.sites:
        m = (table["site"] == site).to_numpy()
        fav = table.loc[m, "is_favela"].to_numpy()
        pc1 = table.loc[m, "PC1"].to_numpy()
        pc2 = table.loc[m, "PC2"].to_numpy() if n_keep > 1 else np.full(m.sum(), np.nan)
        row = dict(site=site, site_name=cfg.site_name(site), n=int(m.sum()), n_favela=int(fav.sum()),
                   mean_PC1_favela=float(pc1[fav].mean()), mean_PC1_formal=float(pc1[~fav].mean()),
                   mean_PC2_favela=float(pc2[fav].mean()), mean_PC2_formal=float(pc2[~fav].mean()),
                   favela_mean_PC1_above_formal=bool(pc1[fav].mean() > pc1[~fav].mean()))
        if 0 < fav.sum() < len(fav):
            row.update(auc_PC1=float(roc_auc_score(fav.astype(int), pc1)))
        summary.append(row)
    summary = pd.DataFrame(summary)
    write_table(summary, out / "persite_summary.csv")

    # ---- summary for Fig. 5 and Results 2.4 ------------------------------------
    print(f"\nexplained variance: PC1 {100 * evr.iloc[0]:.1f}%, PC2 {100 * evr.iloc[1]:.1f}% "
          "(Fig. 5 caption: 17.9% and 8.5%)")
    print("components with reversed sign:", res["flipped"] or "none")
    top = vec.iloc[0].mul(np.sqrt(eigen[0])).sort_values(ascending=False).head(5)
    print("largest PC1 loadings: " + ", ".join(f"{label_of(f)} {v:.2f}" for f, v in top.items()))
    n_disp = int(summary["favela_mean_PC1_above_formal"].sum())
    print(f"favela mean PC1 above formal mean in {n_disp} of {len(summary)} sites")
    if "auc_PC1" in summary:
        print(f"PC1-only AUC {summary['auc_PC1'].min():.3f}-{summary['auc_PC1'].max():.3f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
