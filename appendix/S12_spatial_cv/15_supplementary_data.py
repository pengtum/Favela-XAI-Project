"""Step 15: the per-fold record of the spatial validation (Supplementary Data 1 and 2).

Manuscript items: Supplementary Data 1 and 2, the per-fold record (accuracy, AUC,
favela-class F1, class prevalence, training and test sizes) referred to in the SI
Table S14 caption.

Supplementary Data 1: every fold of the square-block, alternative-unit, random and
pooled leave-site-out designs (8,020 rows with the ten sites). Supplementary Data 2:
every fold of the held-out favela component (116 rows). Both carry two scoring columns
(SI S12 'Scoring'):
  'Fold used'              the fold was fitted and its training set holds at least 50
                           buildings of the minority class (first criterion);
  'Site scored in design'  the used folds of this site, design, buffer and grid placement
                           give a prediction to at least half of the site's favela
                           buildings (second criterion); matched-control rows carry the value
                           of the buffered design; the random 80/20 split and the pooled
                           leave-site-out are scored for every site.
A reported value = rows with both columns True; per site the mean over rows, then the
mean over sites.

Inputs  outputs/appendix/S12_spatial_cv/folds_all.parquet (step 04); site tables (step 01)
Outputs outputs/appendix/S12_spatial_cv/supplementary_data/Supplementary_Data_1_spatialcv_folds.csv,
        Supplementary_Data_2_component_folds.csv
Run     python appendix/S12_spatial_cv/15_supplementary_data.py
"""
import numpy as np
import pandas as pd

from favela_xai import spatial_cv as scv

import s12_common as C

cfg, s, P = C.setup(__file__)
OUT = P.out / "supplementary_data"
OUT.mkdir(exist_ok=True)
F = C.read_folds(cfg)
TOT = C.favela_totals(cfg, P)
NAMES = {site: cfg.site_name(site) for site in cfg.all_sites}
DESIGN = {"sq100": "Square blocks 100 m, 5 folds", "sq250": "Square blocks 250 m, 5 folds",
          "sq500": "Square blocks 500 m, 5 folds", "sq1000": "Square blocks 1,000 m, 5 folds",
          "sq250_K10": "Square blocks 250 m, 10 folds", "sq500_K10": "Square blocks 500 m, 10 folds",
          "hex250": "Hexagons (area of a 250 m square), 5 folds", "hex500": "Hexagons (area of a 500 m square), 5 folds",
          "kmm250": "k-means clusters (about one per 250 m square), 5 folds",
          "kmm500": "k-means clusters (about one per 500 m square), 5 folds",
          "km5": "Five k-means clusters used as the folds",
          "random_80_20": "Random 80/20 split (refitted in this pipeline)",
          "random_skf5": "Random stratified five-fold cross-validation",
          "loso_pooled": "Pooled leave-site-out (trained on the other nine sites)"}
UNIT = {"square": "square block", "hexagon": "hexagon", "kmeans_many": "k-means cluster (many per site)",
        "kmeans5": "k-means cluster (five per site)", "random": "none (random)", "site": "site"}
STATUS = {"ok": "fitted",
          "ok_no_drop_same_as_blocked": "fitted; buffer removed no training building, so the control equals the buffered fold",
          "skipped_empty_test": "not fitted: no test building",
          "skipped_train_lt50": "not fitted: fewer than 50 training buildings",
          "skipped_single_class_train": "not fitted: training set holds a single class",
          "skipped_block_fold_skipped": "not fitted: the paired buffered fold was not fitted"}


def scored_flags(df, unit):
    """Second criterion per design x buffer (per site and grid placement)."""
    flag = pd.Series(False, index=df.index)
    for _, g in df.groupby(["design", "buffer"], dropna=False):
        if g.design.iloc[0] in ("random_80_20", "loso_pooled"):
            flag[g.index] = True
            continue
        _, cov = scv.apply_rules(g[~g.control], TOT, s, unit=unit)
        sc = cov.set_index(unit).scored
        keys = list(zip(*[g[u] for u in unit])) if len(unit) > 1 else list(g[unit[0]])
        flag[g.index] = [bool(sc.get(k, False)) for k in keys]
    return flag


def train_counts(df):
    fav = (df.n_train * df.prev_train).where(df.n_train > 0, 0.0).round()
    return fav.astype(int), (df.n_train - fav).astype(int)


def col(df, name):
    return df[name] if name in df else np.nan


# ----------------------------------------------------------------------------- Supplementary Data 1
S = F[~F.design.isin(["component", "random_80_20_stratified"])].copy()
S["fold_used"] = scv.rule1_mask(S, s).to_numpy()
S["site_scored"] = scored_flags(S, ["site", "repeat"]).to_numpy()
fav, frm = train_counts(S)
SD1 = pd.DataFrame({
    "Site": S.site.map(NAMES), "Design": S.design.map(DESIGN), "Spatial unit": S.block_rule.map(UNIT),
    "Unit size L (m)": S.L, "Folds in design (K)": S.K, "Spatial units in site": S.n_units,
    "k-means clusters": col(S, "n_clusters"), "Grid offset x (m)": col(S, "offset_x"),
    "Grid offset y (m)": col(S, "offset_y"), "Grid placement": S["repeat"] + 1, "Fold": S.fold + 1,
    "Buffer (m)": S.buffer, "Matched control": S.control, "Fit status": S.status.map(STATUS),
    "Training buildings": S.n_train, "Favela training buildings": fav, "Formal-settlement training buildings": frm,
    "Test buildings": S.n_test, "Training buildings removed": S.n_dropped,
    "Favela share, training set": S.prev_train, "Favela share, test set": S.prev_test,
    "Accuracy": S.accuracy, "Balanced accuracy": S.balanced_accuracy, "F1 (favela)": S.f1_favela,
    "F1 (formal settlement)": S.f1_formal, "AUC": S.auc, "Brier score": S.brier,
    "Median distance from test to nearest training building (m)": S.d_test_train_median,
    "90th percentile distance from test to nearest training building (m)": S.d_test_train_p90,
    "Minimum distance from test to nearest training building (m)": col(S, "d_test_train_min"),
    "Nearest training site": col(S, "nearest_train_site").map(NAMES) if "nearest_train_site" in S else np.nan,
    "Distance to nearest training site (m)": col(S, "nearest_train_site_dist"),
    "Fold used": S.fold_used, "Site scored in design": S.site_scored})
dist = [c for c in SD1.columns if c.endswith("(m)") and "distance" in c.lower()]
SD1[dist] = SD1[dist].round(2)
SD1.to_csv(OUT / "Supplementary_Data_1_spatialcv_folds.csv", index=False, float_format="%.6g")

# ----------------------------------------------------------------------------- Supplementary Data 2
L = F[F.design == "component"].copy().sort_values(["site", "buffer", "fold"])
L["fold_used"] = scv.rule1_mask(L, s).to_numpy()
L["site_scored"] = scored_flags(L, ["site"]).to_numpy()
fav, frm = train_counts(L)
SD2 = pd.DataFrame({
    "Site": L.site.map(NAMES), "Design": "Held-out favela component", "Buffer (m)": L.buffer.astype(int),
    "Fold": L.fold + 1, "Anchor favela polygon (2013 polygon code)": L.test_group.astype(int),
    "Favela polygons in held-out group (2013 polygon codes)": L.test_components.astype(str),
    "Fit status": L.status.map(STATUS), "Training buildings": L.n_train, "Favela training buildings": fav,
    "Formal-settlement training buildings": frm, "Test buildings": L.n_test, "Training buildings removed": L.n_dropped,
    "Favela share, training set": L.prev_train, "Favela share, test set": L.prev_test, "Accuracy": L.accuracy,
    "Balanced accuracy": L.balanced_accuracy, "F1 (favela)": L.f1_favela, "AUC": L.auc,
    "Fold used": L.fold_used, "Site scored in design": L.site_scored})
SD2.to_csv(OUT / "Supplementary_Data_2_component_folds.csv", index=False, float_format="%.6g")
print(f"Supplementary Data 1: {len(SD1):,} rows; Supplementary Data 2: {len(SD2):,} rows "
      f"(expected with ten sites: 8,020 and 116)")
