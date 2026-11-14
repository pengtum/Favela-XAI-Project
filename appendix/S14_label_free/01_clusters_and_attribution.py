"""Step 1 of SI Section S14: label-free clusters of the 25 indicators, their alignment with
the favela labels and their contrast against the supervised attribution ranking.

Manuscript items
* SI Section S14 'Results', first paragraph: with all indicators weighted
  equally, the two clusters are separated chiefly by building shape (convexity
  among the three most separating indicators in every site), whereas the
  supervised attribution is led by spacing and adjacency; the two rankings are
  negatively correlated on average (Spearman's rho = -0.36), share their leading
  indicator in none of the ten sites, and the cluster contrast has no consistent
  direction across sites (mean pairwise cosine -0.10)
  (cluster_contrast_vs_attribution.csv, cluster_contrast_summary.csv).
* SI S14 'Results', second paragraph: correlation of the pooled normalised
  label-free profile (equal weights) with the supervised profile, 0.879 (also
  computed, with the other two weightings, by step 2).
* Per-k alignment of the equal-weight clusters with the labels
  (alignment_all_equal.csv; the best values over k and algorithm enter SI
  Table S18 through step 2).

Method (Main Methods 'Label-free clustering of the morphological indicators';
SI S14 'Design'; settings ``label_free``)
* Per site, the 25 indicators are median-imputed and rank-transformed to a
  standard normal distribution; buildings are clustered into 2-6 clusters by
  k-means and by Ward (fitted on at most 4,000 randomly drawn buildings, the
  rest assigned to the nearest centroid; as in label_free_common.cluster, the
  buildings of the subsample are assigned to their nearest Ward centroid as
  well). Labels are used only to score and to orient the result.
* Alignment: adjusted Rand index, normalised mutual information, accuracy and
  balanced accuracy of the majority assignment, and the best balanced accuracy
  over assignments of clusters to classes.
* Two-cluster contrast: k-means with two clusters; the contrast of an
  indicator is the difference between the centroid of the cluster with the
  larger favela share and the other centroid (rank-normal units). Its absolute
  values are compared with the supervised attribution ranking, the mean |SHAP|
  over the site's favela buildings (``importance.population``; the main model,
  Main Methods 4.6): Spearman correlation, leading indicator, three leading
  indicators. The direction of the contrast is compared between sites by the
  cosine of the 25-dimensional contrast vectors.
* Gradient: 6 m profile of the two-cluster score s = (d_other - d_fav) /
  (d_other + d_fav) and of the out-of-fold supervised probability (Main Methods
  4.7), normalised to 1 at d <= -50 m and 0 at d >= +50 m.

Inputs
* outputs/main/02_classification/<site>_buildings_shap.gpkg (products.shap_gpkg):
  label, 25 indicators, 25 shap_ columns of the main model.
* outputs/main/01_features/signed_distance.parquet (products.signed_distance).
* outputs/main/02_classification/oof_predictions.parquet (products.oof_predictions).

Outputs (outputs/appendix/S14_label_free/)
* alignment_all_equal.csv: site x method x k: ARI, NMI, acc_purity,
  bal_purity, bal_best.
* cluster_contrast_vs_attribution.csv: per site: Spearman rho, leading and three
  leading indicators of the cluster contrast and of the attribution, favela
  shares of the favela-like cluster.
* cluster_contrast_summary.csv: the SI S14 values (mean rho, sites sharing the
  leading indicator, indicators among the three leading cluster contrasts in
  every site, mean pairwise cosine, pooled profile correlation) and, for the
  log only, the minimum pairwise cosine, midpoints and 10-90% widths.
* profiles_all_equal.npz: per-site 6 m profiles of the score and of the
  supervised probability.

Run from the repository root:
    python appendix/S14_label_free/01_clusters_and_attribution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score

import label_free_common as lf
from favela_xai.attribution import favela_importance
from favela_xai.features import SHAP_COLUMNS
from favela_xai.io import write_table


def alignment(labels: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of a clustering with the favela labels."""
    k = int(labels.max()) + 1
    share = np.array([y[labels == c].mean() if (labels == c).any() else 0.0 for c in range(k)])
    majority = (share[labels] >= 0.5).astype(int)
    return dict(ARI=adjusted_rand_score(y, labels), NMI=normalized_mutual_info_score(y, labels),
                acc_purity=accuracy_score(y, majority), bal_purity=balanced_accuracy_score(y, majority),
                bal_best=lf.best_balanced_accuracy(labels, y))


def main() -> None:
    cfg = lf.cfg
    if lf.cfg.get("importance.population") != "favela_buildings":
        raise lf.ConfigError("only importance.population: favela_buildings is implemented (ED Fig. 2 caption)")
    features = lf.FEATURES
    rows, contrast_rows, contrasts = [], [], {}
    score_profiles, prob_profiles = [], []
    for site in cfg.sites:
        t = lf.site_table(site)
        y = t["y"].to_numpy(int)
        sd = t["signed_dist"].to_numpy(float)
        Z = lf.rank_normal(t[features].to_numpy(float))       # all indicators equal
        sub = lf.ward_subsample(len(Z))
        for k in lf.LF["n_clusters"]:
            for method in lf.LF["algorithms"]:
                labels = lf.cluster(Z, int(k), method, sub)
                rows.append(dict(site=site, weighting="all_equal", method=method, k=int(k), **alignment(labels, y)))

        # two-cluster contrast and score (labels only orient the result)
        score, km, fav_cluster = lf.two_cluster_score(Z, y)
        score_profiles.append(lf.profile(sd, score))
        prob_profiles.append(lf.profile(sd, t["prob"].to_numpy(float)))
        delta = km.cluster_centers_[fav_cluster] - km.cluster_centers_[1 - fav_cluster]
        contrasts[site] = delta
        imp = favela_importance(t[SHAP_COLUMNS], y == 1)["mean_abs_shap"].to_numpy()
        lab2 = km.labels_
        contrast_rows.append(dict(
            site=site, rho_cluster_contrast_vs_attribution=spearmanr(np.abs(delta), imp).statistic,
            top1_cluster=features[int(np.argmax(np.abs(delta)))], top1_attribution=features[int(np.argmax(imp))],
            top3_cluster="|".join(features[i] for i in np.argsort(-np.abs(delta))[:3]),
            top3_attribution="|".join(features[i] for i in np.argsort(-imp)[:3]),
            favela_share_in_favela_like_cluster=float(y[lab2 == fav_cluster].mean()),
            share_of_favela_in_favela_like_cluster=float((lab2[y == 1] == fav_cluster).mean()),
            share_of_formal_in_favela_like_cluster=float((lab2[y == 0] == fav_cluster).mean())))
        print(f"  {cfg.site_name(site)} done", flush=True)

    align = pd.DataFrame(rows)
    fe = pd.DataFrame(contrast_rows)
    write_table(align, lf.OUT / "alignment_all_equal.csv")
    write_table(fe, lf.OUT / "cluster_contrast_vs_attribution.csv")
    S, P = np.vstack(score_profiles), np.vstack(prob_profiles)
    np.savez(lf.OUT / "profiles_all_equal.npz", cen=lf.CEN, sites=np.array(cfg.sites), score=S, prob=P)

    # ------------------------------------------------------------ summary numbers (SI S14 'Results')
    n_sites = len(cfg.sites)
    top3 = fe["top3_cluster"].str.split("|")
    in_all = [f for f in features if all(f in t3 for t3 in top3)]
    V = np.array([contrasts[s] for s in cfg.sites])
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = Vn @ Vn.T
    off = C[~np.eye(n_sites, dtype=bool)] if n_sites > 1 else np.array([np.nan])
    ps, pp = lf.normalise(lf.pooled(S)), lf.normalise(lf.pooled(P))
    summary = [
        ("mean_spearman_rho", fe["rho_cluster_contrast_vs_attribution"].mean(), "SI S14: rho = -0.36"),
        ("n_sites_same_leading_indicator", int((fe["top1_cluster"] == fe["top1_attribution"]).sum()),
         "SI S14: none of the ten sites"),
        ("indicators_in_top3_cluster_contrast_in_every_site", ";".join(in_all), "SI S14: convexity"),
        ("mean_pairwise_cosine", float(np.nanmean(off)), "SI S14: -0.10"),
        ("min_pairwise_cosine", float(np.nanmin(off)), "log only"),
        ("pooled_profile_correlation_all_equal", lf.profile_correlation(lf.pooled(S), lf.pooled(P)),
         "SI S14 / Table S18: 0.879"),
        ("midpoint_label_free_m", lf.crossing(ps, 0.5), "log only"),
        ("width_10_90_label_free_m", lf.crossing(ps, 0.1) - lf.crossing(ps, 0.9), "log only"),
        ("midpoint_supervised_m", lf.crossing(pp, 0.5), "log only"),
        ("width_10_90_supervised_m", lf.crossing(pp, 0.1) - lf.crossing(pp, 0.9), "log only"),
    ]
    summary = pd.DataFrame(summary, columns=["quantity", "value", "manuscript"])
    write_table(summary, lf.OUT / "cluster_contrast_summary.csv")

    pd.set_option("display.width", 200)
    best = align.loc[align.groupby("site")["bal_best"].idxmax()].set_index("site")
    print("\nBest balanced accuracy of the equal-weight clusters (any k, k-means or Ward):")
    print(best[["method", "k", "ARI", "bal_best"]].round(3).to_string())
    print("\nCluster contrast vs supervised attribution, per site:")
    print(fe[["site", "rho_cluster_contrast_vs_attribution", "top3_cluster", "top3_attribution"]].round(3).to_string(index=False))
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
