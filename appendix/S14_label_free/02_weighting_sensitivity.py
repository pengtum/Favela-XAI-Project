"""Step 2 of SI Section S14: label-free clustering under the three weightings of the indicators.

Manuscript items
* SI Table S18, rows 'All indicators equal', 'Dimensions balanced', 'One
  component per dimension' (means over the ten sites): best ARI 0.045 / 0.079 /
  0.057; best balanced accuracy 0.628 / 0.646 / 0.633; score AUC 0.555 / 0.631 /
  0.616; step 0.173 / 0.475 / 0.442 SD; profile r 0.879 / 0.942 / 0.952
  (weighting_sensitivity.csv, weighting_summary.csv; table: 03_table_S18.py).
* SI S14 'Results': mean best adjusted Rand index at most 0.08, mean best
  balanced accuracy 0.63-0.65, mean score AUC 0.55-0.63, inside-outside
  difference 0.17-0.48 SD of the score; per-site correlation with the
  supervised profile, median 0.508 (range 0.060-0.834) with equal weights and
  0.778 (-0.157 to 0.976) under the inverse-square-root weighting
  (persite_profile_correlation.csv, weighting_summary.csv).
* Data of SI Fig. S23 (panel a profiles, panel b label-free bars; 04_figure_S23.py).

Method (SI S14 'Design'; settings ``label_free``)
* Weightings (label_free_common.weighted_space): all 25 indicators equal; each
  indicator scaled by 1/sqrt(size of its dimension) ("dimensions balanced");
  the standardised first principal component of each dimension ("one
  component per dimension").
* For every weighting: k-means and Ward with 2-6 clusters; best adjusted Rand
  index and best balanced accuracy over k and algorithm (and over assignments
  of clusters to classes); orientation-free AUC max(AUC, 1 - AUC) of the
  two-cluster score; 6 m profile of the score; step = (mean of the profile at
  d <= -50 m minus mean at d >= +50 m) / s.d. of the score.
* Supervised reference: the out-of-fold favela probability of each site from
  five-fold cross-validation (Main Methods 4.7) and its 6 m profile. The column
  sup_AUC is the AUC of the pooled out-of-fold probability of a site (for
  information); the reference AUC and balanced accuracy of SI Table S18 are the
  means over folds and sites of the SI S12 random five-fold rows
  (03_table_S18.py).
* Profile correlation: Pearson correlation of the normalised profiles over the
  bins available in both, pooled (mean of the per-site profiles) and per site.

Inputs: as step 1 (products.shap_gpkg, products.signed_distance,
products.oof_predictions).
Outputs (outputs/appendix/S14_label_free/)
* weighting_sensitivity.csv: site x weighting: best_bal_acc, best_ARI,
  score_AUC, sup_AUC, step_in_sd, per-site profile correlation.
* persite_profile_correlation.csv: the per-site profile correlations in wide form.
* weighting_summary.csv: per weighting, the means over sites, the pooled profile
  correlation and the median / range of the per-site correlations.
* profiles_by_weighting.npz: per-site 6 m profiles of the supervised
  probability ('sup') and of the score under each weighting.

Run from the repository root:
    python appendix/S14_label_free/02_weighting_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score

import label_free_common as lf
from favela_xai.io import write_table


def main() -> None:
    cfg = lf.cfg
    rows = []
    prof = {w: [] for w in lf.WEIGHTINGS}
    sup = []
    for site in cfg.sites:
        t = lf.site_table(site)
        y = t["y"].to_numpy(int)
        sd = t["signed_dist"].to_numpy(float)
        prob = t["prob"].to_numpy(float)
        Z = lf.rank_normal(t[lf.FEATURES].to_numpy(float))
        sup_auc = roc_auc_score(y, prob)
        sup_profile = lf.profile(sd, prob)
        sup.append(sup_profile)
        sub = lf.ward_subsample(len(Z))
        for weighting in lf.WEIGHTINGS:
            S = lf.weighted_space(Z, weighting)
            best_bal, best_ari = 0.5, 0.0
            for k in lf.LF["n_clusters"]:
                for method in lf.LF["algorithms"]:
                    labels = lf.cluster(S, int(k), method, sub)
                    best_bal = max(best_bal, lf.best_balanced_accuracy(labels, y))
                    best_ari = max(best_ari, adjusted_rand_score(y, labels))
            score, _, _ = lf.two_cluster_score(S, y)
            auc = roc_auc_score(y, score)
            p = lf.profile(sd, score)
            prof[weighting].append(p)
            rows.append(dict(site=site, weighting=weighting, best_bal_acc=best_bal, best_ARI=best_ari,
                             score_AUC=max(auc, 1 - auc), sup_AUC=sup_auc,
                             step_in_sd=lf.plateau_step(p) / np.nanstd(score),
                             profile_r_site=lf.profile_correlation(p, sup_profile)))
        print(f"  {cfg.site_name(site)} done", flush=True)

    res = pd.DataFrame(rows)
    write_table(res, lf.OUT / "weighting_sensitivity.csv")
    write_table(res.pivot(index="site", columns="weighting", values="profile_r_site")[lf.WEIGHTINGS].reset_index(),
                lf.OUT / "persite_profile_correlation.csv")
    np.savez(lf.OUT / "profiles_by_weighting.npz", cen=lf.CEN, sites=np.array(cfg.sites), sup=np.vstack(sup),
             **{w: np.vstack(prof[w]) for w in lf.WEIGHTINGS})

    # ------------------------------------------------------------ summary per weighting
    sup_pooled = lf.pooled(np.vstack(sup))
    summary = []
    for w in lf.WEIGHTINGS:
        g = res[res["weighting"] == w]
        summary.append(dict(
            weighting=w, weighting_name=lf.WEIGHTING_NAMES[w],
            best_ARI_mean=g["best_ARI"].mean(), best_bal_acc_mean=g["best_bal_acc"].mean(),
            score_AUC_mean=g["score_AUC"].mean(), sup_AUC_mean=g["sup_AUC"].mean(),
            step_in_sd_mean=g["step_in_sd"].mean(),
            profile_r_pooled=lf.profile_correlation(lf.pooled(np.vstack(prof[w])), sup_pooled),
            profile_r_site_median=g["profile_r_site"].median(), profile_r_site_min=g["profile_r_site"].min(),
            profile_r_site_max=g["profile_r_site"].max()))
    summary = pd.DataFrame(summary)
    write_table(summary, lf.OUT / "weighting_summary.csv")

    pd.set_option("display.width", 220)
    print("\n" + summary.round(3).to_string(index=False))
    print("\nReported in SI S14 / Table S18: ARI 0.045/0.079/0.057, bal. acc. 0.628/0.646/0.633, score AUC 0.555/0.631/0.616, "
          "step 0.173/0.475/0.442, profile r 0.879/0.942/0.952, supervised AUC 0.969; per-site r median "
          "0.508 (0.060-0.834) equal weights, 0.778 (-0.157 to 0.976) dimensions balanced.")
    print("\nScore AUC per site:")
    print(res.pivot(index="site", columns="weighting", values="score_AUC")[lf.WEIGHTINGS]
          .assign(supervised=res.groupby("site")["sup_AUC"].first()).round(3).to_string())


if __name__ == "__main__":
    main()
