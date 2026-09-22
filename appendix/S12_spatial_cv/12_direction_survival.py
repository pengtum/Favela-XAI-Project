"""Step 12: direction of the attributions under spatial blocking (SI S12, last paragraph of 'Stability').

Manuscript items: the direction-survival values of SI S12 'Stability of the
attributions': the three leading indicators keep their main-text sign in 73 of 81
site-indicator-placement combinations (site mean 0.90, 0.83-0.96), the five leading
indicators in 126 of 135 (0.93, 0.87-0.98), and the indicators in which the mismatches
occur.

Direction of indicator j in a site = sign of Spearman's correlation, over favela
buildings, between the raw indicator value and its SHAP value (positive SHAP = toward
favela). Main-text direction: SHAP of the main per-site model over all favela buildings
of the site. Refit direction: out-of-fold SHAP of the primary design (250 m blocks,
200 m buffer, three grid placements; step 10) over the favela buildings explained by
folds that pass the first scoring criterion. Leading indicators: the k largest mean
|SHAP| of the main-text model over the site's favela buildings
(``appendix_S12_spatial_cv.ranking.direction_top_k``). A site x placement is scored
when the explained favela buildings are at least half of the site's (second
criterion); per site the mean over its scored placements; across sites the mean with a
site-bootstrap 95% interval (2,000 resamples).

Inputs  outputs/appendix/S12_spatial_cv/shap_buildings/<site>__sq250_b200_p<r>.npz,
        shap_fold_sums/ (step 10); site tables and main-model SHAP (step 01)
Outputs outputs/appendix/S12_spatial_cv/direction/ direction_by_site_indicator.csv, direction_by_run.csv,
          direction_summary.csv, direction_mismatches.csv, in_text_numbers.txt
Run     python appendix/S12_spatial_cv/12_direction_survival.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import s12_common as C

cfg, s, P = C.setup(__file__)
RK = C.section(cfg)["ranking"]
OUT = P.out / "direction"
OUT.mkdir(exist_ok=True)
log = C.log_to(P.logs / "12_direction_survival.log")
FE = C.FEATURES
TOPK = [int(v) for v in RK["direction_top_k"]]
MIN_N = int(RK["direction_min_buildings"])
PRIM_L, PRIM_B = int(cfg.get("spatial_cv.primary.block_m")), int(cfg.get("spatial_cv.primary.buffer_m"))
PL = list(range(int(cfg.get("spatial_cv.grid_placements"))))
FS = pd.concat([pd.read_parquet(f) for f in sorted(P.shap_folds.glob(f"*__sq{PRIM_L}_b{PRIM_B}_p*.parquet"))
                if "control_" not in f.name], ignore_index=True)


def slope_sign(x, v):
    """Spearman correlation of raw values and SHAP values (NaN with fewer than MIN_N complete pairs)."""
    ok = np.isfinite(x) & np.isfinite(v)
    return spearmanr(x[ok], v[ok]).correlation if ok.sum() >= MIN_N else np.nan


rows = []
for site in cfg.sites:
    d, main = C.load_site(P, site), C.load_main_shap(P, site)
    fav_all = d.y == 1
    mean_abs = np.abs(main[fav_all]).mean(0)
    rank_main = np.empty(len(FE), int)
    rank_main[np.argsort(-mean_abs)] = np.arange(1, len(FE) + 1)
    rho_main = np.array([slope_sign(d.X[fav_all, j], main[fav_all, j]) for j in range(len(FE))])
    for r in PL:
        run = f"sq{PRIM_L}_b{PRIM_B}_p{r}"
        z = np.load(C.require(P.shap_buildings / f"{site}__{run}.npz", "10_oof_shap.py"))
        S, fold_of = z["S"].astype(float), z["fold_of"]
        f = FS[(FS.site == site) & (FS.run == run)]
        drop = f.loc[~f.rule1_ok.astype(bool), "fold"].tolist()          # first criterion
        S[np.isin(fold_of, drop)] = np.nan
        ok = ~np.isnan(S).any(1)
        fav = ok & fav_all
        frac = fav.sum() / fav_all.sum()
        for j, c in enumerate(FE):
            rows.append(dict(site=site, placement=r + 1, indicator=c, rank_main=int(rank_main[j]),
                             rho_main_favela=rho_main[j],
                             rho_refit_favela=slope_sign(d.X[fav, j], S[fav, j]) if fav.sum() else np.nan,
                             favela_explained_frac=frac, n_favela_explained=int(fav.sum())))
D = pd.DataFrame(rows)
D["same"] = (np.sign(D.rho_main_favela) == np.sign(D.rho_refit_favela)) & D.rho_refit_favela.notna() & \
            (D.rho_refit_favela != 0)
D["scored_first"] = D.n_favela_explained > 0
D["scored_both"] = D.favela_explained_frac >= s.min_favela_coverage
D.to_csv(OUT / "direction_by_site_indicator.csv", index=False)

run_rows = []
for (site, pl), g in D.groupby(["site", "placement"], sort=False):
    r = dict(site=site, placement=pl, favela_explained_frac=g.favela_explained_frac.iloc[0],
             scored_first=bool(g.scored_first.iloc[0]), scored_both=bool(g.scored_both.iloc[0]))
    for k in TOPK:
        t = g[g.rank_main <= k]
        r[f"top{k}_kept"] = int(t.same.sum())
        r[f"top{k}_share"] = float(t.same.mean())
    run_rows.append(r)
RUN = pd.DataFrame(run_rows)
RUN.to_csv(OUT / "direction_by_run.csv", index=False)

summ, lines = [], ["SI S12 direction survival (250 m blocks, 200 m buffer, three grid placements)"]
for sc, col in (("both criteria (primary)", "scored_both"), ("first criterion only", "scored_first")):
    Rn = RUN[RUN[col]]
    Si = Rn.groupby("site")[[f"top{k}_share" for k in TOPK]].mean()
    for k in TOPK:
        m, lo, hi, n = C.boot_mean(Si[f"top{k}_share"].to_numpy(), s)
        kept, pairs = int(Rn[f"top{k}_kept"].sum()), len(Rn) * k
        summ.append(dict(scoring=sc, top_k=k, n_sites=n, n_site_placements=len(Rn), site_mean_share=m, ci_lo=lo,
                         ci_hi=hi, pairs_kept=kept, pairs=pairs))
        if sc.startswith("both"):
            lines.append(f"top {k}: {kept} of {pairs} site-indicator-placement combinations; site mean {m:.2f} "
                         f"({lo:.2f}-{hi:.2f}), {n} sites")
pd.DataFrame(summ).to_csv(OUT / "direction_summary.csv", index=False)
lines.append("(manuscript: 73 of 81, 0.90 (0.83-0.96); 126 of 135, 0.93 (0.87-0.98))")
top5 = D[D.scored_both & (D.rank_main <= max(TOPK))]
mm = top5[~top5.same]
mm.to_csv(OUT / "direction_mismatches.csv", index=False)
lines.append("mismatches among the five leading indicators: "
             + ", ".join(f"{k} {v}" for k, v in mm.indicator.value_counts().items()))
always = sorted(set(top5.indicator) - set(mm.indicator))
lines.append("indicators that keep their direction wherever they are among the five leading ones: " + ", ".join(always))
lines.append("(manuscript: mismatches concentrate in adjacency and perimeter wall; mean interbuilding distance, "
             "squareness and average weighted distance keep their direction everywhere)")
(OUT / "in_text_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
for ln in lines:
    log(ln)
