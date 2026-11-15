"""Step 10: out-of-fold Tree SHAP under the spatial designs (attribution side of SI S12).

Manuscript items: the attribution runs behind SI S12 'Stability of the attributions'
(SI Table S16; Extended Data Figs 3-4; the ranking ladder, the site-matched contrasts,
the alternative units and the direction survival quoted in SI S12).

For every site and run the folds of the design are rebuilt with the same fold makers
as the accuracy runs (s12_common.make_design, favela_xai.spatial_cv), each fold is
fitted with the validation-pipeline model (XGBoost, 100 trees, seed 432, training-
median imputation; SI S12 'Designs') and its test buildings are explained with exact
Tree SHAP (log-odds contributions, positive toward favela). Runs
(``appendix_S12_spatial_cv.ranking``, seeds from ``spatial_cv.ranking``):
  sq<L>_b<b>_p0         square blocks of 100-1,000 m, buffers 0, 200, 400 m, first grid placement
  sq<L>_b<b>_p1, _p2    the other two grid placements of 250 m / 0 m and of 250, 500, 1,000 m / 200 m
  skf5_s<seed>          random stratified five-fold cross-validation, shuffle seeds 432, 7, 99, 2024
  null_s<seed>          split-refit null: one model refitted on another random 80/20 split of the
                        site (split seeds 7, 99, 2024), Tree SHAP of every building of the site;
                        spatial_cv.ranking.null_classifier selects the model:
                          model_cv    validation-pipeline model on an unstratified split
                          model_main  main per-site model of Methods 4.4 on a split drawn as
                                      in model_main (unstratified)
  component_b0, _b200   held-out favela component without and with the 200 m buffer
  hex250_b200, kmm250_b200, sq250_K10_b200   the alternative units at 200 m (first placement)
  control_sq250_b200_p<r>  size-matched control: the same test folds, the buffer's number of
                        training buildings removed at random from the fold (SI S12 ladder)

Written per fold (not per building, so that any subset of folds can be aggregated later):
fold status, training class counts (first scoring criterion), test counts, and the sums of
|SHAP| over the fold's favela test buildings for the out-of-fold model (abs_oof_<indicator>)
and for the main per-site model on the same buildings (abs_main_<indicator>). The mean
|SHAP| over the explained favela buildings of any set of folds, and hence the
importance shares and rankings of SI S12, follow from these sums. The per-building
out-of-fold SHAP of the primary design (250 m blocks, 200 m buffer, three placements)
is also kept for the direction survival (step 12).

Inputs  outputs/appendix/S12_spatial_cv/site_tables/ (indicators, main-model SHAP; step 01),
        component_groups/ (step 02)
Outputs outputs/appendix/S12_spatial_cv/shap_fold_sums/<site>__<run>.parquet
        outputs/appendix/S12_spatial_cv/shap_buildings/<site>__sq250_b200_p<r>.npz (S, fold_of)
Run     python appendix/S12_spatial_cv/10_oof_shap.py [--sites ...] [--runs ...] [--n_jobs N] [--status]
"""
import argparse
import datetime
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from favela_xai import spatial_cv as scv
from favela_xai.attribution import tree_shap
from favela_xai.model import fit_classifier, fit_predict_cv, formal_as_class_1, impute_training_median

import s12_common as C

cfg, s, P = C.setup(__file__)
RK = C.section(cfg)["ranking"]
SR = cfg.section("spatial_cv")["ranking"]
FORMAL_AS_1 = formal_as_class_1(cfg, "model_cv")   # SHAP of such models is negated
PRIM_L, PRIM_B = int(cfg.get("spatial_cv.primary.block_m")), int(cfg.get("spatial_cv.primary.buffer_m"))
CTL_L, CTL_B = int(RK["control_design"]["block_m"]), int(RK["control_design"]["buffer_m"])
PLACEMENTS = list(range(int(cfg.get("spatial_cv.grid_placements"))))
NULL_CLASSIFIER = SR.get("null_classifier", "model_cv")      # model_cv | model_main
if NULL_CLASSIFIER not in ("model_cv", "model_main"):
    raise ValueError("spatial_cv.ranking.null_classifier must be 'model_cv' or 'model_main'")
ABS_OOF = ["abs_oof_" + f for f in C.FEATURES]
ABS_MAIN = ["abs_main_" + f for f in C.FEATURES]


def run_list():
    """Run ids with (kind, design name, buffer, placement or seed)."""
    R = {}
    for L in RK["first_placement_blocks_m"]:
        for b in RK["first_placement_buffers_m"]:
            R[f"sq{int(L)}_b{int(b)}_p0"] = ("fold", f"sq{int(L)}", float(b), 0)
    for L, b in RK["three_placement_designs"]:
        for r in PLACEMENTS[1:]:
            R[f"sq{int(L)}_b{int(b)}_p{r}"] = ("fold", f"sq{int(L)}", float(b), r)
    for seed in SR["five_fold_seeds"]:
        R[f"skf5_s{int(seed)}"] = ("fold", "random_skf5", 0.0, int(seed))
    for seed in SR["null_split_seeds"]:
        R[f"null_s{int(seed)}"] = ("null", None, np.nan, int(seed))
    for b in cfg.get("spatial_cv.component_buffers_m"):
        R[f"component_b{int(b)}"] = ("fold", "component", float(b), 0)
    for name in RK["alternative_unit_designs"]:
        R[f"{name}_b{int(RK['alternative_unit_buffer_m'])}"] = ("fold", name, float(RK["alternative_unit_buffer_m"]), 0)
    return R


RUNS = run_list()
KEEP_BUILDINGS = {f"sq{PRIM_L}_b{PRIM_B}_p{r}" for r in PLACEMENTS}                  # direction survival (step 12)
WITH_CONTROL = {f"sq{CTL_L}_b{CTL_B}_p{r}": r for r in PLACEMENTS}                 # ranking ladder (step 11)


def fold_record(site, run, f, y, idx, te, dropped, status, extra=None):
    nf = int(y[idx].sum()) if len(idx) else 0
    rec = dict(site=site, run=run, fold=int(f), status=status, n_train=int(len(idx)), n_fav_train=nf,
               n_formal_train=int(len(idx) - nf), n_minority_train=min(nf, int(len(idx) - nf)),
               n_test=int(len(te)), n_fav_test=int(y[te].sum()) if len(te) else 0, n_dropped=int(dropped))
    # first scoring criterion (SI S12 'Scoring'): fitted and >= 50 minority-class training buildings
    rec["rule1_ok"] = bool(status == "ok" and rec["n_minority_train"] >= s.min_minority_train)
    rec.update(extra or {})
    return rec


def add_sums(rec, S_te, main_te, fav_te):
    rec.update(dict(zip(ABS_OOF, np.abs(S_te[fav_te]).sum(0))))
    rec.update(dict(zip(ABS_MAIN, np.abs(main_te[fav_te]).sum(0))))
    return rec


def explain_fold(d, idx, te, n_jobs):
    """Fit the validation-pipeline model on idx and return Tree SHAP of the test rows (positive toward favela)."""
    model, _, Xte = fit_predict_cv(d.X[idx], d.y[idx], d.X[te], cfg, n_jobs)
    return tree_shap(model, Xte, negate=FORMAL_AS_1)


def run_folds(site, run, spec, d, main, n_jobs):
    """One cross-validation run: fold sums; per-building SHAP for the direction runs; control rows where requested."""
    _, name, b, pl = spec
    splits, meta = C.make_design(name, d, s, cfg, P, placement=pl if name.startswith("sq") else 0,
                                 seed=pl if name == "random_skf5" else None)
    extra = dict(design=meta["design"], placement=pl if name.startswith("sq") else 0, buffer=b)
    rows, crows = [], []
    S = np.full(d.X.shape, np.nan, dtype=np.float32) if run in KEEP_BUILDINGS else None
    fold_of = np.full(d.n, -1, dtype=np.int16)
    for f, tr, te in splits:
        idx, dropped, _ = scv.buffer_train(d.xy, tr, te, b)
        reason = scv.skip_reason(d.y, idx, te, s)
        if reason:
            rows.append(fold_record(site, run, f, d.y, idx, te, dropped, reason, extra))
        else:
            S_te = explain_fold(d, idx, te, n_jobs)
            rows.append(add_sums(fold_record(site, run, f, d.y, idx, te, dropped, "ok", extra), S_te, main[te],
                                 d.y[te] == 1))
            fold_of[te] = f
            if S is not None:
                S[te] = S_te
        if run in WITH_CONTROL:
            # size-matched control: same test fold, the buffer's number of training buildings removed at random
            cextra = dict(extra, design=meta["design"] + "_control")
            if reason:
                crows.append(fold_record(site, "control_" + run, f, d.y, idx, te, dropped, "skipped_block_fold_skipped",
                                         cextra))
                continue
            ridx = scv.control_train(tr, len(idx), f, WITH_CONTROL[run], s) if dropped > 0 else idx
            creason = scv.skip_reason(d.y, ridx, te, s)
            if creason:
                crows.append(fold_record(site, "control_" + run, f, d.y, ridx, te, dropped, creason, cextra))
                continue
            S_c = explain_fold(d, ridx, te, n_jobs) if dropped > 0 else S_te
            crows.append(add_sums(fold_record(site, "control_" + run, f, d.y, ridx, te, dropped, "ok", cextra), S_c,
                                  main[te], d.y[te] == 1))
    return pd.DataFrame(rows), (pd.DataFrame(crows) if crows else None), S, fold_of


def run_null(site, run, spec, d, main, n_jobs):
    """Split-refit null: refit on another random 80/20 split; SHAP of every building (training medians).

    The model and split follow ``spatial_cv.ranking.null_classifier`` (see the module docstring).
    """
    seed = spec[3]
    if NULL_CLASSIFIER == "model_main":
        ref = cfg.section("model_main")
    else:
        ref = cfg.get("spatial_cv.references.random_split")
    itr, _ = train_test_split(np.arange(d.n), test_size=float(ref["test_size"]),
                              stratify=d.y if ref["stratify"] else None, random_state=seed)
    Xtr, Xall = impute_training_median(d.X[itr], d.X)
    if NULL_CLASSIFIER == "model_main":
        # Methods 4.4 model, in its class coding; SHAP negated when formal settlement is class 1
        model = fit_classifier(cfg, "model_main", Xtr, d.y[itr], n_jobs)
        S_all = tree_shap(model, Xall, negate=formal_as_class_1(cfg, "model_main"))
    else:
        model = fit_classifier(cfg, "model_cv", Xtr, d.y[itr], n_jobs, random_state=int(SR["null_model_seed"]))
        S_all = tree_shap(model, Xall, negate=FORMAL_AS_1)
    every = np.arange(d.n)
    rec = fold_record(site, run, 0, d.y, itr, every, 0, "ok",
                      dict(design="random_80_20_refit", placement=0, buffer=np.nan, split_seed=seed,
                           null_classifier=NULL_CLASSIFIER))
    return pd.DataFrame([add_sums(rec, S_all, main, d.y == 1)])


def atomic(df: pd.DataFrame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(cfg.sites))
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--n_jobs", type=int, default=None, help="XGBoost threads (default: xgboost default)")
    ap.add_argument("--status", action="store_true")
    A = ap.parse_args()
    log = C.log_to(P.logs / f"10_oof_shap_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    todo = [(site, run) for site in A.sites for run in A.runs if not (P.shap_folds / f"{site}__{run}.parquet").exists()]
    log(f"runs {len(A.runs)} x sites {len(A.sites)}; remaining {len(todo)}")
    if A.status:
        for site, run in todo:
            log("  todo", site, run)
        raise SystemExit
    cache = {}
    t0 = time.time()
    for site, run in todo:
        t1 = time.time()
        try:
            if site not in cache:
                cache = {site: (C.load_site(P, site), C.load_main_shap(P, site))}
            d, main = cache[site]
            spec = RUNS[run]
            if spec[0] == "null":
                rows = run_null(site, run, spec, d, main, A.n_jobs)
            else:
                if spec[1] == "component" and not C.make_design("component", d, s, cfg, P)[0]:
                    rows = pd.DataFrame([dict(site=site, run=run, fold=-1, status="skipped_single_group", n_train=0,
                                              n_test=d.n, n_fav_test=d.n_favela, rule1_ok=False)])
                    atomic(rows, P.shap_folds / f"{site}__{run}.parquet")
                    log(f"{site:<24} {run:<20} single favela component group: not testable")
                    continue
                rows, crows, S, fold_of = run_folds(site, run, spec, d, main, A.n_jobs)
                if S is not None:
                    np.savez_compressed(P.shap_buildings / f"{site}__{run}.npz", S=S, fold_of=fold_of)
                if crows is not None:
                    atomic(crows, P.shap_folds / f"{site}__control_{run}.parquet")
            atomic(rows, P.shap_folds / f"{site}__{run}.parquet")          # written last = completion marker
            log(f"{site:<24} {run:<20} folds fitted {int((rows.status == 'ok').sum())}/{len(rows)}, "
                f"first criterion {int(rows.rule1_ok.sum())}; {time.time() - t1:.1f} s")
        except Exception:
            log(f"FAILED {site} {run}\n{traceback.format_exc()}")
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
