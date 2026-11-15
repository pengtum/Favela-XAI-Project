"""Step 3: fit every fold of the spatial cross-validation designs (accuracy side).

Manuscript items: the fold record behind SI Section S12 'Classification accuracy
under spatial separation' and 'Distance between test and training buildings'
(SI Tables S10-S15, SI Figs S19-S21), the pooled leave-site-out numbers quoted
in Results 2.1, and Supplementary Data 1-2 (per-fold record, Table S14 caption).

Every fit uses the validation-pipeline classifier (SI S12 'Designs'; Main Methods
'Validation under spatial separation'): XGBoost with 100 trees and seed 432, other
settings at their defaults (the settings of the main per-site models), intercept
= share of class 1 among the fold's training rows (``model_cv.base_score``),
missing indicator values imputed with the median of the fold's training rows
(``model_cv``; favela_xai.model). Units, folds and random draws are assigned by
row position, in the row order of the per-building tables (main/01_features).

Jobs (one shard per site and job; existing shards are skipped, so the script can
be re-run until everything is done):
  random     random 80/20 split (unstratified, seed 432) and random stratified
             five-fold cross-validation (seed 432); per-building predictions kept
  loso       pooled leave-site-out: one classifier trained on the other sites;
             per-building predictions kept
  main       square blocks of 100, 250, 500 and 1,000 m, five folds, three grid
             placements, buffers 0-400 m, matched random-removal controls
             (``spatial_cv``); predictions of the first placement kept at the
             buffers of ``appendix_S12_spatial_cv.prediction_buffers_m``
  sens       hexagons (250, 500 m), k-means units (250, 500 m), five k-means
             clusters as folds, ten folds (250, 500 m); first placement, buffers
             0 and 200 m, matched controls at 200 m; predictions of the five
             k-means folds kept at 0 m (SI Fig. S20a)
  component  held-out favela component, buffers 0 and 200 m (predictions kept),
             and the stratified random 80/20 split drawn in SI Fig. S21
Fold rules (favela_xai.spatial_cv): whole units are assigned greedily to folds,
balancing building count and favela share; the
buffer removes every training building within b m of any test building; the
matched control removes the same number of training buildings at random from the
same fold; a fold is not fitted without a test building, with fewer than 50
training buildings or with a single class in training.

Inputs  outputs/appendix/S12_spatial_cv/site_tables/ (step 01), component_groups/ (step 02)
Outputs outputs/appendix/S12_spatial_cv/fold_shards/<job>__<site>[__<design>].parquet
        outputs/appendix/S12_spatial_cv/prediction_shards/<same name>.parquet
        (row_id = row of the site table, fold, y_true, prob = favela probability,
        d_nn_train = distance to the nearest training building used in the fit,
        misclassified, buffer)
Run     python appendix/S12_spatial_cv/03_run_folds.py [--part all|random|loso|main|sens|component]
                                                        [--sites ...] [--n_jobs N] [--status]
"""
import argparse
import datetime
import time
import traceback

import numpy as np
import pandas as pd

from favela_xai import spatial_cv as scv

import s12_common as C

cfg, s, P = C.setup(__file__)
S12 = C.section(cfg)
SCV = cfg.section("spatial_cv")
BLOCKS = [int(v) for v in SCV["block_sizes_m"]]                     # SI S12: 100, 250, 500, 1,000 m
BUFFERS = [int(v) for v in SCV["buffers_m"]]                        # SI S12: 0-400 m
PLACEMENTS = list(range(int(SCV["grid_placements"])))               # SI S12: three grid placements
SENS_BUFFERS = [int(v) for v in SCV["sensitivity_buffers_m"]]       # 0 and 200 m
COMP_BUFFERS = [int(v) for v in SCV["component_buffers_m"]]         # 0 and 200 m
PRED_BUFFERS = tuple(int(v) for v in S12["prediction_buffers_m"])   # predictions kept, first placement
SENS = ([f"hex{int(L)}" for L in SCV["hexagon_sizes_m"]] + [f"kmm{int(L)}" for L in SCV["kmeans_unit_sizes_m"]]
        + ["km5"] + [f"sq{int(L)}_K{int(SCV['ten_fold']['n_folds'])}" for L in SCV["ten_fold"]["block_sizes_m"]])


def meta_clean(meta: dict) -> dict:
    """Drop the private keys of make_design (group codes) before they enter the fold rows."""
    return {k: v for k, v in meta.items() if not k.startswith("_")}


def with_site_cols(pred: list, site: str, design: str, L, placement: int = 0):
    """Stack prediction frames and add the design columns."""
    if not pred:
        return None
    df = pd.concat(pred, ignore_index=True)
    df.insert(0, "site", site)
    df.insert(1, "design", design)
    df.insert(2, "L", np.float32(L))
    df.insert(3, "repeat", np.int8(placement))
    return df


# ----------------------------------------------------------------------------- jobs
def job_random(site, n_jobs):
    d = C.load_site(P, site)
    rows, preds = [], []
    for name in ("random_80_20", "random_skf5"):
        splits, meta = C.make_design(name, d, s, cfg, P)
        r, pr = scv.run_splits(d.X, d.y, d.xy, splits, [0], meta, cfg, s, placement=0, controls=False,
                               n_jobs=n_jobs, pred_buffers=(0,))
        rows += r
        pw = with_site_cols(pr, site, name, np.nan)
        if pw is not None:
            preds.append(pw)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


_POOLED = {}


def job_loso(site, n_jobs):
    if "arrays" not in _POOLED:                   # the pooled table of all sites is built once per process
        _POOLED["arrays"] = C.pooled_arrays(P, cfg.sites)
    row, pred = C.loso_fold(site, _POOLED["arrays"], list(cfg.sites), cfg, s, n_jobs)
    return pd.DataFrame([row]), with_site_cols([pred] if pred is not None else [], site, "loso_pooled", np.nan)


def job_main(site, L, placement, n_jobs):
    d = C.load_site(P, site)
    splits, meta = C.make_design(f"sq{L}", d, s, cfg, P, placement=placement)
    keep = PRED_BUFFERS if placement == 0 else ()
    rows, pr = scv.run_splits(d.X, d.y, d.xy, splits, BUFFERS, meta, cfg, s, placement=placement, controls=True,
                              n_jobs=n_jobs, pred_buffers=keep)
    return pd.DataFrame(rows), with_site_cols(pr, site, meta["design"], L, placement)


def job_sens(site, name, n_jobs):
    d = C.load_site(P, site)
    splits, meta = C.make_design(name, d, s, cfg, P, placement=0)
    keep = (0,) if name == "km5" else ()
    rows, pr = scv.run_splits(d.X, d.y, d.xy, splits, SENS_BUFFERS, meta, cfg, s, placement=0, controls=True,
                              n_jobs=n_jobs, pred_buffers=keep)
    return pd.DataFrame(rows), with_site_cols(pr, site, meta["design"], meta["L"])


def job_component(site, n_jobs):
    d = C.load_site(P, site)
    rows, preds = [], []
    # stratified random 80/20 split of SI Fig. S21 (diamonds)
    splits, meta = C.make_design("random_80_20_stratified", d, s, cfg, P)
    r, _ = scv.run_splits(d.X, d.y, d.xy, splits, [0], meta, cfg, s, controls=False, n_jobs=n_jobs)
    rows += r
    # held-out favela component (sites with a single group cannot be tested and give no fold)
    splits, meta = C.make_design("component", d, s, cfg, P)
    if splits:
        codes, members = meta["_groups"], meta["_members"]
        r, pr = scv.run_splits(d.X, d.y, d.xy, splits, COMP_BUFFERS, meta_clean(meta), cfg, s, controls=False,
                               n_jobs=n_jobs, pred_buffers=tuple(COMP_BUFFERS))
        for row in r:
            row["test_group"] = int(codes[row["fold"]])
            row["test_components"] = members[codes[row["fold"]]]
        rows += r
        pw = with_site_cols(pr, site, "component", np.nan)
        if pw is not None:
            preds.append(pw)
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else None)


def tasks(parts, sites):
    """(shard name, callable) for every requested job."""
    T = []
    for site in sites:
        if "random" in parts:
            T.append((f"random__{site}", lambda site=site: job_random(site, A.n_jobs)))
        if "loso" in parts and len(cfg.sites) > 1:
            T.append((f"loso__{site}", lambda site=site: job_loso(site, A.n_jobs)))
        if "main" in parts:
            for L in BLOCKS:
                for r in PLACEMENTS:
                    T.append((f"main__{site}__sq{L}__p{r}", lambda site=site, L=L, r=r: job_main(site, L, r, A.n_jobs)))
        if "sens" in parts:
            for name in SENS:
                T.append((f"sens__{site}__{name}", lambda site=site, name=name: job_sens(site, name, A.n_jobs)))
        if "component" in parts:
            T.append((f"component__{site}", lambda site=site: job_component(site, A.n_jobs)))
    return T


def atomic_parquet(df: pd.DataFrame, path):
    """Write to a temporary file first so that an interrupted run never leaves a partial shard."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["all"],
                    choices=["all", "random", "loso", "main", "sens", "component"])
    ap.add_argument("--sites", nargs="+", default=list(cfg.sites))
    ap.add_argument("--n_jobs", type=int, default=None, help="XGBoost threads (default: xgboost default)")
    ap.add_argument("--status", action="store_true", help="list done / remaining shards without fitting")
    A = ap.parse_args()
    parts = {"random", "loso", "main", "sens", "component"} if "all" in A.part else set(A.part)
    for site in A.sites:
        assert site in cfg.sites, f"unknown site {site}"
    log = C.log_to(P.logs / f"03_run_folds_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    T = tasks(parts, A.sites)
    todo = [(n, fn) for n, fn in T if not (P.fold_shards / f"{n}.parquet").exists()]
    log(f"parts {sorted(parts)}; sites {len(A.sites)}; shards {len(T)}; remaining {len(todo)}")
    if A.status:
        for n, _ in todo:
            log("  todo", n)
        raise SystemExit
    t0 = time.time()
    for n, fn in todo:
        t1 = time.time()
        try:
            folds, pred = fn()
        except Exception:
            log(f"FAILED {n}\n{traceback.format_exc()}")
            continue
        if pred is not None and len(pred):
            atomic_parquet(pred, P.pred_shards / f"{n}.parquet")
        atomic_parquet(folds, P.fold_shards / f"{n}.parquet")          # written last = completion marker
        log(f"wrote {n}: {len(folds)} fold rows {folds.status.value_counts().to_dict()}; "
            f"{0 if pred is None else len(pred)} predictions; {time.time() - t1:.1f} s")
    left = sum(not (P.fold_shards / f"{n}.parquet").exists() for n, _ in T)
    log(f"finished in {(time.time() - t0) / 60:.1f} min; remaining {left}")
