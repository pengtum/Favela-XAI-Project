"""Spatial cross-validation designs, fold fitting and the scoring procedure.

Manuscript items: Main Methods 4.5 'Validation under spatial separation'; SI
Section S12 (designs, scoring, Tables S10-S16, Figs S19-S21); ED Figs 3-4
(ranking stability); the blocked boundary-error rates of Results 2.1 / SI S5.

This module is the single implementation of the fold makers used by
appendix/S12_spatial_cv, the ranking-stability figures and the blocked
error-rate analysis. Every seed and setting is read from ``spatial_cv`` and
``model_cv`` in config.yaml.

Designs (SI S12 'Designs')
* square blocks of 100-1,000 m in three grid placements, assigned whole to five
  folds by a greedy rule that balances building count and favela share
  (``design_square``, ``assign_folds``);
* training buffers that remove every training building within b metres of any
  test building (``buffer_train``) and matched random-removal controls
  (``control_train``);
* hexagonal cells and k-means units of the area of a 250 m or 500 m square,
  five k-means clusters used as folds, ten folds (``design_hex``,
  ``design_kmeans_units``, ``design_kmeans_folds``, ``design_square(K=10)``);
* references: random 80/20 split and random stratified five-fold CV.
Scoring (Methods 4.5; SI S12 'Scoring')
* criterion 1: a fold is used only if its training set holds at least 50
  buildings of the minority class (``rule1_mask``);
* criterion 2: a site is scored in a design (per grid placement) only if the
  used folds predict at least half of its favela buildings (``apply_rules``);
* reported value: mean over scored sites of the per-site mean over used folds,
  with a site-bootstrap 95% interval (``summarise``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .model import fit_predict_cv
from .stats import bootstrap_mean_ci


@dataclass(frozen=True)
class CVSettings:
    """Settings of the spatial cross-validation (from config.yaml)."""
    n_folds: int = 5
    fold_seed: int = 42
    placement_seed_base: int = 1000
    control_seed_base: int = 42
    kmeans_min_clusters: int = 5
    kmeans_n_init: int = 10
    kmeans_seed: int = 42
    min_train_buildings: int = 50
    split_test_size: float = 0.2
    split_seed: int = 432
    skf_seed: int = 432
    min_minority_train: int = 50
    min_favela_coverage: float = 0.5
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 42

    @classmethod
    def from_config(cls, cfg) -> "CVSettings":
        s = cfg.section("spatial_cv")
        sc = s["scoring"]
        ref = s["references"]
        return cls(n_folds=int(s["n_folds"]), fold_seed=int(s["fold_seed"]),
                   placement_seed_base=int(s["placement_seed_base"]), control_seed_base=int(s["control_seed_base"]),
                   kmeans_min_clusters=int(s["kmeans_min_clusters"]), kmeans_n_init=int(s["kmeans_n_init"]),
                   kmeans_seed=int(s["kmeans_seed"]), min_train_buildings=int(s["min_train_buildings"]),
                   split_test_size=float(ref["random_split"]["test_size"]),
                   split_seed=int(ref["random_split"]["random_state"]),
                   skf_seed=int(ref["five_fold"]["random_state"]),
                   min_minority_train=int(sc["min_minority_train"]),
                   min_favela_coverage=float(sc["min_favela_coverage"]),
                   bootstrap_resamples=int(sc["bootstrap_resamples"]), bootstrap_seed=int(sc["bootstrap_seed"]))


# -----------------------------------------------------------------------------
# Spatial units
# -----------------------------------------------------------------------------
def square_cells(x, y, size_m: float, origin=None) -> np.ndarray:
    """Square grid cell id of each building centroid; origin defaults to (x.min, y.min)."""
    ox, oy = (x.min(), y.min()) if origin is None else origin
    return (np.floor((x - ox) / size_m).astype(int)) * 100000 + np.floor((y - oy) / size_m).astype(int)


def grid_origin(x, y, size_m: float, placement: int, s: CVSettings):
    """Grid origin of a placement: (x.min, y.min) for placement 0; otherwise the grid is
    shifted by an offset drawn from U[0, L)^2 with default_rng(placement_seed_base + placement).
    Returns (origin, offset)."""
    if placement == 0:
        return (x.min(), y.min()), (0.0, 0.0)
    off = np.random.default_rng(s.placement_seed_base + placement).uniform(0, size_m, size=2)
    return (x.min() - off[0], y.min() - off[1]), (float(off[0]), float(off[1]))


_HEX_K = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))   # side a = HEX_K * L gives hexagon area L^2


def hex_cells(x, y, size_m: float, origin=None):
    """Pointy-top hexagonal cells with the area of a size_m square. Returns (cell_id, q, r)."""
    ox, oy = (x.min(), y.min()) if origin is None else origin
    a = _HEX_K * size_m
    px, py = x - ox, y - oy
    qf = (np.sqrt(3.0) / 3.0 * px - py / 3.0) / a
    rf = (2.0 / 3.0 * py) / a
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
    dx, dy, dz = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)
    c1 = (dx > dy) & (dx > dz)
    c2 = ~c1 & (dy > dz)
    c3 = ~c1 & ~c2
    rx = np.where(c1, -ry - rz, rx)
    ry = np.where(c2, -rx - rz, ry)
    rz = np.where(c3, -rx - ry, rz)
    q, r = rx.astype(np.int64), rz.astype(np.int64)
    return (q + 50000) * 100000 + (r + 50000), q, r


def hex_polygon(q, r, size_m: float, origin) -> np.ndarray:
    """Vertices (6 x 2) of the hexagon with axial coordinates (q, r) (for drawing)."""
    a = _HEX_K * size_m
    ox, oy = origin
    cx = ox + a * (np.sqrt(3.0) * q + np.sqrt(3.0) / 2.0 * r)
    cy = oy + a * 1.5 * r
    ang = np.deg2rad(30 + 60 * np.arange(6))
    return np.c_[cx + a * np.cos(ang), cy + a * np.sin(ang)]


def kmeans_units(xy: np.ndarray, size_m: float, s: CVSettings):
    """k-means clusters of centroids; k = max(min_clusters, round(convex-hull area / L^2))."""
    k = max(s.kmeans_min_clusters, int(round(float(ConvexHull(xy).volume) / size_m ** 2)))
    labels = KMeans(n_clusters=k, random_state=s.kmeans_seed, n_init=s.kmeans_n_init).fit(xy).labels_
    return labels, k


def kmeans_folds(xy: np.ndarray, n_clusters: int, s: CVSettings) -> np.ndarray:
    """Five contiguous k-means clusters used directly as the folds."""
    return KMeans(n_clusters=n_clusters, random_state=s.kmeans_seed, n_init=s.kmeans_n_init).fit(xy).labels_


# -----------------------------------------------------------------------------
# Fold assignment
# -----------------------------------------------------------------------------
def assign_folds(units, y, n_folds: int, seed: int) -> np.ndarray:
    """Greedy assignment of whole units to folds balancing building count and favela share.

    Units are shuffled (seed), sorted by size (largest first) and each is given to
    the fold with the lowest cost n + 0.5 * |favela share after adding - site share| * N.
    """
    units = np.asarray(units)
    y = np.asarray(y)
    bl = pd.DataFrame({"b": units, "y": y}).groupby("b").agg(n=("y", "size"), pos=("y", "sum"))
    bl = bl.sample(frac=1, random_state=seed).sort_values("n", ascending=False)
    fn = np.zeros(n_folds)
    fp = np.zeros(n_folds)
    mapping = {}
    for unit, row in bl.iterrows():
        cost = fn + 0.5 * np.abs((fp + row.pos) / np.maximum(fn + row.n, 1) - y.mean()) * y.size
        j = int(np.argmin(cost))
        mapping[unit] = j
        fn[j] += row.n
        fp[j] += row.pos
    return np.array([mapping[v] for v in units])


def splits_from_labels(fold_of, n_folds: int):
    """[(fold, train_idx, test_idx)] with indices in increasing order."""
    fold_of = np.asarray(fold_of)
    return [(f, np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(n_folds)]


def splits_random_8020(n: int, s: CVSettings):
    """Reference: random 80/20 split of the validation pipeline (unstratified, seed 432)."""
    tr, te = train_test_split(np.arange(n), test_size=s.split_test_size, random_state=s.split_seed)
    return [(0, tr, te)]


def splits_five_fold(X, y, s: CVSettings, n_folds: int = 5, seed: int | None = None):
    """Reference: random stratified K-fold CV (shuffled, seed 432 unless given)."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=s.skf_seed if seed is None else seed)
    return [(f, tr, te) for f, (tr, te) in enumerate(skf.split(X, y))]


# -----------------------------------------------------------------------------
# Buffer and matched control
# -----------------------------------------------------------------------------
def buffer_train(xy, train_idx, test_idx, buffer_m, d_train=None):
    """Keep the training buildings farther than buffer_m from every test building.

    Returns (kept training indices, number dropped, distances of the training
    buildings to the nearest test building)."""
    if buffer_m is None or buffer_m == 0 or len(test_idx) == 0:
        return train_idx, 0, d_train
    if d_train is None:
        d_train, _ = cKDTree(xy[test_idx]).query(xy[train_idx], k=1)
    keep = d_train > buffer_m
    return train_idx[keep], int((~keep).sum()), d_train


def control_train(train_idx, n_keep: int, fold: int, placement: int, s: CVSettings):
    """Matched control: the same number of training buildings drawn at random from the fold's training set."""
    rng = np.random.RandomState(s.control_seed_base + fold + 100 * placement)
    return rng.choice(train_idx, size=n_keep, replace=False)


# -----------------------------------------------------------------------------
# Fitting one fold
# -----------------------------------------------------------------------------
METRIC_KEYS = ["accuracy", "balanced_accuracy", "f1_favela", "f1_formal", "auc", "brier"]


def fold_metrics(y_test, prob) -> dict:
    yp = (prob >= 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = len(np.unique(y_test)) > 1
        return dict(accuracy=accuracy_score(y_test, yp), balanced_accuracy=balanced_accuracy_score(y_test, yp),
                    f1_favela=f1_score(y_test, yp, zero_division=0),
                    f1_formal=f1_score(y_test, yp, pos_label=0, zero_division=0),
                    auc=roc_auc_score(y_test, prob) if two else np.nan,
                    brier=float(np.mean((prob - y_test) ** 2)))


def nn_distance(xy, train_idx, test_idx) -> np.ndarray:
    """Distance of each test building to the nearest training building."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.full(len(test_idx), np.nan)
    d, _ = cKDTree(xy[train_idx]).query(xy[test_idx], k=1)
    return d


def skip_reason(y, train_idx, test_idx, s: CVSettings):
    """Why a fold is not fitted (None if it is fitted)."""
    if len(test_idx) == 0:
        return "skipped_empty_test"
    if len(train_idx) < s.min_train_buildings:
        return "skipped_train_lt50"
    if len(np.unique(y[train_idx])) < 2:
        return "skipped_single_class_train"
    return None


def fit_eval(X, y, xy, train_idx, test_idx, meta: dict, n_dropped: int, cfg, s: CVSettings,
             n_jobs=None, want_pred: bool = False):
    """Fit one fold with the validation-pipeline model; return (metrics row, predictions or None)."""
    row = dict(meta, n_train=len(train_idx), n_test=len(test_idx), n_dropped=n_dropped,
               prev_train=float(y[train_idx].mean()) if len(train_idx) else np.nan,
               prev_test=float(y[test_idx].mean()) if len(test_idx) else np.nan)
    reason = skip_reason(y, train_idx, test_idx, s)
    if reason:
        row.update(status=reason, **{k: np.nan for k in METRIC_KEYS},
                   d_test_train_median=np.nan, d_test_train_p90=np.nan)
        return row, None
    _, prob, _ = fit_predict_cv(X[train_idx], y[train_idx], X[test_idx], cfg, n_jobs)
    d = nn_distance(xy, train_idx, test_idx)
    row.update(status="ok", **fold_metrics(y[test_idx], prob),
               d_test_train_median=float(np.median(d)), d_test_train_p90=float(np.percentile(d, 90)))
    pred = None
    if want_pred:
        pred = pd.DataFrame(dict(row_id=test_idx.astype(np.int32), fold=np.int16(meta["fold"]),
                                 y_true=y[test_idx].astype(np.int8), prob=prob.astype(np.float32),
                                 d_nn_train=d.astype(np.float32),
                                 misclassified=((prob >= 0.5).astype(int) != y[test_idx])))
    return row, pred


def run_splits(X, y, xy, splits, buffers, meta: dict, cfg, s: CVSettings, placement: int = 0,
               controls: bool = True, n_jobs=None, pred_buffers=(), log=None):
    """All buffers x folds of one design; returns (metric rows, prediction frames).

    Control rule: for b > 0, a matched random-removal model is fitted when the
    blocked fold was fitted and dropped buildings; a skipped blocked fold gives a
    control row 'skipped_block_fold_skipped'; when nothing was dropped the control
    equals the blocked fit ('ok_no_drop_same_as_blocked')."""
    rows, preds, dcache = [], [], {}
    for b in buffers:
        for f, tr, te in splits:
            if f not in dcache:
                dcache[f] = (cKDTree(xy[te]).query(xy[tr], k=1)[0] if (len(te) and max(buffers) > 0) else None)
            idx, dropped, _ = buffer_train(xy, tr, te, b, dcache[f])
            m = dict(meta, repeat=placement, fold=f, buffer=float(b), control=False)
            row, pred = fit_eval(X, y, xy, idx, te, m, dropped, cfg, s, n_jobs, want_pred=(b in pred_buffers))
            rows.append(row)
            if pred is not None:
                pred["buffer"] = np.float32(b)
                preds.append(pred)
            if controls and b > 0:
                mc = dict(m, control=True)
                if row["status"] != "ok":
                    rc = dict(row, **mc)
                    rc["status"] = "skipped_block_fold_skipped"
                    for k in METRIC_KEYS + ["d_test_train_median", "d_test_train_p90"]:
                        rc[k] = np.nan
                    rows.append(rc)
                elif dropped == 0:
                    rc = dict(row, **mc)
                    rc["status"] = "ok_no_drop_same_as_blocked"
                    rows.append(rc)
                else:
                    ridx = control_train(tr, len(idx), f, placement, s)
                    rc, _ = fit_eval(X, y, xy, ridx, te, mc, dropped, cfg, s, n_jobs)
                    rows.append(rc)
            if log:
                log(f"  {meta.get('design')} p{placement} b{b} f{f} n_tr={len(idx)} n_te={len(te)} "
                    f"{row['status']} acc={row['accuracy']:.4f}")
    return rows, preds


# -----------------------------------------------------------------------------
# Designs
# -----------------------------------------------------------------------------
def _meta(site, design, rule, size_m, n_folds, units, extra=None):
    m = dict(site=site, design=design, block_rule=rule, L=float(size_m) if size_m is not None else np.nan,
             K=n_folds, n_units=int(len(np.unique(units))) if units is not None else np.nan)
    if extra:
        m.update(extra)
    return m


def design_square(site, y, xy, size_m: float, s: CVSettings, n_folds: int = 5, placement: int = 0):
    """Square blocks of size_m, whole blocks to folds (fold seed = fold_seed + placement)."""
    origin, off = grid_origin(xy[:, 0], xy[:, 1], size_m, placement, s)
    units = square_cells(xy[:, 0], xy[:, 1], size_m, origin)
    fold_of = assign_folds(units, y, n_folds, seed=s.fold_seed + placement)
    name = f"sq{int(size_m)}" + ("" if n_folds == 5 else f"_K{n_folds}")
    return splits_from_labels(fold_of, n_folds), _meta(site, name, "square", size_m, n_folds, units,
                                                       dict(offset_x=off[0], offset_y=off[1]))


def design_hex(site, y, xy, size_m: float, s: CVSettings, n_folds: int = 5):
    units, _, _ = hex_cells(xy[:, 0], xy[:, 1], size_m)
    fold_of = assign_folds(units, y, n_folds, seed=s.fold_seed)
    return splits_from_labels(fold_of, n_folds), _meta(site, f"hex{int(size_m)}", "hexagon", size_m, n_folds, units)


def design_kmeans_units(site, y, xy, size_m: float, s: CVSettings, n_folds: int = 5):
    units, k = kmeans_units(xy, size_m, s)
    fold_of = assign_folds(units, y, n_folds, seed=s.fold_seed)
    return splits_from_labels(fold_of, n_folds), _meta(site, f"kmm{int(size_m)}", "kmeans_many", size_m, n_folds,
                                                       units, dict(n_clusters=k))


def design_kmeans_folds(site, y, xy, s: CVSettings, n_folds: int = 5):
    clusters = kmeans_folds(xy, n_folds, s)
    return splits_from_labels(clusters, n_folds), _meta(site, "km5", "kmeans5", None, n_folds, clusters,
                                                        dict(n_clusters=n_folds))


# -----------------------------------------------------------------------------
# Scoring (Methods 4.5; SI S12 'Scoring')
# -----------------------------------------------------------------------------
SCORED_STATUS = ["ok", "ok_no_drop_same_as_blocked"]


def total_favela(fold_rows: pd.DataFrame) -> pd.Series:
    """Favela buildings per site, from the random stratified five-fold rows (every building tested once)."""
    r = fold_rows[fold_rows.design == "random_skf5"]
    return (r.prev_test * r.n_test).groupby(r.site).sum().round()


def rule1_mask(df: pd.DataFrame, s: CVSettings) -> pd.Series:
    """Criterion 1: fitted fold with at least min_minority_train buildings of the minority class in training."""
    fitted = df.status.isin(SCORED_STATUS) if "status" in df else pd.Series(True, index=df.index)
    fav = (df.n_train * df.prev_train).where(df.n_train > 0, 0.0).round()
    frm = df.n_train - fav
    return fitted & (np.minimum(fav, frm) >= s.min_minority_train)


def apply_rules(df: pd.DataFrame, tot_fav: pd.Series, s: CVSettings, unit=("site", "repeat")):
    """Both criteria for the fold rows of ONE design cell. Returns (kept folds, coverage table)."""
    unit = list(unit)
    d = df.copy()
    if "repeat" in unit and "repeat" not in d:
        d["repeat"] = 0
    d["rule1_ok"] = rule1_mask(d, s)
    d["fav_test"] = d.prev_test * d.n_test
    units = d[unit].drop_duplicates()
    cov = d[d.rule1_ok].groupby(unit).fav_test.sum().rename("fav_covered").reset_index()
    cov = units.merge(cov, on=unit, how="left").fillna({"fav_covered": 0.0})
    cov["fav_total"] = cov.site.map(tot_fav)
    cov["coverage"] = cov.fav_covered / cov.fav_total
    cov["scored"] = cov.coverage >= s.min_favela_coverage
    kept = d[d.rule1_ok].merge(cov.loc[cov.scored, unit], on=unit)
    return kept, cov


def apply_control(ctl: pd.DataFrame, cov: pd.DataFrame, s: CVSettings, unit=("site", "repeat")):
    """Matched controls are scored on the site x placement units kept for the blocked design."""
    unit = list(unit)
    c = ctl.copy()
    if "repeat" in unit and "repeat" not in c:
        c["repeat"] = 0
    c = c[rule1_mask(c, s)]
    return c.merge(cov.loc[cov.scored, unit], on=unit)


def summarise(kept: pd.DataFrame, s: CVSettings, metrics=("accuracy", "balanced_accuracy", "auc", "f1_favela")):
    """Mean over scored sites of per-site means over used folds, with site-bootstrap 95% intervals."""
    metrics = list(metrics)
    site_means = kept.groupby("site")[metrics].mean()
    row = {"n_sites": int(site_means[metrics[1]].notna().sum()), "n_folds": int(len(kept)),
           "sites": ";".join(site_means.index[site_means[metrics[1]].notna()])}
    for m in metrics:
        row[m] = float(site_means[m].mean()) if site_means[m].notna().any() else np.nan
        row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = bootstrap_mean_ci(site_means[m].values, s.bootstrap_resamples,
                                                                 s.bootstrap_seed)
    if "prev_test" in kept:
        row["prev_test_mean"] = float(kept.groupby("site").prev_test.mean().mean())
    return row, site_means
