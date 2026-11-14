"""Shared helpers of appendix/S14_label_free (Supplementary Section S14).

This module is imported by the numbered scripts of this folder; it produces no
manuscript item by itself. It holds:

* ``site_table``: the per-building table of one site - favela label, the 25
  indicators, the Tree-SHAP attributions of the main model, the signed distance
  to the 2013 boundary and the out-of-fold favela probability of the
  validation pipeline (the supervised reference of SI S14);
* the label-free representation: median imputation and rank transform to a
  standard normal distribution (SI S14 'Design'; Main Methods 'Label-free
  clustering of the morphological indicators') and the three weightings;
* clustering (k-means; Ward fitted on a random subsample, the remaining
  buildings and the subsample itself then assigned to the nearest Ward
  centroid) and the scores (adjusted Rand index, best
  balanced accuracy over assignments of clusters to classes, two-cluster score);
* the 6 m boundary profile, its normalisation and level crossings.

Settings: ``label_free`` (Main Methods; SI S14), ``appendix_S14_label_free``
(this folder), ``oof`` / ``model_cv`` (supervised reference),
``signed_distance`` (distance).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyogrio
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from favela_xai.boundary import load_signed_distance, signed_distance
from favela_xai.config import ConfigError, load_config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, dimension_of, features_by_dimension
from favela_xai.io import read_favela_polygons, read_site_shap
from favela_xai.labels import label_to_int

cfg = load_config()
OUT = cfg.output_dir(__file__)                 # outputs/appendix/S14_label_free/
LF = cfg.section("label_free")                 # Methods 'Label-free clustering'; SI S14
FS = cfg.section("appendix_S14_label_free")    # settings specific to this folder

FEATURES = list(MODEL_FEATURE_ORDER)           # column order of every matrix
SEED = int(LF["seed"])
WEIGHTINGS = list(LF["weightings"])            # all_equal, dimensions_balanced, one_component_per_dimension
# Names used in SI S14 / Fig. S23 / Table S18
WEIGHTING_NAMES = dict(FS["weighting_names"])


# -----------------------------------------------------------------------------
# Per-site building table
# -----------------------------------------------------------------------------
_CACHE: dict = {}


def _oof_table() -> pd.DataFrame:
    """Out-of-fold predictions of the validation pipeline (products.oof_predictions)."""
    if "oof" not in _CACHE:
        path = cfg.require_product("oof_predictions")
        _CACHE["oof"] = pd.read_parquet(path)
    return _CACHE["oof"]


def _oof_probability(site: str, uid, y: np.ndarray) -> np.ndarray:
    """Out-of-fold favela probability of the buildings of one site, in table order.

    Supervised reference of SI S14: the out-of-fold predictions of five-fold
    cross-validation (XGBoost, 100 trees, seed 432; Main Methods 4.7; ``oof``,
    ``model_cv``), written by main/02_classification. Rows are matched on uID
    when both tables carry it, otherwise by position within the site (checked
    against the label).
    """
    oof = _oof_table()
    prob_col = next((c for c in FS["oof_probability_columns"] if c in oof.columns), None)
    if prob_col is None or "site" not in oof.columns:
        raise ConfigError(f"products.oof_predictions needs a 'site' column and one of "
                          f"{FS['oof_probability_columns']} (appendix_S14_label_free.oof_probability_columns)")
    part = oof[oof["site"] == site]
    if uid is not None and "uID" in part.columns:
        prob = part.set_index("uID")[prob_col].reindex(np.asarray(uid)).to_numpy(float)
    else:
        if len(part) != len(y):
            raise ConfigError(f"{site}: {len(part)} out-of-fold predictions for {len(y)} buildings")
        label_col = next((c for c in ("y_true", "is_favela", "label") if c in part.columns), None)
        if label_col is not None:
            lab = part[label_col]
            y_oof = lab.to_numpy().astype(int) if pd.api.types.is_numeric_dtype(lab) else label_to_int(lab)
            if not np.array_equal(y_oof, y):
                raise ConfigError(f"{site}: out-of-fold predictions are not in the order of the building table")
        prob = part[prob_col].to_numpy(float)
    if np.isnan(prob).any():
        raise ConfigError(f"{site}: {int(np.isnan(prob).sum())} buildings have no out-of-fold probability")
    return prob


def site_table(site: str) -> pd.DataFrame:
    """Per-building table of one site.

    Columns: [uID,] y (1 = favela), signed_dist (m; negative on the
    favela-labelled side, Methods 4.8), prob (out-of-fold favela probability),
    the 25 indicators and the 25 shap_ columns (main model, positive = toward
    favela). Row order = the per-site table of main/02_classification.
    """
    path = cfg.require_product("shap_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    need = ["label"] + FEATURES + list(SHAP_COLUMNS)
    missing = [c for c in need if c not in fields]
    if missing:
        raise ConfigError(f"{path.name} lacks the columns {missing}; re-run main/02_classification.")
    if "uID" in fields:
        raw = pyogrio.read_dataframe(path, columns=["uID"] + need, read_geometry=False)
        y = label_to_int(raw["label"])
        if "sd" not in _CACHE:
            _CACHE["sd"] = load_signed_distance(cfg)
        sd = _CACHE["sd"]
        sd = sd[sd["site"] == site].set_index("uID")["signed_dist"].reindex(raw["uID"].to_numpy()).to_numpy(float)
        if np.isnan(sd).any():
            raise ConfigError(f"{site}: {int(np.isnan(sd).sum())} buildings have no signed distance; "
                              "re-run main/01_features (products.signed_distance)")
        uid = raw["uID"].to_numpy()
    else:                                   # no identifier: distance from the table's own footprints
        raw = read_site_shap(cfg, site, columns=need)
        y = label_to_int(raw["label"])
        if "polygons" not in _CACHE:
            _CACHE["polygons"] = read_favela_polygons(cfg, 2013)
        sd = signed_distance(raw, y.astype(bool), _CACHE["polygons"], cfg)
        uid = None
    out = pd.DataFrame({"y": y, "signed_dist": sd, "prob": _oof_probability(site, uid, y)})
    if uid is not None:
        out.insert(0, "uID", uid)
    for c in FEATURES + list(SHAP_COLUMNS):
        out[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
    return out


# -----------------------------------------------------------------------------
# Label-free representation
# -----------------------------------------------------------------------------
def rank_normal(X: np.ndarray) -> np.ndarray:
    """Median imputation and rank transform to a standard normal distribution
    (SI S14 'Design')."""
    if LF["transform"] != "rank_to_normal" or LF["imputation"] != "median":
        raise ConfigError("only label_free.transform: rank_to_normal with imputation: median is implemented")
    X = np.asarray(X, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    if np.isnan(X).any():
        raise ConfigError("an indicator is missing for every building of a site")
    qt = QuantileTransformer(n_quantiles=min(int(LF["n_quantiles_max"]), len(X)), output_distribution="normal",
                             random_state=SEED)
    return qt.fit_transform(X)


def weighted_space(Z: np.ndarray, weighting: str) -> np.ndarray:
    """One of the three weightings of SI S14 'Design'.

    * all_equal: the 25 transformed indicators unweighted;
    * dimensions_balanced: each indicator scaled by 1/sqrt(size of its
      morphometric dimension) (3/13/4/3/2, Methods 4.3);
    * one_component_per_dimension: the standardised first principal component
      of each dimension (five inputs, dimensions in SI Table S2 order).
    """
    if weighting == "all_equal":
        return Z
    groups = features_by_dimension()
    if weighting == "dimensions_balanced":
        w = np.array([1 / np.sqrt(len(groups[dimension_of(f)])) for f in FEATURES])
        return Z * w
    if weighting == "one_component_per_dimension":
        cols = []
        for members in groups.values():
            idx = [FEATURES.index(f) for f in members]
            pc1 = PCA(1, random_state=SEED).fit_transform(Z[:, idx])
            cols.append(StandardScaler().fit_transform(pc1))
        return np.column_stack(cols)
    raise ConfigError(f"unknown weighting '{weighting}' in label_free.weightings")


# -----------------------------------------------------------------------------
# Clustering and scores
# -----------------------------------------------------------------------------
def ward_subsample(n: int) -> np.ndarray:
    """Random subsample (at most ward_max_sample buildings) on which Ward is fitted;
    one draw per site, shared by every k and weighting."""
    return np.random.RandomState(SEED).choice(n, min(int(LF["ward_max_sample"]), n), replace=False)


def cluster(S: np.ndarray, k: int, method: str, sub: np.ndarray) -> np.ndarray:
    """Cluster labels of every building.

    k-means: KMeans(k, n_init, random_state) on all buildings. Ward: fitted on
    the subsample, every building then assigned to the nearest Ward centroid
    (SI S14 'Design').
    """
    if method == "kmeans":
        return KMeans(k, n_init=int(LF["kmeans_n_init"]), random_state=SEED).fit_predict(S)
    if method == "ward":
        fit = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(S[sub])
        centres = np.vstack([S[sub][fit.labels_ == c].mean(axis=0) for c in range(k)])
        return np.argmin(((S[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    raise ConfigError(f"unknown clustering method '{method}' in label_free.algorithms")


def best_balanced_accuracy(labels: np.ndarray, y: np.ndarray) -> float:
    """Best balanced accuracy over assignments of clusters to classes.

    Clusters are ranked by their favela share; every threshold on that share
    that assigns both classes is tried (0.5, chance, if none does)."""
    k = int(labels.max()) + 1
    share = np.array([y[labels == c].mean() if (labels == c).any() else 0.0 for c in range(k)])
    best = 0.5
    for th in np.unique(share):
        pred = (share[labels] >= th).astype(int)
        if 0 < pred.mean() < 1:
            best = max(best, balanced_accuracy_score(y, pred))
    return best


def two_cluster_score(S: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, KMeans, int]:
    """Continuous two-cluster score s = (d_other - d_fav)/(d_other + d_fav) (SI S14 'Separation').

    d are the distances to the two k-means centroids (k = score_clusters). The
    label is used only to orient the score (positive = closer to the cluster
    with the larger favela share). Returns (score, fitted model, index of that
    cluster).
    """
    km = KMeans(int(FS["score_clusters"]), n_init=int(LF["kmeans_n_init"]), random_state=SEED).fit(S)
    fav_cluster = int(np.argmax([y[km.labels_ == c].mean() for c in (0, 1)]))
    d = np.sqrt(((S[:, None, :] - km.cluster_centers_[None]) ** 2).sum(axis=2))
    other = 1 - fav_cluster
    return (d[:, other] - d[:, fav_cluster]) / (d[:, other] + d[:, fav_cluster]), km, fav_cluster


# -----------------------------------------------------------------------------
# Boundary profile (6 m bins)
# -----------------------------------------------------------------------------
_W = float(LF["profile_bin_width_m"])
_H = float(LF["profile_half_range_m"])
EDGES = np.arange(-_H, _H + _W, _W)          # -90, -84, ..., 90
CEN = 0.5 * (EDGES[:-1] + EDGES[1:])
NORM_AT = float(LF["profile_normalise_at_m"])


def profile(sd: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mean of v in the 6 m signed-distance bins [lo, hi); NaN below the minimum count."""
    sd = np.asarray(sd, dtype=float)
    v = np.asarray(v, dtype=float)
    in_range = (sd >= EDGES[0]) & (sd < EDGES[-1])
    idx = np.digitize(sd, EDGES) - 1
    min_n = int(LF["profile_min_buildings_per_bin"])
    out = np.full(len(CEN), np.nan)
    for b in range(len(CEN)):
        sel = in_range & (idx == b)
        if sel.sum() >= min_n:
            out[b] = v[sel].mean()
    return out


def normalise(p: np.ndarray) -> np.ndarray:
    """Profile rescaled to 1 in the favela interior (d <= -50 m) and 0 in the
    formal-settlement interior (d >= +50 m) (SI S14 'Gradient')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        inner = np.nanmean(p[CEN <= -NORM_AT])
        outer = np.nanmean(p[CEN >= NORM_AT])
    return (p - outer) / (inner - outer)


def plateau_step(p: np.ndarray) -> float:
    """Inner minus outer plateau (d <= -50 m minus d >= +50 m) of an un-normalised profile."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(p[CEN <= -NORM_AT]) - np.nanmean(p[CEN >= NORM_AT]))


def pooled(per_site: np.ndarray) -> np.ndarray:
    """Mean of the per-site profiles (sites x bins), ignoring empty bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(per_site, dtype=float), axis=0)


def profile_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two normalised profiles over the bins finite in both."""
    na, nb = normalise(a), normalise(b)
    ok = np.isfinite(na) & np.isfinite(nb)
    return float(np.corrcoef(na[ok], nb[ok])[0, 1]) if ok.sum() >= 3 else np.nan


def crossing(pn: np.ndarray, level: float) -> float:
    """First downward crossing of ``level`` by a normalised profile (linear interpolation)."""
    for i in range(len(CEN) - 1):
        if np.isfinite(pn[i]) and np.isfinite(pn[i + 1]) and pn[i] >= level > pn[i + 1]:
            return float(CEN[i] + (pn[i] - level) / (pn[i] - pn[i + 1]) * (CEN[i + 1] - CEN[i]))
    return np.nan


def supervised_five_fold(sites, metric: str = "balanced_accuracy") -> pd.Series:
    """Per-site value of the supervised reference (SI Table S18; SI Fig. S23b).

    Mean over the five folds of each site of ``metric`` (``balanced_accuracy``
    or ``auc``) in the random stratified five-fold cross-validation of the SI
    S12 pipeline (XGBoost, 100 trees, seed 432), read from the fold table of
    appendix/S12_spatial_cv
    (``products.spatial_cv_folds``; rows of design
    ``appendix_S14_label_free.supervised_folds_design``, fitted folds, no
    training buffer, no matched control).
    """
    path = cfg.require_product("spatial_cv_folds")
    folds = pd.read_parquet(path)
    keep = (folds["design"] == FS["supervised_folds_design"]) & folds["site"].isin(list(sites))
    if "status" in folds.columns:
        keep &= folds["status"] == "ok"
    if "buffer" in folds.columns:
        keep &= folds["buffer"].fillna(0) == 0
    if "control" in folds.columns:
        keep &= ~folds["control"].fillna(False).astype(bool)
    f = folds[keep]
    n_folds = f.groupby("site").size()
    expected = int(cfg.get("oof.n_folds"))
    if len(n_folds) != len(list(sites)) or (n_folds != expected).any():
        raise ConfigError(f"{path.name}: expected {expected} fitted '{FS['supervised_folds_design']}' folds "
                          f"for every site, found {n_folds.to_dict()}")
    return f.groupby("site")[metric].mean().reindex(list(sites))


def site_display_name(site: str, short: bool = True) -> str:
    """Site name as printed in Fig. S23 (short forms from appendix_S14_label_free.short_site_names)."""
    names = FS.get("short_site_names") or {}
    return names[site] if short and site in names else cfg.site_name(site)
