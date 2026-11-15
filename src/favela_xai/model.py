"""Classifiers: the main per-site model, the validation-pipeline model and the SI S9 classifier.

Manuscript items
* Main per-site classifier (Main Methods 4.4 'Explainable-AI framework:
  gradient-boosted favela classification'): XGBoost with the xgboost defaults
  of the analysed models (100 trees, max depth 6, learning rate 0.3, hist tree
  method; xgboost 2.1, environment.yml, checked when the model is fitted:
  ``require_main_model_xgboost``), random seed 432, one model per site,
  fitted on an unstratified 80/20 train/test split (seed 432); missing values
  imputed with the training median. Its test-set metrics are SI Table S3 /
  Main Fig. 2b; its Tree-SHAP attributions are the "deep features" of the
  whole analysis; the same models and test splits are used for the cross-site
  transfer (SI S8) and the Morph column of the terrain ablation (SI S11).
* Validation-pipeline classifier (Main Methods 4.5 and 4.7; SI S5, S12, S14,
  S15): the same settings (100 trees, seed 432, xgboost defaults otherwise),
  missing values imputed with the training median, and the intercept of each
  fit set to the share of class 1 among its training rows
  (``model_cv.base_score``). Used for the out-of-fold predictions (Main Fig.
  2c, ED Fig. 1) and every spatial cross-validation fit.
* Classifier of the SI S9 feature-set comparison (SI Tables S6-S7):
  ``block_complexity.classifier``.

Row order of the main split (``main_split``): the split is drawn with
sklearn's train_test_split on the buildings of the site ordered by uID, and the
training rows are passed to XGBoost in the order train_test_split returns them
(xgboost's hist quantile sketch depends on that order in the larger sites).
The row positions returned refer to the per-building table as read, so every
output stays aligned to that table.

Class coding: with ``<section>.fit_formal_as_class_1`` a classifier is fitted
with formal settlement as class 1 (the coding of the analysed models); its
favela probability is 1 - P(class 1) (``favela_probability``) and its Tree-SHAP
values are negated to point toward favela (attribution.tree_shap). Labels
passed to the functions here are always 1 = favela, 0 = formal settlement.

All settings come from ``model_main``, ``model_cv``, ``oof`` and
``block_complexity.classifier`` in config.yaml. Feature columns are always used
in ``features.MODEL_FEATURE_ORDER``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .features import MODEL_FEATURE_ORDER

# XGBoost parameters read from a classifier section of config.yaml (others stay at the xgboost defaults)
_XGB_PARAMS = {"n_estimators": int, "max_depth": int, "learning_rate": float, "random_state": int,
               "eval_metric": str}

# xgboost series with which the analysed per-site models (model_main) were fitted
MAIN_MODEL_XGBOOST_SERIES = "2.1"


def require_main_model_xgboost() -> None:
    """Stop unless the installed xgboost is of the series of the analysed per-site models (2.1).

    ``model_main`` leaves the intercept (base_score) to xgboost, and xgboost
    3.x estimates it from the training labels differently from xgboost 2.1;
    with another series the main models (their Tree SHAP, SI Table S3 and the
    SI S8 transfer) and the model_main fits of SI S11 and S12 would change
    without notice. The validation and SI S9 fits set their intercept
    explicitly.
    """
    if not xgboost.__version__.startswith(MAIN_MODEL_XGBOOST_SERIES + "."):
        raise RuntimeError(
            f"xgboost {xgboost.__version__} is installed; the per-site models (model_main) are fitted with "
            f"xgboost {MAIN_MODEL_XGBOOST_SERIES}.x and its default intercept (environment.yml, "
            "requirements.txt; README.md, Installation)")


# -----------------------------------------------------------------------------
# Classifier construction and class coding (all sections)
# -----------------------------------------------------------------------------
def _intercept(p: dict, y_fit) -> dict:
    """``base_score`` keyword of one fit from the section's ``base_score`` rule.

    ``xgboost_default`` (or absent): not set; the installed xgboost estimates
    the intercept from the training labels (xgboost 2.1: one Newton step from
    0.5). ``training_label_mean``: the share of class 1 among the training
    rows of the fit, the intercept that xgboost 3.0 and later estimate by
    default. It needs both classes among the training rows (xgboost 2.1
    requires 0 < base_score < 1); the flows of the repository never fit a
    one-class training set (the spatial folds skip it,
    spatial_cv.skip_reason; the other folds and splits are stratified).
    """
    rule = p.get("base_score", "xgboost_default")
    if rule in (None, "xgboost_default"):
        return {}
    if rule == "training_label_mean":
        if y_fit is None:
            raise ValueError("base_score 'training_label_mean' needs the training labels of the fit (y_fit)")
        share = float(np.mean(np.asarray(y_fit, dtype=float)))
        if not 0.0 < share < 1.0:
            raise ValueError(f"base_score 'training_label_mean': the training rows of the fit hold one class "
                             f"only (share of class 1 = {share}); xgboost needs 0 < base_score < 1")
        return {"base_score": share}
    raise ValueError(f"unknown base_score rule '{rule}' (xgboost_default | training_label_mean)")


def xgb_classifier(p: dict, y_fit=None, n_jobs: int | None = None) -> XGBClassifier:
    """Unfitted XGBoost classifier with the settings of one config section ``p``.

    ``y_fit``: the labels the model will be fitted on (in the class coding of
    the fit), needed for the intercept rule ``training_label_mean``.
    """
    kwargs = {k: f(p[k]) for k, f in _XGB_PARAMS.items() if p.get(k) is not None}
    kwargs.update(_intercept(p, y_fit))
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    return XGBClassifier(**kwargs)


def formal_as_class_1(cfg, section: str) -> bool:
    """Class coding of the fits of a classifier section: True = formal settlement is class 1.

    Without the key: true for model_cv, false for the other sections.
    """
    return bool(cfg.get(f"{section}.fit_formal_as_class_1", section == "model_cv"))


def fit_classifier(cfg, section: str, X, y, n_jobs: int | None = None, **params) -> XGBClassifier:
    """Fit the classifier of ``section`` (model_main | model_cv | block_complexity.classifier).

    ``y`` is 1 for favela; the model is fitted on 1 - y when the section sets
    ``fit_formal_as_class_1``. The rows of ``X`` are passed to XGBoost in the
    given order. ``params`` override single XGBoost parameters (e.g. a seed).
    A ``model_main`` fit needs xgboost 2.1 (``require_main_model_xgboost``).
    """
    if section == "model_main":
        require_main_model_xgboost()
    y = np.asarray(y, dtype=int)
    y_fit = 1 - y if formal_as_class_1(cfg, section) else y
    model = xgb_classifier(cfg.section(section), y_fit, n_jobs)
    if params:
        model.set_params(**params)
    return model.fit(X, y_fit)


def favela_probability(model, X, formal_as_1: bool) -> np.ndarray:
    """P(favela) of a fitted classifier: P(class 1), or 1 - P(class 1) for a model fitted formal as class 1."""
    p1 = model.predict_proba(X)[:, 1]
    return 1 - p1 if formal_as_1 else p1


# -----------------------------------------------------------------------------
# Main per-site classifier (Methods 4.4)
# -----------------------------------------------------------------------------
def main_classifier(cfg, n_jobs: int | None = None, y_fit=None) -> XGBClassifier:
    """Unfitted XGBoost classifier with the Methods 4.4 settings (``model_main``; needs xgboost 2.1)."""
    require_main_model_xgboost()
    return xgb_classifier(cfg.section("model_main"), y_fit, n_jobs)


def main_split(y: np.ndarray, cfg, uid) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the 80/20 train/test split of the main per-site model.

    The split is drawn with train_test_split (``model_main``: test_size,
    stratify, split_random_state) on the buildings ordered by ``uid`` (stable
    sort), as the analysed models drew it; ``uid=None`` takes the rows in the
    order given. Returned: positions in the row order of the table passed in;
    the training rows in the order train_test_split returns them (the order in
    which they are passed to XGBoost), the test rows in table order.
    """
    p = cfg.section("model_main")
    y = np.asarray(y)
    order = np.arange(len(y)) if uid is None else np.argsort(np.asarray(uid), kind="stable")
    tr, te = train_test_split(np.arange(len(y)), test_size=float(p["test_size"]),
                              stratify=y[order] if p["stratify"] else None,
                              random_state=int(p["split_random_state"]))
    return order[tr], np.sort(order[te])


@dataclass
class SiteModel:
    """A fitted per-site model with everything needed downstream."""
    model: XGBClassifier
    imputer: SimpleImputer
    train_idx: np.ndarray          # row positions of the training buildings, in the order passed to XGBoost
    test_idx: np.ndarray           # row positions of the held-out test buildings
    X_imputed: pd.DataFrame        # all buildings of the site, imputed with training medians
    y: np.ndarray                  # 1 = favela, 0 = formal settlement
    formal_as_1: bool = False      # True: the model was fitted with formal settlement as class 1

    @property
    def medians(self) -> pd.Series:
        return pd.Series(self.imputer.statistics_, index=MODEL_FEATURE_ORDER, name="training_median")

    def favela_probability(self, X: pd.DataFrame | None = None) -> np.ndarray:
        X = self.X_imputed if X is None else X
        return favela_probability(self.model, X[MODEL_FEATURE_ORDER], self.formal_as_1)

    def favela_margin(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """Raw model output (log-odds) toward favela."""
        X = self.X_imputed if X is None else X
        margin = self.model.predict(X[MODEL_FEATURE_ORDER], output_margin=True)
        return -margin if self.formal_as_1 else margin

    def predict(self, X: pd.DataFrame | None = None) -> np.ndarray:
        return (self.favela_probability(X) >= 0.5).astype(int)

    def is_test(self) -> np.ndarray:
        flag = np.zeros(len(self.y), dtype=bool)
        flag[self.test_idx] = True
        return flag

    def test_metrics(self) -> dict:
        return classification_metrics(self.y[self.test_idx], self.predict()[self.test_idx])


def fit_main_model(X: pd.DataFrame, y, cfg, uid) -> SiteModel:
    """Fit the Methods 4.4 classifier of one site.

    ``uid``: the uID of every row of ``X`` (the split is drawn on the rows
    ordered by uID; ``main_split``). Missing indicator values are replaced by
    the median of the training rows; the same medians are applied to every
    building of the site before prediction and Tree SHAP. The rows of ``X``
    keep their order in every output.
    """
    y = np.asarray(y, dtype=int)
    X = X[MODEL_FEATURE_ORDER].astype(float)
    tr, te = main_split(y, cfg, uid)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(X.iloc[tr])
    Xi = pd.DataFrame(imputer.transform(X), columns=MODEL_FEATURE_ORDER, index=X.index)
    model = fit_classifier(cfg, "model_main", Xi.iloc[tr], y[tr])
    return SiteModel(model=model, imputer=imputer, train_idx=tr, test_idx=te, X_imputed=Xi, y=y,
                     formal_as_1=formal_as_class_1(cfg, "model_main"))


def classification_metrics(y_true, y_pred) -> dict:
    """SI Table S3 columns with favela (1) as the positive class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return dict(
        n_test=int(len(y_true)),
        favela_share_test=float(y_true.mean()) if len(y_true) else np.nan,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_favela=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


# -----------------------------------------------------------------------------
# Validation-pipeline classifier (Methods 4.5, 4.7; SI S12)
# -----------------------------------------------------------------------------
def cv_classifier(cfg, n_jobs: int | None = None, y_fit=None) -> XGBClassifier:
    """Unfitted XGBoost classifier of the validation pipeline.

    100 trees and seed 432, the intercept of ``model_cv.base_score``, the
    xgboost defaults otherwise (max depth 6, learning rate 0.3).

    ``y_fit`` (the labels of the fit, in its class coding) sets the intercept
    of ``model_cv.base_score``.
    """
    return xgb_classifier(cfg.section("model_cv"), y_fit, n_jobs)


def impute_training_median(Xtr: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Replace NaN by the column medians of the training rows (0 for an all-missing column)."""
    med = np.nanmedian(Xtr, axis=0) if len(Xtr) else np.zeros(Xtr.shape[1])
    med = np.where(np.isnan(med), 0.0, med)
    return tuple(np.where(np.isnan(a), med, a) for a in (Xtr, *others))


def fit_predict_cv(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, cfg, n_jobs: int | None = None):
    """Fit the validation-pipeline model and return (model, favela probability of Xte, imputed Xte).

    ``ytr`` is 1 for favela. With ``model_cv.fit_formal_as_class_1`` the model
    is trained on 1 - y and the favela probability is
    1 - P(class 1); Tree SHAP values of such a model must be negated to point
    toward favela (see attribution.tree_shap).
    """
    Xtr, Xte = impute_training_median(np.asarray(Xtr, float), np.asarray(Xte, float))
    m = fit_classifier(cfg, "model_cv", Xtr, ytr, n_jobs)
    return m, favela_probability(m, Xte, formal_as_class_1(cfg, "model_cv")), Xte


def oof_predictions(X: np.ndarray, y: np.ndarray, cfg, n_jobs: int | None = None) -> np.ndarray:
    """Out-of-fold favela probability of every building (Methods 4.7).

    Stratified five-fold cross-validation (shuffled, seed 432) with the
    validation-pipeline model; each building is predicted by the model of the
    fold in which it was held out. Folds are assigned by row position, so the
    rows are used in the order given (the row order of the per-building table).
    """
    o = cfg.section("oof")
    X = np.asarray(X, float)
    y = np.asarray(y, dtype=int)
    prob = np.full(len(y), np.nan)
    skf = StratifiedKFold(int(o["n_folds"]), shuffle=bool(o["shuffle"]), random_state=int(o["random_state"]))
    for tr, te in skf.split(X, y):
        _, prob[te], _ = fit_predict_cv(X[tr], y[tr], X[te], cfg, n_jobs)
    return prob
