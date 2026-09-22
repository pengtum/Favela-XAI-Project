"""Classifiers: the main per-site model and the validation-pipeline model.

Manuscript items
* Main per-site classifier (Main Methods 4.4 'Explainable-AI framework:
  gradient-boosted favela classification'): XGBoost, 200 trees, max depth 6,
  learning rate 0.1, random seed 42, stratified 80/20 train/test split, one
  model per site. Its test-set metrics are SI Table S3 / Main Fig. 2b; its
  Tree-SHAP attributions are the "deep features" of the whole analysis; the
  same models and test splits are used for the cross-site transfer (SI S8).
* Validation-pipeline classifier (Main Methods 4.5 and 4.7; SI S5, S12, S14):
  XGBoost, 100 trees, seed 432, other settings at their defaults, missing
  values imputed with the training median. Used for the out-of-fold
  predictions (Main Fig. 2c, ED Fig. 1) and every spatial cross-validation fit.

All settings come from ``model_main``, ``model_cv`` and ``oof`` in config.yaml.
Feature columns are always used in ``features.MODEL_FEATURE_ORDER``.
Class coding of the main model: 1 = favela, 0 = formal settlement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .features import MODEL_FEATURE_ORDER


# -----------------------------------------------------------------------------
# Main per-site classifier (Methods 4.4)
# -----------------------------------------------------------------------------
def main_classifier(cfg) -> XGBClassifier:
    """Unfitted XGBoost classifier with the Methods 4.4 settings."""
    p = cfg.section("model_main")
    return XGBClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
                         learning_rate=float(p["learning_rate"]), random_state=int(p["random_state"]),
                         eval_metric=p["eval_metric"])


def main_split(y: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the 80/20 train/test split (stratified by label, seed 42)."""
    p = cfg.section("model_main")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=float(p["test_size"]),
                              stratify=np.asarray(y) if p["stratify"] else None,
                              random_state=int(p["split_random_state"]))
    return np.sort(tr), np.sort(te)


@dataclass
class SiteModel:
    """A fitted per-site model with everything needed downstream."""
    model: XGBClassifier
    imputer: SimpleImputer
    train_idx: np.ndarray          # row positions of the training buildings
    test_idx: np.ndarray           # row positions of the held-out test buildings
    X_imputed: pd.DataFrame        # all buildings of the site, imputed with training medians
    y: np.ndarray                  # 1 = favela, 0 = formal settlement

    @property
    def medians(self) -> pd.Series:
        return pd.Series(self.imputer.statistics_, index=MODEL_FEATURE_ORDER, name="training_median")

    def favela_probability(self, X: pd.DataFrame | None = None) -> np.ndarray:
        X = self.X_imputed if X is None else X
        return self.model.predict_proba(X[MODEL_FEATURE_ORDER])[:, 1]

    def predict(self, X: pd.DataFrame | None = None) -> np.ndarray:
        return (self.favela_probability(X) >= 0.5).astype(int)

    def is_test(self) -> np.ndarray:
        flag = np.zeros(len(self.y), dtype=bool)
        flag[self.test_idx] = True
        return flag

    def test_metrics(self) -> dict:
        return classification_metrics(self.y[self.test_idx], self.predict()[self.test_idx])


def fit_main_model(X: pd.DataFrame, y, cfg) -> SiteModel:
    """Fit the Methods 4.4 classifier of one site.

    Missing indicator values are replaced by the median of the training rows;
    the same medians are applied to every building of the site before
    prediction and Tree SHAP.
    """
    y = np.asarray(y, dtype=int)
    X = X[MODEL_FEATURE_ORDER].astype(float)
    tr, te = main_split(y, cfg)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(X.iloc[tr])
    Xi = pd.DataFrame(imputer.transform(X), columns=MODEL_FEATURE_ORDER, index=X.index)
    model = main_classifier(cfg).fit(Xi.iloc[tr], y[tr])
    return SiteModel(model=model, imputer=imputer, train_idx=tr, test_idx=te, X_imputed=Xi, y=y)


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
def cv_classifier(cfg, n_jobs: int | None = None) -> XGBClassifier:
    """Unfitted XGBoost classifier of the validation pipeline (100 trees, seed 432, defaults otherwise)."""
    p = cfg.section("model_cv")
    kwargs = dict(n_estimators=int(p["n_estimators"]), random_state=int(p["random_state"]),
                  eval_metric=p["eval_metric"])
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    return XGBClassifier(**kwargs)


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
    ytr = np.asarray(ytr, dtype=int)
    formal_as_1 = bool(cfg.get("model_cv.fit_formal_as_class_1", True))
    m = cv_classifier(cfg, n_jobs).fit(Xtr, 1 - ytr if formal_as_1 else ytr)
    p1 = m.predict_proba(Xte)[:, 1]
    return m, (1 - p1 if formal_as_1 else p1), Xte


def oof_predictions(X: np.ndarray, y: np.ndarray, cfg, n_jobs: int | None = None) -> np.ndarray:
    """Out-of-fold favela probability of every building (Methods 4.7).

    Stratified five-fold cross-validation (shuffled, seed 432) with the
    validation-pipeline model; each building is predicted by the model of the
    fold in which it was held out.
    """
    o = cfg.section("oof")
    X = np.asarray(X, float)
    y = np.asarray(y, dtype=int)
    prob = np.full(len(y), np.nan)
    skf = StratifiedKFold(int(o["n_folds"]), shuffle=bool(o["shuffle"]), random_state=int(o["random_state"]))
    for tr, te in skf.split(X, y):
        _, prob[te], _ = fit_predict_cv(X[tr], y[tr], X[te], cfg, n_jobs)
    return prob
