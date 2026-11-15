"""Tree-SHAP attributions ("deep features") and their site-level summaries.

Manuscript items
* Main Methods 4.6 'Deep-feature extraction via XAI feature attribution and PCA
  signature': Tree SHAP of each site's trained XGBoost model gives an N x 25
  deep-feature matrix; the 25 attribution columns are standardised over the
  buildings of all ten sites and one PCA is fitted to the pooled matrix
  (first two components kept; Main Fig. 5; SI S7 and SI Fig. S10).
* Site-level importance: mean |SHAP| over a site's favela buildings and its
  share of the total (ED Fig. 2 per-site importance; ED Figs 3-4 ranking
  stability; SI S8 attribution composition).
* Dominant feature of a building: the indicator with the largest signed
  contribution toward favela (Main Fig. 3).

Sign convention: positive SHAP values push a building toward the favela class
(orange in Main Figs 3-4). Models fitted with formal settlement as class 1
(``fit_formal_as_class_1`` of model_main and model_cv, the coding of the
analysed models) are negated with ``negate=True``; a model fitted with favela
as class 1 already follows this convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MODEL_FEATURE_ORDER, SHAP_COLUMNS


def tree_shap(model, X, negate: bool = False, model_output: str = "raw") -> np.ndarray:
    """Exact Tree-SHAP values (log-odds contributions), shape (n buildings, n features).

    Path-dependent Tree SHAP (Methods 4.6): no background sample is passed, so
    shap.TreeExplainer uses feature_perturbation="tree_path_dependent", which
    weights the two branches of a split on a withheld indicator by the cover
    recorded in the fitted trees.
    """
    import shap

    explainer = shap.TreeExplainer(model, model_output=model_output,
                                   feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(X)
    if isinstance(values, list):             # older shap versions: one array per class
        values = values[1]
    values = np.asarray(values, dtype=float)
    return -values if negate else values


def shap_frame(values: np.ndarray, index=None, features=MODEL_FEATURE_ORDER) -> pd.DataFrame:
    """SHAP matrix as a DataFrame with columns ``shap_<indicator>``."""
    return pd.DataFrame(values, index=index, columns=["shap_" + f for f in features])


def favela_importance(shap_df: pd.DataFrame, is_favela, features=MODEL_FEATURE_ORDER) -> pd.DataFrame:
    """Mean |SHAP| over the favela buildings of one site and each indicator's share.

    Returns a DataFrame indexed by indicator with columns ``mean_abs_shap`` and
    ``share`` (shares sum to one).
    """
    cols = ["shap_" + f for f in features]
    fav = np.asarray(is_favela, dtype=bool)
    mean_abs = shap_df.loc[fav, cols].abs().mean().to_numpy()
    out = pd.DataFrame({"mean_abs_shap": mean_abs}, index=pd.Index(features, name="feature"))
    out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out


def dominant_feature(shap_df: pd.DataFrame, features=MODEL_FEATURE_ORDER) -> pd.Series:
    """Indicator with the largest signed contribution toward favela, per building."""
    cols = ["shap_" + f for f in features]
    idx = np.argmax(shap_df[cols].to_numpy(), axis=1)
    return pd.Series(np.asarray(features)[idx], index=shap_df.index, name="dominant_feature")


# -----------------------------------------------------------------------------
# Pooled attribution PCA (Methods 4.6; Main Fig. 5)
# -----------------------------------------------------------------------------
def attribution_pca(shap_matrix: np.ndarray, is_favela, cfg) -> dict:
    """Standardise the pooled 25 attribution columns and fit one PCA.

    ``shap_matrix``: SHAP values of the buildings of all ten sites stacked
    (rows) in ``MODEL_FEATURE_ORDER`` (columns). Standardisation uses the
    population standard deviation (scikit-learn StandardScaler).

    Component signs are arbitrary and can differ between library versions,
    so they are fixed explicitly (``pca.orient_pc1``, ``pca.orient_pc2`` in
    config.yaml).

    Returns a dict with ``scores`` (n x 25), ``explained_variance_ratio``,
    ``loadings`` (25 components x 25 indicators, DataFrame), ``scaler_mean``,
    ``scaler_scale`` and ``flipped`` (components whose sign was reversed).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    p = cfg.section("pca")
    X = np.asarray(shap_matrix, dtype=float)
    fav = np.asarray(is_favela, dtype=bool)
    scaler = StandardScaler(with_mean=True, with_std=bool(p["standardise"]))
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=X.shape[1], svd_solver="full")
    scores = pca.fit_transform(Z)
    comps = pca.components_.copy()

    flipped = []
    # PC1: favela mean above formal mean
    if p.get("orient_pc1") == "favela_positive" and scores[fav, 0].mean() < scores[~fav, 0].mean():
        scores[:, 0] *= -1; comps[0] *= -1; flipped.append(1)
    # PC2: largest absolute loading positive (sign convention)
    if p.get("orient_pc2") == "largest_loading_positive" and comps[1, np.argmax(np.abs(comps[1]))] < 0:
        scores[:, 1] *= -1; comps[1] *= -1; flipped.append(2)

    names = [f"PC{i + 1}" for i in range(comps.shape[0])]
    return dict(
        scores=scores,
        explained_variance_ratio=pd.Series(pca.explained_variance_ratio_, index=names, name="explained_variance_ratio"),
        loadings=pd.DataFrame(comps, index=names, columns=MODEL_FEATURE_ORDER),
        scaler_mean=pd.Series(scaler.mean_, index=MODEL_FEATURE_ORDER, name="mean"),
        scaler_scale=pd.Series(scaler.scale_ if scaler.scale_ is not None else np.ones(X.shape[1]),
                               index=MODEL_FEATURE_ORDER, name="scale"),
        flipped=flipped,
    )


def youden_threshold(score, is_favela) -> dict:
    """ROC-Youden optimum of a score with favela as the positive class (SI S7, SI Fig. S10).

    ``threshold`` is the ROC operating point that maximises TPR - FPR; at that
    point (and in ``tpr`` / ``fpr``) buildings with ``score >= threshold`` count
    as positive. The maps of SI Fig. S10 call a building favela-like when its
    score is strictly above the threshold (``appendix_S07_pca.favela_like_rule``);
    the two conventions differ only for a building exactly at the threshold.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(is_favela, dtype=int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    k = int(np.argmax(tpr - fpr))
    return dict(threshold=float(thr[k]), youden_j=float(tpr[k] - fpr[k]), tpr=float(tpr[k]),
                fpr=float(fpr[k]), auc=float(roc_auc_score(y, s)), n=int(len(y)))


__all__ = ["tree_shap", "shap_frame", "favela_importance", "dominant_feature", "attribution_pca",
           "youden_threshold", "SHAP_COLUMNS"]
