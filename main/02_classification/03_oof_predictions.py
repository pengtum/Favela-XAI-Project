"""Step 3: out-of-fold prediction of every building (stratified five-fold cross-validation).

Manuscript items
  * Methods 4.7: every building predicted out of fold with stratified
    five-fold cross-validation (XGBoost, 100 trees, seed 432); Fig. 2c.
  * Input of the misclassification rates of Results 2.1, Fig. 2c, SI Section
    S5, SI Table S4 and Extended Data Fig. 1 (04_boundary_error_rates.py,
    06_figure2.py), and of the supervised reference of SI Section S14
    (label-free clustering).

Settings (config.yaml)
  oof        5 folds, shuffled, StratifiedKFold seed 432 (Methods 4.7)
  model_cv   XGBoost 100 trees, seed 432, other settings at their defaults; missing values
             imputed with the training-fold median (0 for an all-missing column);
             fit_formal_as_class_1: true = formal settlement fitted as class 1 and favela
             probability = 1 - p, false = favela fitted as class 1
             (favela_xai.model.oof_predictions)
  A building counts as misclassified when (favela probability >= 0.5) differs from its label.

Inputs
  outputs/main/01_features/<site>_buildings_features.gpkg   products.features_gpkg (label, 25 indicators)
  outputs/main/01_features/signed_distance.parquet          products.signed_distance
                                                            (computed on first use if missing)

Outputs (outputs/main/02_classification/)
  oof_predictions.parquet   products.oof_predictions: site, uID, is_favela, signed_dist, prob
                            (out-of-fold favela probability), pred, error
  oof_summary.csv           per site: buildings, out-of-fold accuracy, balanced accuracy,
                            F1 (favela), AUC, error rate

Run from the repository root:
  python main/02_classification/03_oof_predictions.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from favela_xai.boundary import load_signed_distance
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER
from favela_xai.io import read_site_features, write_table
from favela_xai.labels import label_to_int
from favela_xai.model import oof_predictions


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    o, mc = cfg.section("oof"), cfg.section("model_cv")
    print(f"stratified {o['n_folds']}-fold CV (seed {o['random_state']}); XGBoost {mc['n_estimators']} trees, "
          f"seed {mc['random_state']}")

    signed = load_signed_distance(cfg, cfg.sites)
    parts, summary = [], []
    for site in cfg.sites:
        g = read_site_features(cfg, site, columns=["uID", "label"] + MODEL_FEATURE_ORDER)
        X = g[MODEL_FEATURE_ORDER].to_numpy(dtype=float)
        y = label_to_int(g["label"])
        prob = oof_predictions(X, y, cfg)
        pred = (prob >= 0.5).astype(int)

        sd = signed[signed["site"] == site].set_index("uID").reindex(g["uID"].to_numpy())
        if sd["signed_dist"].isna().any() or not np.array_equal(sd["is_favela"].to_numpy(bool), y.astype(bool)):
            raise SystemExit(f"{site}: signed distance and feature table do not match; "
                             "re-run main/01_features/05_signed_distance.py")
        parts.append(pd.DataFrame({"site": site, "uID": g["uID"].to_numpy(), "is_favela": y.astype(np.int8),
                                   "signed_dist": sd["signed_dist"].to_numpy(float), "prob": prob,
                                   "pred": pred.astype(np.int8), "error": pred != y}))
        summary.append(dict(site=site, n=len(y), accuracy=accuracy_score(y, pred),
                            balanced_accuracy=balanced_accuracy_score(y, pred), f1_favela=f1_score(y, pred),
                            auc=roc_auc_score(y, prob), error_rate=float((pred != y).mean())))
        s = summary[-1]
        print(f"{cfg.site_name(site):24s} out-of-fold accuracy {s['accuracy']:.3f}  AUC {s['auc']:.3f}  "
              f"error rate {100 * s['error_rate']:.2f}%")

    oof = pd.concat(parts, ignore_index=True)
    write_table(oof, cfg.product("oof_predictions"))
    summary = pd.DataFrame(summary)
    write_table(summary, out_dir / "oof_summary.csv")
    print(f"all sites: {len(oof):,} buildings, {int(oof['error'].sum()):,} misclassified "
          f"({100 * oof['error'].mean():.2f}%); mean out-of-fold accuracy {summary['accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()
