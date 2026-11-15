"""Step 1: per-site XGBoost classifiers and their Tree-SHAP attributions (the "deep features").

Manuscript items
  * Methods 4.4: one XGBoost classifier per site on the 25 indicators, with
    the xgboost defaults of the analysed models (100 trees, max depth 6,
    learning rate 0.3, hist tree method; xgboost 2.1), seed 432, on an
    unstratified 80/20 train/test split (seed 432). The test-set predictions
    give SI Table S3, Fig. 2b and the accuracies of Results 2.1
    (02_table_S3.py, 06_figure2.py); the predictions of every building
    (training and held-out test) give Fig. 2a.
  * Methods 4.6: path-dependent Tree SHAP (no background sample) of each
    site's model, an N x 25 attribution ("deep-feature") matrix per site in
    log-odds (SI Section S7); positive values push a building toward the
    favela class (Figs 3-4).
  * The per-site tables written here are the input of main/03-05, the
    Extended Data figures and most SI analyses (products.shap_gpkg); the
    models are the classifiers of the cross-site transfer (SI Section S8;
    products.model_json).

Settings (config.yaml)
  model_main   100 trees, max depth 6, learning rate 0.3 (xgboost defaults), seed 432, unstratified
               80/20 split with seed 432, eval_metric logloss; missing indicator values imputed with
               the median of the training rows; fit_formal_as_class_1: true = fitted with formal
               settlement as class 1 (the coding of the analysed models), favela probability
               = 1 - P(class 1) and Tree SHAP negated, so that the outputs point toward favela.
  shap         model_output raw (log-odds), explain all buildings of the site

Row order
  The building table is read in its own row order and every output follows that order. The
  split is drawn on the buildings ordered by uID (train_test_split), and the training rows are
  passed to XGBoost in the order train_test_split returns them (favela_xai.model.main_split);
  README.md of this folder, 'Row order'.

Inputs
  outputs/main/01_features/<site>_buildings_features.gpkg   products.features_gpkg (uID, label, 25 indicators)

Outputs (outputs/main/02_classification/)
  <site>_model.json              products.model_json: fitted XGBoost model (xgboost JSON format;
                                 class 1 = formal settlement with model_main.fit_formal_as_class_1)
  <site>_imputer_medians.csv     products.imputer_medians: training medians used for imputation
  <site>_split.csv               products.split_csv: uID, is_test (held-out 20%)
  <site>_buildings_shap.gpkg     products.shap_gpkg: uID, label, is_test, predicted_label
                                 (model prediction for every building; in-sample for the
                                 training buildings), prob_favela, the 25 indicators (as
                                 computed, before imputation), the 25 shap_<indicator>
                                 columns (log-odds, positive = toward favela), geometry
                                 (EPSG:31983)
  shap_expected_values.csv       per site: Tree-SHAP base value and the largest deviation
                                 from local additivity (base + sum of SHAP = model margin)

Run from the repository root (all sites, or a subset):
  python main/02_classification/01_train_models.py
  python main/02_classification/01_train_models.py --sites palmeras
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

from favela_xai.attribution import shap_frame, tree_shap
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER
from favela_xai.io import read_site_features, write_gpkg, write_table
from favela_xai.labels import FAVELA, FORMAL, label_to_int
from favela_xai.model import fit_main_model, formal_as_class_1


def parse_sites(cfg, description: str) -> list[str]:
    """Sites given with --sites (default: every site in config.yaml)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sites", nargs="+", help="site keys to process (default: all)")
    args = parser.parse_args()
    sites = args.sites or cfg.sites
    unknown = sorted(set(sites) - set(cfg.all_sites))
    if unknown:
        raise SystemExit(f"unknown site key(s): {unknown}; valid keys: {cfg.all_sites}")
    return sites


def check_settings(cfg) -> None:
    """The main model must be the Methods 4.4 configuration with favela as the positive class."""
    if cfg.get("model_main.imputation") != "median":
        raise ValueError("model_main.imputation: only 'median' is implemented")
    if cfg.get("model_main.positive_class") != FAVELA:
        raise ValueError("model_main.positive_class must be 'favela' (SI Table S3)")
    if cfg.get("shap.explain") != "all_buildings":
        raise ValueError("shap.explain: only 'all_buildings' is implemented (Methods 4.6)")


def main() -> None:
    cfg = load_config()
    sites = parse_sites(cfg, "Fit the per-site classifiers (Methods 4.4) and compute Tree SHAP (Methods 4.6).")
    out_dir = cfg.output_dir(__file__)
    check_settings(cfg)
    m = cfg.section("model_main")
    formal_as_1 = formal_as_class_1(cfg, "model_main")
    print(f"XGBoost: {m['n_estimators']} trees, max depth {m['max_depth']}, learning rate {m['learning_rate']}, "
          f"seed {m['random_state']}; {100 * (1 - m['test_size']):.0f}/{100 * m['test_size']:.0f} split, "
          f"stratified = {m['stratify']}, split seed {m['split_random_state']}, drawn on the rows in uID order; "
          f"class 1 of the fit = {'formal settlement' if formal_as_1 else 'favela'}")

    base_rows = []
    for site in sites:
        # Rows in the order of the per-site building table; the split is drawn on the rows in uID order.
        g = read_site_features(cfg, site)
        X = pd.DataFrame(g[MODEL_FEATURE_ORDER].astype(float).to_numpy(), columns=MODEL_FEATURE_ORDER)
        y = label_to_int(g["label"])
        sm = fit_main_model(X, y, cfg, uid=g["uID"].to_numpy())
        is_test = sm.is_test()

        # Model, imputation medians and split. The model is written through its Booster:
        # XGBClassifier.save_model of xgboost 2.1.4 fails with scikit-learn 1.8 (observed;
        # sklearn.base.ClassifierMixin no longer defines _estimator_type). The Booster JSON has
        # no scikit_learn attribute and is read with XGBClassifier.load_model (SI S8).
        sm.model.get_booster().save_model(str(cfg.product("model_json", site=site, create_parent=True)))
        write_table(sm.medians.rename_axis("indicator").reset_index(), cfg.product("imputer_medians", site=site))
        write_table(pd.DataFrame({"uID": g["uID"].to_numpy(), "is_test": is_test}), cfg.product("split_csv", site=site))

        # Tree SHAP of every building (imputed with the training medians), positive toward favela:
        # negated when the model was fitted with formal settlement as class 1.
        values = tree_shap(sm.model, sm.X_imputed, negate=sm.formal_as_1, model_output=cfg.get("shap.model_output"))
        shap = shap_frame(values, index=g.index)
        margin = sm.favela_margin()
        base = float(np.mean(margin - values.sum(axis=1)))
        additivity = float(np.max(np.abs(margin - base - values.sum(axis=1))))
        base_rows.append(dict(site=site, expected_value=base, max_additivity_error=additivity))

        prob = sm.favela_probability()
        table = pd.DataFrame({"uID": g["uID"].to_numpy(), "label": g["label"].to_numpy(), "is_test": is_test,
                              "predicted_label": np.where(prob >= 0.5, FAVELA, FORMAL), "prob_favela": prob},
                             index=g.index)
        table = pd.concat([table, g[MODEL_FEATURE_ORDER], shap], axis=1)
        write_gpkg(gpd.GeoDataFrame(table, geometry=g.geometry, crs=g.crs), cfg.product("shap_gpkg", site=site))

        met = sm.test_metrics()
        print(f"{cfg.site_name(site):24s} n {len(y):6,d} (test {int(is_test.sum()):5,d})  "
              f"test accuracy {met['accuracy']:.3f}  F1 (favela) {met['f1_favela']:.3f}  "
              f"SHAP additivity error {additivity:.1e}")

    # Base values: keep the rows of sites not processed in this run (config site order)
    path = out_dir / "shap_expected_values.csv"
    new = pd.DataFrame(base_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["site"].isin(new["site"])], new], ignore_index=True)
    order = {s: i for i, s in enumerate(cfg.all_sites)}
    write_table(new.sort_values("site", key=lambda s: s.map(order)).reset_index(drop=True), path)


if __name__ == "__main__":
    main()
