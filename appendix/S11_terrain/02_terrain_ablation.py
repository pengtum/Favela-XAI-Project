"""Step 2 of the terrain ablation: Morph, Terrain and Combined classifiers per site.

Manuscript items
* SI Table S9 (Morph / Terrain / Combined per-site accuracy) and SI S11
  Experiments 2 and 3 (terrain-only accuracy, per-site drops, gains of the
  combined classifier); Results paragraph on terrain; Methods 4.10.
* Data of SI Fig. S14 (slope gap per site), SI Fig. S16 (terrain-only Tree
  SHAP importance), SI Fig. S17 (three-model accuracies) and SI Fig. S18
  (combined-model Tree SHAP importance).

Design (SI S11; config.yaml ``terrain``, ``appendix_S11_terrain``)
* Three classifiers per site on one shared train/test split: the 25
  morphological indicators (Morph), the nine terrain features (Terrain) and
  their union (Combined, 34 features).
* Classifier (``terrain.classifier``): ``model_main`` = the main per-site
  model of Methods 4.4 (XGBoost, 100 trees, max depth 6, learning rate 0.3,
  seed 432; unstratified 80/20 split with seed 432 drawn on the buildings in
  uID order; training rows in split order; class coding of
  ``model_main.fit_formal_as_class_1``), so that the Morph model is the main
  model of main/02_classification and the Morph column equals SI Table S3;
  ``model_cv`` = XGBoost with 100 trees and seed 432, unstratified 80/20 split
  with seed 432.
* Missing values: median of the training rows of the site.
* Tree SHAP of the Terrain and Combined models on the test buildings
  (``appendix_S11_terrain.shap_rows``); site importance = mean |SHAP| per
  feature, then the mean over the sites.
* Slope gap (SI S11 Experiment 1): |mean slope of favela buildings - mean
  slope of formal-settlement buildings| over all buildings of the site, using
  the per-building mean slope.

Inputs
* outputs/appendix/S11_terrain/<site>_buildings_with_terrain.gpkg (01).

Outputs (outputs/appendix/S11_terrain/); accuracies are fractions (0-1)
* ``summary_three_model_comparison.csv``: per site, test-set accuracy, favela
  F1 and macro F1 of the three models, and the differences to Morph
  (SI Table S9; SI Fig. S17).
* ``exp1_global_site_summary.csv``: per site, favela / formal mean slope, slope
  gap, building counts and the Morph accuracy (SI Fig. S14; the two mean slopes
  are also the slope columns of SI Table S1).
* ``exp1_pooled_test_buildings.csv``: test buildings of all sites with label,
  Morph prediction and the nine terrain features (SI Fig. S15 when
  ``appendix_S11_terrain.ovl_population: test_set``).
* ``exp2_terrain_shap_importance.csv``, ``exp3_combined_shap_importance.csv``:
  mean |SHAP| per feature and site and the average over sites (SI Figs S16, S18).

Run from the repository root:
    python appendix/S11_terrain/02_terrain_ablation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio
from sklearn.impute import SimpleImputer

from favela_xai.attribution import tree_shap
from favela_xai.config import ConfigError, load_config
from favela_xai.features import MODEL_FEATURE_ORDER
from favela_xai.io import write_table
from favela_xai.labels import label_to_int
from favela_xai.model import classification_metrics, favela_probability, fit_classifier, formal_as_class_1

import terrain_common as tc


def fit_on_split(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, cfg) -> dict:
    """Fit the ablation classifier on the training rows and predict the test rows.

    Missing values are replaced by the medians of the training rows (a feature
    that is missing for every training building is set to 0). ``y``: 1 =
    favela; the class coding of the fit follows the classifier section. The
    training rows are passed to XGBoost in the order of ``train_idx``.
    """
    cols = list(X.columns)
    X = X.astype(float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(X.iloc[train_idx])
    Xi = pd.DataFrame(imputer.transform(X), columns=cols, index=X.index)
    section = tc.ablation_classifier_section(cfg)
    formal_as_1 = formal_as_class_1(cfg, section)
    model = fit_classifier(cfg, section, Xi.iloc[train_idx], y[train_idx])
    p_fav = favela_probability(model, Xi.iloc[test_idx], formal_as_1)
    pred_test = (p_fav >= 0.5).astype(int)
    return dict(model=model, formal_as_1=formal_as_1, X_test=Xi.iloc[test_idx], pred_test=pred_test,
                metrics=classification_metrics(y[test_idx], pred_test))


def mean_abs_shap(fit: dict, cfg) -> pd.Series:
    """Mean |Tree SHAP| per feature over the explained buildings of one site.

    SHAP values point toward favela (negated for a model fitted with formal
    settlement as class 1); only their absolute values enter the importance.
    """
    if cfg.get("appendix_S11_terrain.shap_rows") != "test_set":
        raise ConfigError("appendix_S11_terrain.shap_rows: only 'test_set' is implemented")
    values = tree_shap(fit["model"], fit["X_test"], negate=fit["formal_as_1"],
                       model_output=cfg.get("shap.model_output"))
    return pd.Series(np.abs(values).mean(axis=0), index=fit["X_test"].columns)


def importance_table(per_site: dict[str, pd.Series], features: list[str], terrain: list[str] | None = None) -> pd.DataFrame:
    """Feature x site table of mean |SHAP| with the unweighted mean over the sites."""
    df = pd.DataFrame({site: s.reindex(features) for site, s in per_site.items()})
    df.index.name = "feature"
    df["average"] = df.mean(axis=1)
    df = df.reset_index()
    if terrain is not None:
        df.insert(1, "type", np.where(df["feature"].isin(terrain), "terrain", "morphological"))
    return df.sort_values("average", ascending=False, kind="mergesort").reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    terr = tc.terrain_features(cfg)
    comb = tc.combined_features(cfg)
    fav_label = cfg.get("labels.favela")

    if not cfg.get("terrain.shared_split"):
        raise ConfigError("terrain.shared_split must be true: the three models share one split per site")

    rows, slope_rows, test_frames = [], [], []
    shap_terrain, shap_combined = {}, {}
    for site in cfg.sites:
        path = tc.require(out / tc.BUILDINGS_GPKG.format(site=site), "01_building_terrain.py")
        df = pyogrio.read_dataframe(path, columns=["uID", "label"] + list(MODEL_FEATURE_ORDER) + terr,
                                    read_geometry=False)
        y = label_to_int(df["label"])

        # One split per site, shared by the three models.
        train_idx, test_idx = tc.ablation_split(y, df["uID"].to_numpy(), cfg)
        morph = fit_on_split(df[MODEL_FEATURE_ORDER], y, train_idx, test_idx, cfg)
        terrain = fit_on_split(df[terr], y, train_idx, test_idx, cfg)
        combined = fit_on_split(df[comb], y, train_idx, test_idx, cfg)

        m, t, c = morph["metrics"], terrain["metrics"], combined["metrics"]
        rows.append(dict(
            site=site, n_buildings=len(df), n_test=m["n_test"], favela_share_test=m["favela_share_test"],
            morph_accuracy=m["accuracy"], terrain_accuracy=t["accuracy"], combined_accuracy=c["accuracy"],
            delta_terrain_morph=t["accuracy"] - m["accuracy"], delta_combined_morph=c["accuracy"] - m["accuracy"],
            morph_f1_favela=m["f1_favela"], terrain_f1_favela=t["f1_favela"], combined_f1_favela=c["f1_favela"],
            morph_f1_macro=m["f1_macro"], terrain_f1_macro=t["f1_macro"], combined_f1_macro=c["f1_macro"]))

        # Tree SHAP importance of the Terrain and Combined models (Figs S16, S18).
        shap_terrain[site] = mean_abs_shap(terrain, cfg)
        shap_combined[site] = mean_abs_shap(combined, cfg)

        # Slope gap over all buildings of the site (SI S11 Experiment 1; Fig. S14).
        is_fav = df["label"].astype(str) == fav_label
        fav_slope = df.loc[is_fav, "slope_mean"].mean()
        formal_slope = df.loc[~is_fav, "slope_mean"].mean()
        slope_rows.append(dict(site=site, n_favela=int(is_fav.sum()), n_formal=int((~is_fav).sum()),
                               favela_slope=fav_slope, nonfavela_slope=formal_slope,
                               slope_gap=abs(fav_slope - formal_slope),
                               morph_accuracy=m["accuracy"]))

        # Test buildings with the Morph prediction (Fig. S15 with ovl_population = test_set).
        test = df.iloc[test_idx][["uID", "label"] + terr].copy()
        test.insert(0, "site", site)
        test["y_true"] = y[test_idx]
        test["y_pred_morph"] = morph["pred_test"]
        test["correct"] = test["y_true"] == test["y_pred_morph"]
        test_frames.append(test)

        print(f"  {site:<24s} Morph {m['accuracy']:.3f}  Terrain {t['accuracy']:.3f}  "
              f"Combined {c['accuracy']:.3f}  (n = {len(df):,}, test {m['n_test']:,})")

    summary = pd.DataFrame(rows)
    write_table(summary, out / tc.THREE_MODEL_CSV)
    write_table(pd.DataFrame(slope_rows), out / tc.SITE_SUMMARY_CSV)
    write_table(pd.concat(test_frames, ignore_index=True), out / tc.POOLED_TEST_CSV)
    write_table(importance_table(shap_terrain, terr), out / tc.TERRAIN_SHAP_CSV)
    write_table(importance_table(shap_combined, comb, terrain=terr), out / tc.COMBINED_SHAP_CSV)

    means = summary[["morph_accuracy", "terrain_accuracy", "combined_accuracy"]].mean()
    print(f"  mean accuracy: Morph {means.iloc[0]:.3f}, Terrain {means.iloc[1]:.3f}, Combined {means.iloc[2]:.3f}")
    print(f"  tables written to {out}")


if __name__ == "__main__":
    main()
