"""Helpers shared by the numbered scripts of appendix/S11_terrain.

Manuscript items: SI Section S11 (SI Table S9, SI Figs S14-S18) and Methods
4.10. This module is imported by the scripts 01-06 of this folder and is not
run on its own.

Contents
* names of the output files of this folder (all in outputs/appendix/S11_terrain/);
* the nine terrain features and the column orders of the three classifiers;
* the classifier, its class coding and the train/test split of the ablation
  (config ``terrain``); with ``terrain.classifier: model_main`` the Morph
  model is the main per-site model of main/02_classification (same split,
  training-row order, settings and class coding), so the Morph column of SI
  Table S9 equals the accuracy of SI Table S3;
* the per-site accuracy drawn in SI Figs S14 and S15;
* site names used in the figures.

All settings come from the sections ``terrain`` and ``appendix_S11_terrain`` of
config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from favela_xai.config import ConfigError
from favela_xai.features import MODEL_FEATURE_ORDER

# -----------------------------------------------------------------------------
# Output files of this folder (outputs/appendix/S11_terrain/)
# -----------------------------------------------------------------------------
SLOPE_RASTER = "slope_5m.tif"                                  # 01
BUILDINGS_GPKG = "{site}_buildings_with_terrain.gpkg"         # 01
THREE_MODEL_CSV = "summary_three_model_comparison.csv"        # 02 -> SI Table S9, Fig. S17
SITE_SUMMARY_CSV = "exp1_global_site_summary.csv"             # 02 -> Fig. S14 (and SI Table S1 slopes)
POOLED_TEST_CSV = "exp1_pooled_test_buildings.csv"            # 02 -> Fig. S15 when ovl_population = test_set
TERRAIN_SHAP_CSV = "exp2_terrain_shap_importance.csv"         # 02 -> Fig. S16
COMBINED_SHAP_CSV = "exp3_combined_shap_importance.csv"       # 02 -> Fig. S18
SLOPE_GAP_CORR_CSV = "exp1_global_correlations.csv"           # 03
OVERLAP_CSV = "terrain_overlap_by_site.csv"                   # 04
OVERLAP_CORR_CSV = "terrain_overlap_correlations.csv"         # 04
TABLE_S9_TEX = "table_S9_terrain_only_tabular.tex"            # 06
IN_TEXT_NUMBERS = "S11_in_text_numbers.txt"                   # 06

# Figure files (the file names used in the SI).
FIG_S14 = "exp1_fig3_slope_gap_vs_accuracy.pdf"
FIG_S15 = "terrain_separability_vs_accuracy"                  # .pdf and .svg
FIG_S16 = "exp2_fig6_terrain_shap_importance.pdf"
FIG_S17 = "summary_fig8_three_model_comparison.pdf"
FIG_S18 = "exp3_fig7_combined_shap_top20.pdf"


def require(path: Path, step: str) -> Path:
    """Stop with a clear message when an input written by an earlier step is missing."""
    if not Path(path).exists():
        raise ConfigError(f"{path} does not exist; run {step} of appendix/S11_terrain first.")
    return Path(path)


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
def terrain_features(cfg) -> list[str]:
    """The nine terrain features in the column order of the terrain models.

    The set is the one listed in SI S11 'Data' (``terrain.zonal_statistics``);
    the column order is ``appendix_S11_terrain.model_feature_order``.
    """
    order = list(cfg.get("appendix_S11_terrain.model_feature_order"))
    stated = list(cfg.get("terrain.zonal_statistics"))
    if sorted(order) != sorted(stated) or len(order) != 9:
        raise ConfigError("appendix_S11_terrain.model_feature_order must list the nine features of terrain.zonal_statistics")
    return order


def combined_features(cfg) -> list[str]:
    """25 morphological indicators followed by the nine terrain features (34 columns)."""
    return list(MODEL_FEATURE_ORDER) + terrain_features(cfg)


# -----------------------------------------------------------------------------
# Classifier and split of the ablation (SI Table S9 caption)
# -----------------------------------------------------------------------------
def ablation_classifier_section(cfg) -> str:
    """Config section of the classifier used for all three feature sets (``terrain.classifier``).

    ``model_main``: the Methods 4.4 settings (100 trees, max depth 6, learning
    rate 0.3, seed 432; xgboost defaults), fitted with the class coding of
    ``model_main.fit_formal_as_class_1``. ``model_cv``: 100 trees, seed 432,
    other settings at their defaults, intercept of ``model_cv.base_score``.
    Fitted with favela_xai.model.fit_classifier.
    """
    kind = cfg.get("terrain.classifier")
    if kind in ("model_main", "model_cv"):
        return kind
    raise ConfigError(f"terrain.classifier must be model_main or model_cv, not '{kind}'")


def ablation_split(y: np.ndarray, uid: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the train and test buildings of one site.

    ``model_main``: the Methods 4.4 split of the main per-site model
    (unstratified 80/20, seed 432, drawn on the buildings ordered by ``uid``;
    favela_xai.model.main_split), training rows in the order in which the
    main model is fitted. ``model_cv``: the split of
    ``appendix_S11_terrain.model_cv_split`` (80/20, unstratified, seed 432).
    """
    from sklearn.model_selection import train_test_split

    from favela_xai.model import main_split

    if cfg.get("terrain.classifier") == "model_main":
        return main_split(y, cfg, uid)
    s = cfg.get("appendix_S11_terrain.model_cv_split")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=float(s["test_size"]),
                              stratify=np.asarray(y) if s["stratify"] else None,
                              random_state=int(s["random_state"]))
    return np.sort(tr), np.sort(te)


# -----------------------------------------------------------------------------
# Per-site accuracy drawn in SI Figs S14 and S15
# -----------------------------------------------------------------------------
def _site_keys(values, cfg) -> list[str]:
    """Map a column of site keys or site names to the site keys of config.yaml."""
    by_name = {cfg.site_name(s): s for s in cfg.all_sites}
    out = []
    for v in values:
        v = str(v)
        if v in cfg.all_sites:
            out.append(v)
        elif v in by_name:
            out.append(by_name[v])
        else:
            raise ConfigError(f"unknown site '{v}' in the per-site metrics table")
    return out


def table_s3_accuracy(cfg) -> pd.Series:
    """Test-set accuracy of the main per-site classifiers (SI Table S3), as a fraction.

    Read from ``products.site_metrics`` (written by main/02_classification); the
    table needs a site column (``site`` or ``site_key``) and an ``accuracy`` column.
    """
    path = cfg.require_product("site_metrics")
    df = pd.read_csv(path)
    site_col = next((c for c in ("site", "site_key") if c in df.columns), None)
    if site_col is None or "accuracy" not in df.columns:
        raise ConfigError(f"{path} needs a 'site' (or 'site_key') column and an 'accuracy' column")
    acc = pd.Series(df["accuracy"].astype(float).to_numpy(), index=_site_keys(df[site_col], cfg), name="accuracy")
    if acc.max() > 1.0:                      # given in percent
        acc = acc / 100.0
    return acc


def figure_accuracy(cfg, out_dir: Path) -> pd.Series:
    """Per-site accuracy (fraction) used in SI Figs S14 and S15.

    ``appendix_S11_terrain.figure_accuracy``: ``table_S3`` (SI Table S3, main
    per-site classifier) or ``ablation_morph`` (Morph column of SI Table S9,
    written by 02_terrain_ablation.py).
    """
    source = cfg.get("appendix_S11_terrain.figure_accuracy")
    if source == "table_S3":
        acc = table_s3_accuracy(cfg)
    elif source == "ablation_morph":
        df = pd.read_csv(require(out_dir / THREE_MODEL_CSV, "02_terrain_ablation.py"))
        acc = df.set_index("site")["morph_accuracy"].astype(float).rename("accuracy")
    else:
        raise ConfigError(f"appendix_S11_terrain.figure_accuracy must be table_S3 or ablation_morph, not '{source}'")
    missing = [s for s in cfg.sites if s not in acc.index]
    if missing:
        raise ConfigError(f"no accuracy for sites {missing} ({source})")
    return acc.loc[cfg.sites]


# -----------------------------------------------------------------------------
# Site names in the figures
# -----------------------------------------------------------------------------
def display_name(cfg, site: str) -> str:
    """Site name in figures: config name, or the short form of appendix_S11_terrain.figure_short_names."""
    short = cfg.get("appendix_S11_terrain.figure_short_names", {}) or {}
    return short.get(site, cfg.site_name(site))


def figure_site_order(cfg) -> list[str]:
    """Order of the sites along the x axis of SI Fig. S17 (layout only)."""
    keep = set(cfg.sites)
    return [s for s in cfg.get("appendix_S11_terrain.figure_site_order") if s in keep]
