"""Helpers shared by the numbered scripts of appendix/S05_boundary_errors (not run on its own).

Used for SI Section S5 'Misclassification at favela boundaries' (Supplementary
Figs S2-S5, Supplementary Table S4) and Extended Data Fig. 1. The out-of-fold
error-rate tables are computed in main/02_classification (steps 03-05); this
folder lays them out and adds the maps and attribution figures.

Contents: output file names, the reader of the tables of main/02_classification,
``main_outputs`` (per-building label, predicted class, test flag and Tree SHAP
of the main per-site classifier), distance-band labels and small helpers.
Settings: ``boundary_errors`` and ``appendix_S05_boundary_errors`` in config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.config import ConfigError
from favela_xai.features import SHAP_COLUMNS
from favela_xai.io import read_site_shap
from favela_xai.labels import label_to_int
from favela_xai.model import main_split

SECTION = "appendix_S05_boundary_errors"
MAIN_02 = "main/02_classification"

# -----------------------------------------------------------------------------
# Files written by this folder (outputs/appendix/S05_boundary_errors/)
# -----------------------------------------------------------------------------
FIGURE_EXPOSURE = "Figure_R1_boundary_exposure.pdf"      # 01
TABLE_S4 = "table_R1_exposure_bands.tex"                 # 02
FIGURE_MAPPING = "Figure_R1_spatial_mapping.pdf"         # 03
MAPPING_SITES_DIR = "spatial_mapping_sites"              # 03 (one map per site, for assembly)
FAILURE_CSV = "failure_drivers.csv"                      # 04
FIGURE_FAILURE = "Figure_R1_failure_shap.pdf"            # 04
FIGURE_CASES_CLASS = "Figure_R1_casestudies_class.pdf"   # 05
FIGURE_CASES_DRIVER = "Figure_R1_casestudies_driver.pdf" # 05

# Column that holds the predicted class of the main per-site model in the tables of
# main/02_classification (first name found is used; main/02_classification writes 'predicted_label')
PREDICTION_COLUMNS = ("predicted_label", "prediction", "pred_label", "y_pred")
# Column that marks the held-out test buildings of the main 80/20 split, if present
TEST_FLAG_COLUMNS = ("is_test", "test", "split")


def settings(cfg) -> dict:
    """The ``appendix_S05_boundary_errors`` section of config.yaml."""
    return cfg.section(SECTION)


def require(path: Path, step: str) -> Path:
    """Stop with a clear message when an input written by an earlier step is missing."""
    if not Path(path).exists():
        raise ConfigError(f"{path} does not exist; run {step} first (see README.md).")
    return Path(path)


def main02_table(cfg, key: str) -> Path:
    """A table written by main/02_classification (file names in ``inputs_from_main_02``)."""
    name = settings(cfg)["inputs_from_main_02"][key]
    return require(cfg.output_dir(MAIN_02, create=False) / name, f"{MAIN_02} (writes {name})")


def is_favela(values) -> np.ndarray:
    """True for favela: label strings ('favela' / 'not-favela') or 1/0 codes."""
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=float) == 1
    return label_to_int(s).astype(bool)


def main_outputs(cfg, site: str, geometry: bool = False, extra=()):
    """Per-building outputs of the main per-site classifier (Methods 4.4; main/02_classification).

    Columns: uID (if present), is_favela, pred_favela (full-data prediction of
    the main model; in-sample for the training buildings), is_test (held-out 20%
    test split), the 25 ``shap_`` columns (positive toward favela) and any
    ``extra`` indicator columns; with ``geometry=True`` a GeoDataFrame with the
    footprints.

    The test flag is read from the table (main/02_classification writes
    ``is_test``); for a table without one it is re-derived with
    favela_xai.model.main_split on the labels, with the rows ordered by uID
    (as the main models draw their split; table order if the table has no uID).
    """
    path = cfg.require_product("shap_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    missing = [c for c in ["label"] + list(SHAP_COLUMNS) if c not in fields]
    if missing:
        raise ConfigError(f"{path} lacks the columns {missing[:3]}...; re-run main/02_classification.")
    pred_col = next((c for c in PREDICTION_COLUMNS if c in fields), None)
    if pred_col is None:
        raise ConfigError(f"{path} has no predicted-class column (looked for {PREDICTION_COLUMNS}).")
    test_col = next((c for c in TEST_FLAG_COLUMNS if c in fields), None)
    cols = [c for c in ("uID",) if c in fields] + ["label", pred_col] + ([test_col] if test_col else [])
    cols += list(SHAP_COLUMNS) + [c for c in extra if c not in cols]
    if geometry:
        raw = read_site_shap(cfg, site, columns=cols)
    else:
        raw = pyogrio.read_dataframe(path, columns=cols, read_geometry=False)
    out = raw.copy()
    out["is_favela"] = is_favela(raw["label"]).astype(np.int8)
    out["pred_favela"] = is_favela(raw[pred_col]).astype(np.int8)
    if test_col == "split":
        out["is_test"] = raw["split"].astype(str).str.lower().eq("test").to_numpy()
    elif test_col is not None:
        out["is_test"] = raw[test_col].astype(bool).to_numpy()
    else:
        _, te = main_split(out["is_favela"].to_numpy(int), cfg, out["uID"].to_numpy() if "uID" in out else None)
        flag = np.zeros(len(out), dtype=bool)
        flag[te] = True
        out["is_test"] = flag
    for c in SHAP_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def outcome(y_true, y_pred) -> np.ndarray:
    """Classification outcome per building: 'TP' (favela correct), 'FN', 'TN' (formal correct), 'FP'."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return np.select([(y_true == 1) & (y_pred == 1), (y_true == 1) & (y_pred == 0),
                      (y_true == 0) & (y_pred == 0), (y_true == 0) & (y_pred == 1)], ["TP", "FN", "TN", "FP"], "")


def band_labels(cfg) -> list[str]:
    """Labels of the bands of |signed distance| as written by main/02_classification:
    '< 6 m', '6-30 m', '30-90 m', '>= 90 m' for ``boundary_errors.band_edges_m`` [0, 6, 30, 90]."""
    e = [float(v) for v in cfg.get("boundary_errors.band_edges_m")]
    return [f"< {e[1]:g} m"] + [f"{a:g}-{b:g} m" for a, b in zip(e[1:-1], e[2:])] + [f">= {e[-1]:g} m"]


def site_label(cfg, site: str, short: bool = False) -> str:
    """Display name of a site; the abbreviated form where a figure abbreviates (``short_names``)."""
    if short:
        names = settings(cfg).get("short_names") or {}
        if site in names:
            return names[site]
    return cfg.site_name(site)
