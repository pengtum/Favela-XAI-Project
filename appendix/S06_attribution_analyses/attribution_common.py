"""Helpers shared by the scripts of appendix/S06_attribution_analyses.

Imported by scripts 01-06 (Extended Data Fig. 2, Supplementary Section S6 and
SI Figs S6-S9); not run on its own.

* ``shap_table``: per-building table of one site from main/02_classification
  (label, 25 Tree-SHAP columns and, on request, the 25 indicators and the
  predicted class), read without geometry;
* ``add_signed_distance``: joins the signed distance to the official 2013
  favela boundary (``products.signed_distance``);
* ``dimension_panel_order``: panel order of SI Figs S7 and S8 (same rule as Fig. 4);
* site order, display names and figure style.

Settings: config.yaml sections ``importance``, ``dependence``,
``boundary_profiles``, ``boundary_direction`` and
``appendix_S06_attribution_analyses``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.boundary import load_signed_distance
from favela_xai.config import ConfigError
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, features_by_dimension
from favela_xai.labels import label_to_int
from favela_xai.plotting_style import use_style

SECTION = "appendix_S06_attribution_analyses"

# Column that holds the predicted class of the main per-site model in the
# per-site tables of main/02_classification (the first name found is used).
PREDICTION_COLUMNS = ("predicted_label", "prediction", "pred_label", "y_pred")


def settings(cfg) -> dict:
    """The ``appendix_S06_attribution_analyses`` section of config.yaml."""
    return cfg.section(SECTION)


# -----------------------------------------------------------------------------
# Per-site building tables
# -----------------------------------------------------------------------------
def _is_favela(values: pd.Series) -> np.ndarray:
    """True for favela: label strings ('favela' / 'not-favela') or 1/0 codes."""
    if pd.api.types.is_numeric_dtype(values):
        return np.asarray(values, dtype=float) == 1
    return label_to_int(values).astype(bool)


def shap_table(cfg, site: str, indicators: bool = False, prediction: bool = False) -> pd.DataFrame:
    """Per-building table of one site, without geometry.

    Columns: ``site``, ``uID`` (when present), ``is_favela`` (label from the
    official 2013 polygons), the 25 ``shap_<indicator>`` columns (Tree SHAP of
    the main per-site model, positive = toward favela) and, on request, the 25
    indicator values, ``pred_favela`` (1 when the main model classifies the
    building favela) and ``is_test`` (held-out test building of the main
    model's split, when the table carries that flag).
    """
    path = cfg.require_product("shap_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    missing = [c for c in ["label"] + SHAP_COLUMNS if c not in fields]
    if missing:
        raise ConfigError(f"{path} lacks the columns {missing}; re-run main/02_classification.")
    cols = [c for c in ("uID",) if c in fields] + ["label"] + list(SHAP_COLUMNS)
    if indicators:
        cols += [f for f in MODEL_FEATURE_ORDER if f not in cols]
    pred_col = None
    if prediction:
        pred_col = next((c for c in PREDICTION_COLUMNS if c in fields), None)
        if pred_col is None:
            raise ConfigError(f"{path} has no predicted-class column (looked for {PREDICTION_COLUMNS}).")
        cols.append(pred_col)
        if "is_test" in fields:
            cols.append("is_test")
    raw = pyogrio.read_dataframe(path, columns=cols, read_geometry=False)

    out = pd.DataFrame({"site": site}, index=raw.index)
    if "uID" in raw.columns:
        out["uID"] = raw["uID"].to_numpy()
    out["is_favela"] = _is_favela(raw["label"])
    numeric = list(SHAP_COLUMNS) + (list(MODEL_FEATURE_ORDER) if indicators else [])
    for c in numeric:
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    if pred_col is not None:
        out["pred_favela"] = _is_favela(raw[pred_col]).astype(float)
    if "is_test" in raw.columns:
        out["is_test"] = raw["is_test"].astype(bool).to_numpy()
    return out.reset_index(drop=True)


def add_signed_distance(cfg, table: pd.DataFrame) -> pd.DataFrame:
    """Add ``signed_dist`` (m; negative on the favela-labelled side) to a site table.

    The distance is the shared product of favela_xai.boundary.load_signed_distance
    (Methods 4.8: centroid to the nearest edge of the official 2013 favela
    polygon), joined on (site, uID).
    """
    if "uID" not in table.columns:
        raise ConfigError("the per-site table has no uID column, so the signed distance cannot be joined")
    sites = list(table["site"].unique())
    sd = load_signed_distance(cfg, sites=sites)[["site", "uID", "is_favela", "signed_dist"]]
    sd = sd.rename(columns={"is_favela": "is_favela_sd"})
    out = table.merge(sd, on=["site", "uID"], how="left", validate="one_to_one")
    n_missing = int(out["signed_dist"].isna().sum())
    if n_missing:
        raise ConfigError(f"{n_missing} buildings of {sites} have no signed distance; "
                          "re-run main/01_features (products.signed_distance).")
    if not np.array_equal(out["is_favela"].to_numpy(bool), out["is_favela_sd"].to_numpy(bool)):
        raise ConfigError(f"labels of main/02_classification and of the signed-distance product differ for {sites}")
    return out.drop(columns="is_favela_sd")


# -----------------------------------------------------------------------------
# Orders and names
# -----------------------------------------------------------------------------
def site_order(cfg) -> list[str]:
    """Sites in the figure panel order (``site_order``), restricted to the sites processed."""
    order = [s for s in settings(cfg)["site_order"] if s in cfg.sites]
    return order + [s for s in cfg.sites if s not in order]


def site_label(cfg, site: str, short: bool = False) -> str:
    """Display name of a site (the abbreviated form where a figure abbreviates)."""
    if short:
        names = settings(cfg).get("short_site_names") or {}
        if site in names:
            return names[site]
    return cfg.site_name(site)


def dimension_panel_order(buildings: pd.DataFrame) -> tuple[list[str], list[str], pd.Series]:
    """Panel order of SI Figs S7 and S8 (same rule as Fig. 4).

    Mean |SHAP| of each indicator over all buildings of all sites; dimensions
    ordered by the mean of their indicators' mean |SHAP|, indicators by their
    own mean |SHAP| within the dimension. Returns (indicator order, dimension
    order, mean |SHAP| by indicator).
    """
    imp = buildings[SHAP_COLUMNS].abs().mean()
    imp.index = MODEL_FEATURE_ORDER
    groups = features_by_dimension()
    gmean = {g: float(np.mean([imp[f] for f in fs])) for g, fs in groups.items()}
    gorder = sorted(groups, key=lambda g: -gmean[g])
    order = [f for g in gorder for f in sorted(groups[g], key=lambda f: -imp[f])]
    return order, gorder, imp


# -----------------------------------------------------------------------------
# Figure style
# -----------------------------------------------------------------------------
def figure_style(font_size: float) -> None:
    """Shared figure style (Arial, embedded TrueType fonts, black axes) with the
    given base font size; label sizes relative to the base size."""
    use_style(**{
        "font.size": font_size,
        "axes.titlesize": "large", "axes.labelsize": "medium",
        "xtick.labelsize": "medium", "ytick.labelsize": "medium", "legend.fontsize": "medium",
        "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
        "text.color": "black", "axes.labelcolor": "black",
    })


def save_pdf(fig, path: Path, dpi: int = 300) -> Path:
    """Save a figure as PDF with a tight bounding box (dpi applies to rasterised layers only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return path
