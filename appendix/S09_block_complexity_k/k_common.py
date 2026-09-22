"""Helpers shared by the scripts of appendix/S09_block_complexity_k (imported, not run).

Manuscript items: SI Section S9 (SI Tables S6-S7, SI Fig. S13) and Methods 4.9.

* ``read_k``: per-building k written by 02_block_complexity_k.py.
* ``site_table``: one site's buildings without geometry (uID, label, y with
  1 = favela, the 25 raw indicators of main/01_features, the 25 Tree-SHAP
  columns of the main per-site model of main/02_classification, and k), joined
  on the building id ``uID``.
* The two building subsets of SI S9:
  - ``valid_k`` (distribution of k; SI S9.2 and SI Fig. S13): a valid k
    (k >= 1; buildings outside every block, without a tessellation cell or not
    connected to the block boundary have none), complete Tree-SHAP
    attributions and a label;
  - ``evaluation_subset`` (classification comparison; SI Tables S6-S7): in
    addition, all 25 raw indicators present.
* ``tex_number`` / ``tex_int``: numbers as typeset in the SI tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.config import ConfigError
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS
from favela_xai.labels import label_to_int

RAW = list(MODEL_FEATURE_ORDER)       # the 25 indicators in model column order
SHAP = list(SHAP_COLUMNS)
K_FILE = "block_complexity_k.csv"


def read_k(out: Path) -> pd.DataFrame:
    """Per-building k written by 02_block_complexity_k.py (site, uID, label, block, k)."""
    path = Path(out) / K_FILE
    if not path.exists():
        raise ConfigError(f"{path} not found; run appendix/S09_block_complexity_k/02_block_complexity_k.py first")
    return pd.read_csv(path)


def is_favela(values) -> np.ndarray:
    """1 for favela, 0 for formal settlement (label strings or 1/0 codes)."""
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values):
        return (values.to_numpy(dtype=float) == 1).astype(int)
    return label_to_int(values)


def _read_columns(path: Path, wanted: list[str], required: list[str]) -> pd.DataFrame:
    """Attribute table of a GeoPackage (no geometry), restricted to the columns present."""
    fields = set(pyogrio.read_info(path)["fields"])
    missing = [c for c in required if c not in fields]
    if missing:
        raise ConfigError(f"{path} lacks the columns {missing[:4]}{' ...' if len(missing) > 4 else ''}")
    return pyogrio.read_dataframe(path, columns=[c for c in wanted if c in fields], read_geometry=False)


def site_table(cfg, site: str, k: pd.DataFrame | None = None) -> pd.DataFrame:
    """Buildings of one site: uID, label, y, 25 raw indicators, 25 SHAP columns (and k)."""
    fpath = cfg.require_product("features_gpkg", site=site)
    tab = _read_columns(fpath, ["uID", "label", *RAW], ["uID", "label", *RAW])
    if not tab["uID"].is_unique:
        raise ConfigError(f"{site}: uID is not unique in {fpath}")

    spath = cfg.require_product("shap_gpkg", site=site)
    sh = _read_columns(spath, ["uID", "label", *SHAP], SHAP)
    if "uID" in sh.columns:
        sh = sh.set_index("uID")
        if not sh.index.is_unique:
            raise ConfigError(f"{site}: uID is not unique in {spath}")
        sh = sh.reindex(tab["uID"].to_numpy())
    elif len(sh) != len(tab):
        raise ConfigError(f"{site}: {spath} and {fpath} hold different buildings")
    if "label" in sh.columns:
        known = sh["label"].notna().to_numpy()
        if (is_favela(sh["label"])[known] != is_favela(tab["label"])[known]).any():
            raise ConfigError(f"{site}: labels differ between {fpath.name} and {spath.name}")

    tab = pd.concat([tab, pd.DataFrame(sh[SHAP].to_numpy(float), columns=SHAP, index=tab.index)], axis=1)
    tab["y"] = is_favela(tab["label"])
    tab["site"] = site
    if k is not None:
        ks = k.loc[k["site"] == site].set_index("uID")["k"]
        tab["k"] = ks.reindex(tab["uID"].to_numpy()).to_numpy()
    return tab


def valid_k(tab: pd.DataFrame) -> pd.Series:
    """Distribution subset: valid k (>= 1), complete Tree-SHAP attributions, label present."""
    return tab["k"].notna() & (tab["k"] >= 1) & tab[SHAP].notna().all(axis=1) & tab["label"].notna()


def evaluation_subset(tab: pd.DataFrame) -> pd.Series:
    """Classification subset: ``valid_k`` and all 25 raw indicators present."""
    return valid_k(tab) & tab[RAW].notna().all(axis=1)


def tex_number(x, decimals: int | None = 3) -> str:
    """Decimal as typeset in the SI: 0.744 -> 0.744."""
    text = f"{x:.{decimals}f}" if decimals is not None else str(x)
    return text


def tex_int(n) -> str:
    """Integer with the SI thousands separator: 166874 -> 166{,}874."""
    return f"{int(n):,}".replace(",", "{,}")
