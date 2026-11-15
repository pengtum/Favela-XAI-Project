"""Helpers shared by the scripts of appendix/S12_spatial_cv (imported; not a step of its own).

Manuscript items: SI Section S12 'Spatial cross-validation' (SI Tables S10-S16,
SI Figs S19-S21), Extended Data Figs 3-4 and Main Methods 'Validation under
spatial separation'.

The fold makers, buffers, matched controls, the two scoring criteria, the site
bootstrap and the validation-pipeline classifier are in the shared package
(``favela_xai.spatial_cv``, ``favela_xai.model``, ``favela_xai.stats``); this
module connects them to the files of this folder:

* ``setup``: configuration, cross-validation settings and the output folders
  (all under outputs/appendix/S12_spatial_cv/);
* ``load_site`` / ``load_main_shap`` / ``load_groups``: per-site tables written
  by steps 01 and 02;
* ``make_design``: the splits of every named design (square blocks in a grid
  placement, hexagons, k-means units, five k-means folds, ten folds, random
  references, held-out favela component), used by the accuracy runs (step 03)
  and by the attribution runs (step 10) so that both see identical folds;
* ``loso_fold``: one fold of the pooled leave-site-out design;
* ``read_folds`` / ``read_predictions``: the fold record and the per-building
  out-of-fold predictions (step 04);
* ``score_cell``: the scoring procedure of SI S12 'Scoring' applied to the fold
  rows of one design cell, under both criteria and under the first criterion alone;
* number formats and the LaTeX table wrapper used by the SI tables.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split

from favela_xai import spatial_cv as scv
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER
from favela_xai.stats import bootstrap_mean_ci

FEATURES = list(MODEL_FEATURE_ORDER)           # column order of every classifier
SHAP_COLS = ["shap_" + f for f in FEATURES]
METRICS = ("accuracy", "balanced_accuracy", "auc", "f1_favela")


# -----------------------------------------------------------------------------
# Configuration and folders
# -----------------------------------------------------------------------------
def setup(script_file: str):
    """Return (cfg, cross-validation settings, folder namespace) for a script of this folder.

    All folders are created on first use under outputs/appendix/S12_spatial_cv/.
    """
    cfg = load_config()
    s = scv.CVSettings.from_config(cfg)
    out = cfg.output_dir(script_file)
    p = SimpleNamespace(
        out=out,
        site_tables=out / "site_tables",          # 01: per-site building tables and main-model SHAP
        groups=out / "component_groups",          # 02: held-out favela component groups
        fold_shards=out / "fold_shards",          # 03: fold rows, one file per site x job
        pred_shards=out / "prediction_shards",    # 03: per-building out-of-fold predictions
        shap_folds=out / "shap_fold_sums",        # 10: per-fold attribution sums, one file per site x run
        shap_buildings=out / "shap_buildings",    # 10: per-building out-of-fold SHAP (direction survival)
        tables=out / "tables",                    # LaTeX files of SI Tables S10-S16
        figures=out / "figures",                  # PDF files of SI Figs S19-S21 and ED Figs 3-4
        logs=out / "logs",
    )
    for v in vars(p).values():
        v.mkdir(parents=True, exist_ok=True)
    return cfg, s, p


def section(cfg) -> dict:
    """Settings of this folder (section appendix_S12_spatial_cv of config.yaml)."""
    return cfg.section("appendix_S12_spatial_cv")


def log_to(path: Path):
    """A print function that also writes every line to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*args):
        line = " ".join(str(a) for a in args)
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log


def require(path: Path, step: str) -> Path:
    """Stop with a clear message when the output of an earlier step is missing."""
    if not Path(path).exists():
        sys.exit(f"missing input {path}\n(run {step} of appendix/S12_spatial_cv first; see README.md)")
    return Path(path)


# -----------------------------------------------------------------------------
# Per-site data
# -----------------------------------------------------------------------------
@dataclass
class SiteData:
    site: str
    table: pd.DataFrame      # uID, x, y, is_favela, signed_dist and the 25 indicators (row order of main/01_features)
    X: np.ndarray            # n x 25 indicator matrix (MODEL_FEATURE_ORDER), missing values as NaN
    y: np.ndarray            # 1 = favela, 0 = formal settlement
    xy: np.ndarray           # n x 2 footprint centroids (EPSG:31983)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def n_favela(self) -> int:
        return int(self.y.sum())


def load_site(p, site: str) -> SiteData:
    """Per-site table written by 01_build_site_tables.py."""
    t = pd.read_parquet(require(p.site_tables / f"{site}.parquet", "01_build_site_tables.py"))
    return SiteData(site=site, table=t, X=t[FEATURES].to_numpy(float), y=t["is_favela"].to_numpy(int),
                    xy=t[["x", "y"]].to_numpy(float))


def load_main_shap(p, site: str) -> np.ndarray:
    """Tree SHAP of the main per-site model (n x 25, positive toward favela), in site-table row order."""
    t = pd.read_parquet(require(p.site_tables / f"{site}_main_shap.parquet", "01_build_site_tables.py"))
    return t[SHAP_COLS].to_numpy(float)


def load_groups(p, site: str) -> pd.DataFrame:
    """Held-out favela component of every building (02_component_groups.py), site-table row order."""
    return pd.read_parquet(require(p.groups / f"{site}_groups.parquet", "02_component_groups.py"))


def favela_totals(cfg, p) -> pd.Series:
    """Favela buildings per site (denominator of the second scoring criterion)."""
    return pd.Series({site: float(load_site(p, site).n_favela) for site in cfg.sites})


# -----------------------------------------------------------------------------
# Designs (SI S12 'Designs'). Names follow the fold record:
#   sq<L>            square blocks of L m, five folds (grid placement 0, 1, 2)
#   sq<L>_K10        square blocks of L m, ten folds (first placement)
#   hex<L>           hexagonal cells with the area of an L m square
#   kmm<L>           k-means units covering on average an L m square
#   km5              five k-means clusters used directly as the folds
#   random_80_20     random 80/20 split of this pipeline (unstratified, seed 432)
#   random_skf5      random stratified five-fold cross-validation (seed 432 unless given)
#   component        held-out favela component (one group per fold)
#   random_80_20_stratified   the stratified split drawn in SI Fig. S21
# -----------------------------------------------------------------------------
def make_design(name: str, d: SiteData, s: scv.CVSettings, cfg, p=None, placement: int = 0, seed: int | None = None):
    """Return (splits, meta) of a named design for one site; splits is a list of (fold, train, test)."""
    site, y, xy = d.site, d.y, d.xy
    if name.startswith("sq"):
        size, _, k = name[2:].partition("_K")
        return scv.design_square(site, y, xy, float(size), s, n_folds=int(k or s.n_folds), placement=placement)
    if name.startswith("hex"):
        return scv.design_hex(site, y, xy, float(name[3:]), s, n_folds=s.n_folds)
    if name.startswith("kmm"):
        return scv.design_kmeans_units(site, y, xy, float(name[3:]), s, n_folds=s.n_folds)
    if name == "km5":
        return scv.design_kmeans_folds(site, y, xy, s, n_folds=int(cfg.get("spatial_cv.kmeans_fold_clusters")))
    if name == "random_80_20":
        meta = dict(site=site, design=name, block_rule="random", L=np.nan, K=1, n_units=np.nan)
        return scv.splits_random_8020(d.n, s), meta
    if name == "random_skf5":
        meta = dict(site=site, design=name, block_rule="random", L=np.nan, K=s.n_folds, n_units=np.nan,
                    shuffle_seed=s.skf_seed if seed is None else seed)
        return scv.splits_five_fold(d.X, y, s, n_folds=s.n_folds, seed=seed), meta
    if name == "random_80_20_stratified":
        r = section(cfg)["component_figure_reference"]
        tr, te = train_test_split(np.arange(d.n), test_size=float(r["test_size"]),
                                  stratify=y if r["stratify"] else None, random_state=int(r["random_state"]))
        meta = dict(site=site, design=name, block_rule="random", L=np.nan, K=1, n_units=np.nan)
        return [(0, tr, te)], meta
    if name == "component":
        g = load_groups(p, site)
        assert (g["is_favela"].to_numpy() == y).all(), site
        splits, codes, members = component_splits(g)
        meta = dict(site=site, design=name, block_rule="component_group", L=np.nan, K=len(codes),
                    n_units=len(codes))
        return splits, dict(meta, _groups=codes, _members=members)
    raise ValueError(f"unknown design '{name}'")


def component_splits(groups: pd.DataFrame):
    """Leave-one-group-out folds; fold index = position of the group code in sorted order.

    Returns (splits, sorted group codes, {group code: '+'-joined member polygon codes}).
    A site with a single group has no split (it cannot be tested)."""
    g = groups["group"].to_numpy()
    codes = sorted(np.unique(g).tolist())
    members = groups.groupby("group")["component"].apply(lambda v: "+".join(map(str, sorted(set(v))))).to_dict()
    if len(codes) < 2:
        return [], codes, members
    fold_of = np.searchsorted(codes, g)
    return scv.splits_from_labels(fold_of, len(codes)), codes, members


# -----------------------------------------------------------------------------
# Pooled leave-site-out (SI S12 'Designs': one classifier trained on nine sites
# and tested on the tenth; no buffer)
# -----------------------------------------------------------------------------
def pooled_arrays(p, sites):
    """Stack the sites (indicators aligned by name). Returns X, y, xy, site of each row, row offsets."""
    Xs, ys, xys, sid, offsets = [], [], [], [], {}
    start = 0
    for site in sites:
        d = load_site(p, site)
        Xs.append(d.X)
        ys.append(d.y)
        xys.append(d.xy)
        sid.append(np.full(d.n, site, dtype=object))
        offsets[site] = start
        start += d.n
    return np.vstack(Xs), np.concatenate(ys), np.vstack(xys), np.concatenate(sid), offsets


def loso_fold(held: str, pooled, sites, cfg, s, n_jobs=None):
    """Fit the validation-pipeline model on all sites but ``held`` and test on ``held``.

    Returns (fold row, per-building predictions with within-site row ids). The
    row also records the distance from the held-out site to the nearest training
    building and the nearest training site (SI Table S10 caption, SI Table S15).
    """
    X, y, xy, sid, offsets = pooled
    te = np.where(sid == held)[0]
    tr = np.where(sid != held)[0]
    meta = dict(site=held, design="loso_pooled", block_rule="site", L=np.nan, K=len(sites), n_units=len(sites),
                repeat=0, fold=sites.index(held), buffer=np.nan, control=False)
    row, pred = scv.fit_eval(X, y, xy, tr, te, meta, 0, cfg, s, n_jobs, want_pred=True)
    d = scv.nn_distance(xy, tr, te)
    row["d_test_train_min"] = float(np.min(d)) if len(d) else np.nan
    nearest = {other: float(np.min(cKDTree(xy[sid == other]).query(xy[te], k=1)[0])) for other in sites if other != held}
    row["nearest_train_site"] = min(nearest, key=nearest.get)
    row["nearest_train_site_dist"] = nearest[row["nearest_train_site"]]
    if pred is not None:
        pred["row_id"] = (pred["row_id"].to_numpy() - offsets[held]).astype(np.int32)
        pred["buffer"] = np.float32(np.nan)
    return row, pred


# -----------------------------------------------------------------------------
# Fold record and predictions (step 04)
# -----------------------------------------------------------------------------
def read_folds(cfg) -> pd.DataFrame:
    """All fold rows of the accuracy runs (04_collect_folds.py; products.spatial_cv_folds)."""
    return pd.read_parquet(require(cfg.product("spatial_cv_folds"), "04_collect_folds.py"))


def read_predictions(p, columns=None) -> pd.DataFrame:
    """Per-building out-of-fold predictions kept by step 03 (04_collect_folds.py)."""
    return pd.read_parquet(require(p.out / "predictions_oof.parquet", "04_collect_folds.py"), columns=columns)


# -----------------------------------------------------------------------------
# Scoring (Main Methods 'Validation under spatial separation'; SI S12 'Scoring')
# -----------------------------------------------------------------------------
def score_cell(x: pd.DataFrame, tot_fav: pd.Series, s, ctl: pd.DataFrame | None = None, unit=("site", "repeat"),
               random_split: bool = False, metrics=METRICS) -> dict:
    """Scoring of the fold rows of one design cell.

    Returns a dict with the summary under both criteria ('both'), under the first
    criterion alone ('first'), the matched controls ('ctl_both', 'ctl_first'),
    the kept folds, the coverage table and fold/test-building counts. The random
    80/20 split tests 20% of each site by design, so the second criterion is not
    applied to it (SI S12 'Scoring'). Matched controls are scored on the site x
    placement units scored for the blocked design, plus the first criterion on
    the control fold itself.
    """
    x = x.copy()
    unit = list(unit)
    if "repeat" in unit and "repeat" not in x:
        x["repeat"] = 0
    kept, cov = scv.apply_rules(x, tot_fav, s, unit=unit)
    first = x[scv.rule1_mask(x, s)]
    if random_split:
        kept = first
        cov = cov.assign(scored=True)
    out = dict(kept=kept, first_kept=first, cov=cov)
    out["both"], out["both_site_means"] = scv.summarise(kept, s, metrics)
    out["first"], _ = scv.summarise(first, s, metrics)
    if ctl is not None and len(ctl):
        c = ctl.copy()
        if "repeat" in unit and "repeat" not in c:
            c["repeat"] = 0
        out["ctl_both"], _ = scv.summarise(scv.apply_control(c, cov, s, unit=unit), s, metrics)
        out["ctl_first"], _ = scv.summarise(c[scv.rule1_mask(c, s)], s, metrics)
    fitted = x.status.isin(scv.SCORED_STATUS)
    r1 = scv.rule1_mask(x, s)
    tot_test = x.n_test.sum()
    out["counts"] = dict(n_fold_rows=len(x), n_fitted=int(fitted.sum()), n_not_fitted=int((~fitted).sum()),
                         n_excluded_first=int((fitted & ~r1).sum()), n_used_first=int(r1.sum()),
                         n_used_final=len(kept), test_buildings=int(tot_test),
                         share_test_scored=float(kept.n_test.sum() / tot_test) if tot_test else np.nan)
    return out


def boot_mean(values, s) -> tuple[float, float, float, int]:
    """Mean over sites and site-bootstrap 95% interval (2,000 resamples of the sites, default_rng(42))."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    lo, hi = bootstrap_mean_ci(v, s.bootstrap_resamples, s.bootstrap_seed)
    return float(v.mean()), lo, hi, len(v)


# -----------------------------------------------------------------------------
# Names, number formats and LaTeX tables
# -----------------------------------------------------------------------------
def site_names(cfg, sites) -> str:
    return ", ".join(cfg.site_name(x) for x in sites)


def tex_dot(v: float, nd: int = 3) -> str:
    """Decimal as printed in the SI, e.g. 0.6452 -> 0.645 ('--' when missing)."""
    if v is None or not np.isfinite(v):
        return "--"
    txt = f"{v:.{nd}f}"
    sign = ""
    if txt.startswith("-"):
        sign, txt = "$-$", txt[1:]
    return sign + txt


def f3(v) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.3f}"


def f2(v) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.2f}"


def ci3(r: dict, m: str, pre: str = "") -> str:
    """'0.645 (0.584--0.714)' from a summary dict (plain decimals, as in Tables S11-S12)."""
    v = r.get(f"{pre}{m}", np.nan)
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:.3f} ({r[f'{pre}{m}_ci_lo']:.3f}--{r[f'{pre}{m}_ci_hi']:.3f})"


def tex_thousands(v: float, nd: int = 1) -> str:
    """1234.5 -> 1{,}234.5 ('--' when missing)."""
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:,.{nd}f}".replace(",", "{,}")


def write_tex_table(path: Path, generator: str, caption: str, label: str, body_lines: list[str], cfg,
                    size: str = "\\scriptsize", resize: bool = True, preamble: list[str] | None = None,
                    tabular: str = "", requires: str = "booktabs,graphicx") -> Path:
    """Write one SI table as a LaTeX table[p] environment (booktabs).

    ``caption`` is the full caption text (without \\caption{}), ``tabular`` the
    column specification, ``body_lines`` the lines between \\begin{tabular} and
    \\end{tabular} (header rules included).
    """
    lines = [f"% generated by appendix/S12_spatial_cv/{generator}; requires \\usepackage{{{requires}}}", "",
             "\\begin{table}[p]"]
    lines += ["\\centering", size]
    lines += preamble or []
    lines += [f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{tabular}}}")
    lines += body_lines
    lines.append("\\end{tabular}" + ("}" if resize else ""))
    lines += ["\\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
