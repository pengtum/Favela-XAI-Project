"""Sensitivity of the attribution rankings and boundary results to the adjacency ratio.

Manuscript items
* SI Section S15 and SI Table S19: every site analysed in three specifications,
  all fitted with the out-of-fold protocol of the boundary error analysis
  (Methods 4.7; the folds, classifier and imputation of
  main/02_classification/03_oof_predictions.py), so that the prediction and the
  attributions of every building come from a model not trained on it:
    reference              the 25 indicators as analysed (products.features_gpkg),
                           the adjacency ratio on the queen contiguity graph of the
                           tessellation cells;
    footprint contiguity   the adjacency ratio recomputed on the queen contiguity
                           graph of the building footprints, the alternative
                           setting of the configuration
                           (indicators.adjacency_contiguity: buildings); the other
                           24 indicators as analysed;
    omitted                the other 24 indicators.
* Methods 4.3 and SI S15: mean per-site Spearman rho with the reference 0.91
  (footprint contiguity) and 0.95 (omitted), 0.97 and 0.96 for the pooled
  ranking; sign-change distance x0 of the summed spatial-distribution
  attributions +4.8, +5.1 and +4.4 m; misclassified within 6 m / at 90 m or more
  26.8% / 3.7% (reference, as in Fig. 2c) and 26.5% / 3.5% (footprint contiguity).

Definitions (config.yaml)
* Folds and classifier: sections 'oof' and 'model_cv', through
  favela_xai.model.fit_predict_cv. The folds depend on the labels only, so they
  are the same in the three specifications. Attributions: exact Tree SHAP of
  each fold's model on its held-out buildings (log-odds, positive toward favela;
  favela_xai.attribution.tree_shap).
* Rankings ('importance.population'): each indicator's share of the mean |SHAP|
  over the favela buildings of a site, as in Extended Data Fig. 2
  (favela_xai.attribution.favela_importance). Agreement with the reference over
  the indicators the two specifications share (25; 24 for the omitted
  specification, with the reference shares renormalised over the 24): Spearman's
  rho per site; rho of the pooled ranking (mean of the site shares); sites that
  keep at least two of their three leading indicators
  ('spatial_cv.ranking.top_k'); sites that keep their leading indicator when the
  indicator is omitted.
* Boundary transition ('boundary_profiles'): cross-site profile of the summed
  attributions of the spatial-distribution indicators (the Spatial distribution
  dimension of the registry: four indicators, three when the adjacency ratio is
  omitted) and of all indicators against the signed distance to the official
  2013 favela boundary (products.signed_distance), with the bins, smoothing and
  transition metrics of Fig. 4 and SI Table S5 (favela_xai.boundary): dS, x0 and
  w, and the number of sites whose own profile has dS > 0 among the sites where
  dS can be evaluated.
* Boundary errors ('boundary_errors.band_edges_m', 'near_far_bands'): share of
  buildings misclassified out of fold (favela probability >= 0.5 differs from
  the label) within 6 m of the boundary and at 90 m or more, pooled over the
  sites, and the number of sites with the higher rate near the line; given for
  the two specifications of the indicator (reference and footprint contiguity),
  as in SI Table S19.

Inputs (outputs/main/01_features/ and outputs/main/02_classification/)
  <site>_buildings_features.gpkg     products.features_gpkg: uID, label, 25 indicators
  <site>_buildings_tessellated.gpkg, <site>_tessellation.gpkg
                                     footprints and cells written by 02_tessellation.py,
                                     for the footprint-contiguity ratio
  signed_distance.parquet            products.signed_distance
  oof_predictions.parquet            products.oof_predictions (optional; when present, the
                                     reference specification is checked against it)

Outputs (outputs/appendix/S15_adjacency_sensitivity/)
  adjacency_footprint_contiguity/<site>.parquet
                                     uID, adjacency ratio on footprint contiguity (computed
                                     once, then reused; --recompute_adjacency recomputes it)
  importance_shares.csv              site x specification x indicator: mean |SHAP| over the
                                     favela buildings and share
  rankings_per_site.csv              site x specification: rho with the reference, leading
                                     indicators, overlap of the three leading indicators
  rankings_pooled.csv                indicator: pooled shares of each specification
  boundary_profiles.csv              specification x group x 2.5 m bin: cross-site mean,
                                     s.d. and number of sites
  boundary_transition.csv            specification x group: dS, x0, w, sites with dS > 0
  boundary_transition_by_site.csv    specification x group x site: dS of the site profile
  boundary_errors.csv                reference and footprint contiguity: buildings,
                                     misclassified buildings and rate within 6 m and at
                                     90 m or more; sites with the higher rate near the line
  boundary_errors_by_site.csv        the same counts per site
  table_S19_adjacency_sensitivity.tex   SI Table S19
  in_text_numbers.txt                the values quoted in SI Section S15 and Methods 4.3

Run from the repository root, after main/01_features (steps 01-05):
  python appendix/S15_adjacency_sensitivity/01_adjacency_sensitivity.py [--sites ...] [--n_jobs N]
"""
from __future__ import annotations

import argparse
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from favela_xai.attribution import favela_importance, shap_frame, tree_shap
from favela_xai.boundary import (band_index, binned_mean, cross_site_profile, load_signed_distance, profile_edges,
                                 transition_metrics)
from favela_xai.config import Config, load_config
from favela_xai.features import MODEL_FEATURE_ORDER, compute_indicators, features_by_dimension
from favela_xai.io import read_site_features, write_table, write_text
from favela_xai.labels import label_to_int
from favela_xai.model import fit_predict_cv

INDICATOR = "adjacency"                        # the adjacency ratio (SI S3 item 19)
REFERENCE, FOOTPRINT, OMITTED = "reference", "footprint_contiguity", "omitted"
SPECIFICATIONS = [REFERENCE, FOOTPRINT, OMITTED]
COLUMN_TITLES = {REFERENCE: "Reference", FOOTPRINT: "Footprint contiguity", OMITTED: "Omitted"}
# Error rows of SI Table S19: the two specifications of the indicator.
ERROR_SPECIFICATIONS = [REFERENCE, FOOTPRINT]
# Boundary-profile groups: summed attributions of one dimension of the registry (None = all indicators).
GROUPS = {"spatial distribution": "Spatial distribution", "all indicators": None}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def parse_args(cfg) -> tuple[list[str], argparse.Namespace]:
    """Sites (default: every site in config.yaml, in config order) and options."""
    parser = argparse.ArgumentParser(
        description="Sensitivity of the attribution rankings and boundary results to the adjacency ratio (SI S15).")
    parser.add_argument("--sites", nargs="+", help="site keys to process (default: all)")
    parser.add_argument("--n_jobs", type=int, default=None, help="XGBoost threads (default: xgboost default)")
    parser.add_argument("--recompute_adjacency", action="store_true",
                        help="recompute the footprint-contiguity ratio even if it is stored")
    args = parser.parse_args()
    sites = args.sites or cfg.sites
    unknown = sorted(set(sites) - set(cfg.all_sites))
    if unknown:
        raise SystemExit(f"unknown site key(s): {unknown}; valid keys: {cfg.all_sites}")
    order = {s: i for i, s in enumerate(cfg.all_sites)}
    return sorted(dict.fromkeys(sites), key=order.get), args


def specification_features(spec: str) -> list[str]:
    """Indicator columns of a specification, in MODEL_FEATURE_ORDER."""
    return [f for f in MODEL_FEATURE_ORDER if not (spec == OMITTED and f == INDICATOR)]


def with_setting(cfg: Config, section: str, key: str, value) -> Config:
    """The configuration with one setting replaced (config.yaml itself is not changed)."""
    data = yaml.safe_load(cfg.source.read_text(encoding="utf-8"))
    data[section][key] = value
    return Config(data, root=cfg.root, source=cfg.source)


def read_tessellation(cfg, site: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Buildings and cells of main/01_features step 02, both indexed by uID and in the same order."""
    folder = cfg.output_dir("main/01_features", create=False)
    b_path, c_path = folder / f"{site}_buildings_tessellated.gpkg", folder / f"{site}_tessellation.gpkg"
    for p in (b_path, c_path):
        if not p.exists():
            raise SystemExit(f"{p} not found; run main/01_features/02_tessellation.py first")
    buildings = gpd.read_file(b_path)
    cells = gpd.read_file(c_path)
    buildings.index = pd.Index(buildings["uID"].astype(int).to_numpy())
    cells.index = pd.Index(cells["uID"].astype(int).to_numpy())
    cells = cells.drop(columns="uID")
    if not buildings.index.is_unique or set(buildings.index) != set(cells.index):
        raise SystemExit(f"{site}: buildings and tessellation cells do not share the same uIDs")
    return buildings, cells.loc[buildings.index]


def footprint_adjacency(cfg, site: str, folder, recompute: bool = False) -> pd.Series:
    """Adjacency ratio of every building on the queen contiguity graph of the footprints (index uID).

    favela_xai.features.compute_indicators with indicators.adjacency_contiguity
    set to 'buildings', the other settings as in config.yaml; only the adjacency
    ratio is kept. Stored per site and reused.
    """
    path = folder / f"{site}.parquet"
    if path.exists() and not recompute:
        stored = pd.read_parquet(path)
        return pd.Series(stored["adjacency"].to_numpy(float), index=stored["uID"].to_numpy(), name=INDICATOR)
    buildings, cells = read_tessellation(cfg, site)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # momepy / libpysal notes on islands and deprecations
        indicators = compute_indicators(buildings, cells,
                                        with_setting(cfg, "indicators", "adjacency_contiguity", "buildings"))
    adj = indicators[INDICATOR]
    write_table(pd.DataFrame({"uID": adj.index.to_numpy(), "adjacency": adj.to_numpy(float)}), path)
    return pd.Series(adj.to_numpy(float), index=adj.index.to_numpy(), name=INDICATOR)


# -----------------------------------------------------------------------------
# Out-of-fold predictions and attributions (Methods 4.7)
# -----------------------------------------------------------------------------
def oof_folds(X: np.ndarray, y: np.ndarray, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds of the out-of-fold error analysis (config 'oof', as favela_xai.model.oof_predictions)."""
    o = cfg.section("oof")
    skf = StratifiedKFold(int(o["n_folds"]), shuffle=bool(o["shuffle"]), random_state=int(o["random_state"]))
    return list(skf.split(X, y))


def out_of_fold(X: np.ndarray, y: np.ndarray, folds, cfg, n_jobs: int | None = None):
    """Out-of-fold favela probability and Tree SHAP (positive toward favela) of every building."""
    negate = bool(cfg.get("model_cv.fit_formal_as_class_1", True))
    prob = np.full(len(y), np.nan)
    shap = np.full(X.shape, np.nan)
    for tr, te in folds:
        model, prob[te], X_te = fit_predict_cv(X[tr], y[tr], X[te], cfg, n_jobs)
        shap[te] = tree_shap(model, X_te, negate=negate)
    return prob, shap


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def leading(shares: pd.Series, k: int) -> list[str]:
    """The k indicators with the largest shares."""
    return list(shares.sort_values(ascending=False).index[:k])


def restricted(shares: pd.Series, features) -> pd.Series:
    """Shares over a subset of the indicators, renormalised to sum to one."""
    v = shares[list(features)]
    return v / v.sum()


def signed(v: float) -> str:
    return "--" if not np.isfinite(v) else ("$+$" if v >= 0 else "$-$") + f"{abs(v):.1f}"


def band_text(edges: list[float], j: int, tex: bool = False) -> str:
    """'within 6 m' (first band), 'at 90 m or more' (last band) or 'at 6-30 m' for distance band j."""
    unit = "\\,m" if tex else " m"
    if j == 0:
        return f"within {edges[1]:g}{unit}"
    if j == len(edges) - 1:
        return f"at {edges[j]:g}{unit} or more"
    return f"at {edges[j]:g}-{edges[j + 1]:g}{unit}"


def table_s19(rank: pd.DataFrame, pooled: dict, trans: pd.DataFrame, errors: pd.DataFrame, n_sites: int,
              top_k: int, near_label: str, far_label: str) -> str:
    """SI Table S19 in LaTeX (rows as printed in the SI)."""
    def rho_cell(spec):
        r = rank[rank.specification == spec].rho
        return f"{r.mean():.2f} ({r.min():.2f}--{r.max():.2f})"

    def keep_cell(spec):
        r = rank[rank.specification == spec]
        return f"{int((r.leading_overlap >= top_k - 1).sum())} of {n_sites}"

    def xw_cell(spec, group):
        r = trans.set_index(["specification", "group"]).loc[(spec, group)]
        return f"{signed(r.x0)} / {r.w:.1f}" if np.isfinite(r.w) else "--"

    def err_cell(spec):
        if spec not in set(errors.specification):
            return "--"
        r = errors.set_index("specification").loc[spec]
        return f"{100 * r.rate_near:.1f}\\% / {100 * r.rate_far:.1f}\\%"

    k_word, k1_word = WORDS.get(top_k, str(top_k)), WORDS.get(top_k - 1, str(top_k - 1))
    rows = [
        ("Rank correlation with the reference, per site: mean (range)", "--", rho_cell(FOOTPRINT),
         rho_cell(OMITTED)),
        ("Rank correlation with the reference, pooled ranking", "--", f"{pooled[FOOTPRINT]:.2f}",
         f"{pooled[OMITTED]:.2f}"),
        (f"Sites keeping at least {k1_word} of their {k_word} leading indicators", "--", keep_cell(FOOTPRINT),
         keep_cell(OMITTED)),
        ("Sign-change distance $x_0$ / width $w$ (m), spatial-distribution indicators",
         *[xw_cell(s, "spatial distribution") for s in SPECIFICATIONS]),
        ("Sign-change distance $x_0$ / width $w$ (m), all indicators",
         *[xw_cell(s, "all indicators") for s in SPECIFICATIONS]),
        (f"Misclassified {near_label} / {far_label}", *[err_cell(s) for s in SPECIFICATIONS]),
    ]
    lines = [
        "% Supplementary Table S19. Generated by appendix/S15_adjacency_sensitivity/01_adjacency_sensitivity.py",
        "% Requires the booktabs and array packages.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Sensitivity of the attribution rankings and boundary results to the",
        r"adjacency ratio.} All three specifications use the out-of-fold protocol of the boundary",
        r"error analysis (stratified five-fold cross-validation within each site; Methods), so that",
        r"the prediction and the attributions of every building come from a model not trained on it.",
        r"\emph{Reference}: the 25 indicators as analysed; its error rates are those of main-text",
        r"Figure~2c and Table~\ref{tab:exposure_bands}. \emph{Footprint contiguity}: the adjacency",
        r"ratio recomputed on the queen contiguity graph of the building footprints instead of the",
        r"tessellation cells. \emph{Omitted}: the other 24 indicators. Rank correlation: Spearman's",
        r"$\rho$ between the rankings of the indicators by their share of the mean absolute",
        r"attribution over a site's favela buildings (as in Extended Data Fig.~2), over the",
        r"indicators that the specification shares with the reference; per site, mean over the ten",
        r"sites with the range in parentheses; pooled, rankings by the mean of the ten site shares.",
        r"$x_0$ and $w$: sign-change distance ($x_0>0$ outside the official line) and width of the",
        r"cross-site profile of the summed attributions of the four spatial-distribution indicators",
        r"(three when the adjacency ratio is omitted) or of all indicators, computed as for",
        r"Table~\ref{tab:spillover_metrics}, which gives the indicators one by one. Misclassified:",
        r"pooled out-of-fold rates within 6\,m of the 2013 boundary and at 90\,m or more, given for",
        r"the two specifications of the indicator.}",
        r"\label{tab:adjacency_sensitivity}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.41\textwidth}ccc@{}}",
        r"\toprule",
        " & " + " & ".join(COLUMN_TITLES[s] for s in SPECIFICATIONS) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    sites, args = parse_args(cfg)
    out = cfg.output_dir(__file__)
    if cfg.get("indicators.adjacency_contiguity") != "tessellation":
        raise ValueError("SI S15 compares the indicators as analysed (indicators.adjacency_contiguity: "
                         "tessellation) with the ratio recomputed on footprint contiguity; set it to tessellation")
    if cfg.get("importance.population") != "favela_buildings":
        raise ValueError("SI S15 ranks the indicators by their share over the favela buildings "
                         "(importance.population: favela_buildings)")
    o, mc = cfg.section("oof"), cfg.section("model_cv")
    print(f"out-of-fold protocol of the boundary error analysis: stratified {o['n_folds']}-fold CV "
          f"(seed {o['random_state']}); XGBoost {mc['n_estimators']} trees, seed {mc['random_state']} "
          "(config.yaml 'oof', 'model_cv')")

    top_k = int(cfg.get("spatial_cv.ranking.top_k"))
    spatial = set(features_by_dimension()[GROUPS["spatial distribution"]])
    edges, centres = profile_edges(cfg)
    min_n = int(cfg.get("boundary_profiles.min_buildings_per_site_bin"))
    be = cfg.section("boundary_errors")
    band_edges = [float(v) for v in be["band_edges_m"]]
    near_band, far_band = [int(v) for v in be["near_far_bands"]]

    signed_dist = load_signed_distance(cfg, sites)
    oof_path = cfg.product("oof_predictions")
    stored_oof = pd.read_parquet(oof_path, columns=["site", "uID", "prob"]) if oof_path.exists() else None
    adj_dir = out / "adjacency_footprint_contiguity"
    adj_dir.mkdir(parents=True, exist_ok=True)

    share_rows, error_rows, check_rows = [], [], []
    shares, profiles = {}, {(s, g): [] for s in SPECIFICATIONS for g in GROUPS}
    for site in sites:
        t0 = time.time()
        g = read_site_features(cfg, site, columns=["uID", "label"] + MODEL_FEATURE_ORDER)
        uid = g["uID"].to_numpy()
        y = label_to_int(g["label"])
        is_favela = y.astype(bool)
        sd = signed_dist[signed_dist["site"] == site].set_index("uID").reindex(uid)
        if sd["signed_dist"].isna().any() or not np.array_equal(sd["is_favela"].to_numpy(bool), is_favela):
            raise SystemExit(f"{site}: signed distance and feature table do not match; "
                             "re-run main/01_features/05_signed_distance.py")
        d = sd["signed_dist"].to_numpy(float)
        bands = band_index(np.abs(d), band_edges)

        adj = footprint_adjacency(cfg, site, adj_dir, args.recompute_adjacency)
        missing = np.setdiff1d(uid, adj.index.to_numpy())
        if len(missing):
            raise SystemExit(f"{site}: {len(missing)} buildings have no footprint-contiguity ratio; "
                             "run again with --recompute_adjacency")
        X = {REFERENCE: g[MODEL_FEATURE_ORDER].to_numpy(float)}
        X[FOOTPRINT] = X[REFERENCE].copy()
        X[FOOTPRINT][:, MODEL_FEATURE_ORDER.index(INDICATOR)] = adj.reindex(uid).to_numpy(float)
        X[OMITTED] = X[REFERENCE][:, [MODEL_FEATURE_ORDER.index(f) for f in specification_features(OMITTED)]]
        folds = oof_folds(X[REFERENCE], y, cfg)       # labels only: the same folds in every specification

        for spec in SPECIFICATIONS:
            feats = specification_features(spec)
            prob, shap = out_of_fold(X[spec], y, folds, cfg, args.n_jobs)
            imp = favela_importance(shap_frame(shap, features=feats), is_favela, features=feats)
            shares[(site, spec)] = imp["share"]
            share_rows += [dict(site=site, specification=spec, indicator=f, mean_abs_shap=float(r.mean_abs_shap),
                                share=float(r.share)) for f, r in imp.iterrows()]
            for group, dim in GROUPS.items():
                cols = [i for i, f in enumerate(feats) if dim is None or f in spatial]
                profiles[(spec, group)].append(binned_mean(d, shap[:, cols].sum(axis=1), edges, min_n))
            if spec in ERROR_SPECIFICATIONS:
                err = (prob >= 0.5).astype(int) != y
                near, far = bands == near_band, bands == far_band
                error_rows.append(dict(site=site, specification=spec, n_near=int(near.sum()),
                                       misclassified_near=int(err[near].sum()), n_far=int(far.sum()),
                                       misclassified_far=int(err[far].sum())))
            if spec == REFERENCE and stored_oof is not None and site in set(stored_oof["site"]):
                ref = stored_oof[stored_oof["site"] == site].set_index("uID").reindex(uid)["prob"].to_numpy(float)
                check_rows.append(dict(site=site, n=len(uid), max_abs_difference=float(np.nanmax(np.abs(ref - prob))),
                                       class_differences=int(((ref >= 0.5) != (prob >= 0.5)).sum())))
        print(f"  {cfg.site_name(site):24s} {len(uid):>7,} buildings: three specifications fitted "
              f"({time.time() - t0:.0f} s)")
    n_sites = len(sites)

    # ---- rankings: agreement with the reference
    rank_rows, pooled_rho, pooled_cols = [], {}, {}
    for spec in (FOOTPRINT, OMITTED):
        common = specification_features(spec)
        for site in sites:
            a, b = restricted(shares[(site, REFERENCE)], common), restricted(shares[(site, spec)], common)
            la, lb = leading(a, top_k), leading(b, top_k)
            rank_rows.append(dict(site=site, specification=spec, n_indicators=len(common),
                                  rho=float(spearmanr(a, b[a.index]).statistic),
                                  leading_reference=";".join(la), leading_specification=";".join(lb),
                                  leading_overlap=len(set(la) & set(lb)), same_leading_indicator=la[0] == lb[0]))
        pa = pd.concat([restricted(shares[(s, REFERENCE)], common) for s in sites], axis=1).mean(axis=1)
        pb = pd.concat([restricted(shares[(s, spec)], common) for s in sites], axis=1).mean(axis=1)
        pooled_rho[spec] = float(spearmanr(pa, pb[pa.index]).statistic)
        pooled_cols[f"reference_over_{len(common)}"] = pa
        pooled_cols[spec] = pb
    rank = pd.DataFrame(rank_rows)
    write_table(rank, out / "rankings_per_site.csv")
    write_table(pd.DataFrame(share_rows), out / "importance_shares.csv")
    pooled = pd.DataFrame(pooled_cols).reindex(MODEL_FEATURE_ORDER).rename_axis("indicator").reset_index()
    write_table(pooled, out / "rankings_pooled.csv")

    # ---- boundary transition of the summed attributions
    trans_rows, prof_rows, site_ds_rows = [], [], []
    for spec in SPECIFICATIONS:
        for group, dim in GROUPS.items():
            per_site = np.vstack(profiles[(spec, group)])
            mu, sdv = cross_site_profile(per_site)
            n_with_bin = np.isfinite(per_site).sum(axis=0)
            m = transition_metrics(mu, centres, cfg)
            ds = np.array([transition_metrics(p, centres, cfg)["delta_s"] for p in per_site])
            n_ind = len([f for f in specification_features(spec) if dim is None or f in spatial])
            trans_rows.append(dict(specification=spec, group=group, n_indicators=n_ind, delta_s=m["delta_s"],
                                   x0=m["x0"], w=m["w"], resolved=m["resolved"],
                                   sites_delta_s_positive=int(np.sum(ds[np.isfinite(ds)] > 0)),
                                   sites_evaluable=int(np.isfinite(ds).sum())))
            prof_rows += [dict(specification=spec, group=group, bin_lo=edges[b], bin_hi=edges[b + 1],
                               bin_centre=centres[b], mean_shap=mu[b], sd_shap=sdv[b], n_sites=int(n_with_bin[b]))
                          for b in range(len(centres))]
            site_ds_rows += [dict(specification=spec, group=group, site=s, delta_s=v) for s, v in zip(sites, ds)]
    trans = pd.DataFrame(trans_rows)
    write_table(trans, out / "boundary_transition.csv")
    write_table(pd.DataFrame(prof_rows), out / "boundary_profiles.csv")
    write_table(pd.DataFrame(site_ds_rows), out / "boundary_transition_by_site.csv")

    # ---- boundary errors (reference and footprint contiguity)
    by_site = pd.DataFrame(error_rows)
    by_site["rate_near"] = by_site.misclassified_near / by_site.n_near.where(by_site.n_near > 0)
    by_site["rate_far"] = by_site.misclassified_far / by_site.n_far.where(by_site.n_far > 0)
    write_table(by_site, out / "boundary_errors_by_site.csv")
    err_rows = []
    for spec in ERROR_SPECIFICATIONS:
        e = by_site[by_site.specification == spec]
        ev = e[(e.n_near > 0) & (e.n_far > 0)]
        n_near, k_near = e.n_near.sum(), e.misclassified_near.sum()
        n_far, k_far = e.n_far.sum(), e.misclassified_far.sum()
        err_rows.append(dict(specification=spec, n_near=int(n_near), misclassified_near=int(k_near),
                             rate_near=k_near / n_near, n_far=int(n_far), misclassified_far=int(k_far),
                             rate_far=k_far / n_far, sites_near_above_far=int((ev.rate_near > ev.rate_far).sum()),
                             sites_evaluable=len(ev)))
    errors = pd.DataFrame(err_rows)
    write_table(errors, out / "boundary_errors.csv")

    # ---- SI Table S19 and the values of SI S15 / Methods 4.3
    write_text(table_s19(rank, pooled_rho, trans, errors, n_sites, top_k, band_text(band_edges, near_band, tex=True),
                         band_text(band_edges, far_band, tex=True)), out / "table_S19_adjacency_sensitivity.tex")
    T = trans.set_index(["specification", "group"])
    E = errors.set_index("specification")
    lines = []
    for spec in (FOOTPRINT, OMITTED):
        r = rank[rank.specification == spec]
        lines.append(f"{COLUMN_TITLES[spec].lower()}: per-site rho with the reference {r.rho.mean():.3f} "
                     f"({r.rho.min():.3f}-{r.rho.max():.3f}) over {int(r.n_indicators.iloc[0])} indicators; "
                     f"pooled rho {pooled_rho[spec]:.3f}; sites keeping at least {top_k - 1} of their {top_k} "
                     f"leading indicators {int((r.leading_overlap >= top_k - 1).sum())} of {n_sites}")
    r = rank[rank.specification == OMITTED]
    lines.append(f"omitted: sites keeping their leading indicator {int(r.same_leading_indicator.sum())} of {n_sites}")
    def num(v, fmt):
        return format(v, fmt) if np.isfinite(v) else "--"

    for group in GROUPS:
        lines.append(f"x0 / w (m), {group}: " + "; ".join(
            f"{COLUMN_TITLES[s].lower()} {num(T.loc[(s, group), 'x0'], '+.1f')} / {num(T.loc[(s, group), 'w'], '.1f')}"
            for s in SPECIFICATIONS))
    lines.append(f"widths {T.w.min():.1f}-{T.w.max():.1f} m; sites with dS > 0 among the evaluable sites: "
                 + "; ".join(f"{COLUMN_TITLES[s].lower()}, {g} {int(T.loc[(s, g), 'sites_delta_s_positive'])} of "
                             f"{int(T.loc[(s, g), 'sites_evaluable'])}" for s, g in T.index))
    for spec in ERROR_SPECIFICATIONS:
        v = E.loc[spec]
        lines.append(f"{COLUMN_TITLES[spec].lower()}: misclassified {band_text(band_edges, near_band)} "
                     f"{100 * v.rate_near:.1f}% ({int(v.misclassified_near):,} of {int(v.n_near):,}), "
                     f"{band_text(band_edges, far_band)} {100 * v.rate_far:.1f}% "
                     f"({int(v.misclassified_far):,} of {int(v.n_far):,}); higher near the line in "
                     f"{int(v.sites_near_above_far)} of {int(v.sites_evaluable)} sites")
    if check_rows:
        c = pd.DataFrame(check_rows)
        lines.append(f"reference against products.oof_predictions ({len(c)} sites, {int(c.n.sum()):,} buildings): "
                     f"largest probability difference {c.max_abs_difference.max():.2g}, "
                     f"{int(c.class_differences.sum())} buildings with a different predicted class")
    else:
        lines.append("reference against products.oof_predictions: not checked (run "
                     "main/02_classification/03_oof_predictions.py first)")
    lines.append("SI S15 / Table S19 / Methods 4.3: rho 0.91 (0.79-0.97) and 0.95 (0.88-0.99), pooled 0.97 and "
                 "0.96; at least two of three leading indicators kept in 10 of 10 sites in both, the leading "
                 "indicator among the other 24 in every site when the indicator is omitted; x0 / w spatial "
                 "distribution +4.8 / 16.7, "
                 "+5.1 / 17.7, +4.4 / 18.3, all indicators +4.2 / 14.4, +4.4 / 16.0, +4.0 / 15.3; dS > 0 in all "
                 "nine evaluable sites; misclassified 26.8% / 3.7% (reference) and 26.5% / 3.5% (footprint "
                 "contiguity), higher near the line in all ten sites in both")
    write_text("\n".join(lines) + "\n", out / "in_text_numbers.txt")
    print("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
