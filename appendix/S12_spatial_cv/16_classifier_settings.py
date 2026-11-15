"""Step 16: the site-level results under alternative classifier settings (SI Table S16, rows 'Classifier settings').

Manuscript items
* Methods 4.4 and SI S12 'Stability of the attributions': the site-level results of the
  main per-site models recomputed under four alternative classifier settings
  (config.yaml 'classifier_settings_sensitivity'; every variant is reported):
    V1  200 trees, maximum depth 6, learning rate 0.1, stratified 80/20 split with seed 42
    V2  maximum depth 4
    V3  maximum depth 8
    V4  learning rate 0.1 with 300 trees
  everything else as in the main analysis (model_main: 100 trees, maximum depth 6, learning
  rate 0.3, unstratified 80/20 split with seed 432).
* SI Table S16, rows 'Classifier settings' (rho, rho minus the split-refit null, top three):
  appended by 11_ranking_stability.py when the output of this step exists
  (products.classifier_settings_by_site).
* tables/table_classifier_settings.tex: held-out accuracy, rho of the per-site and of the
  cross-site rankings, leading indicator and top three kept, and x0 of mean interbuilding
  distance and adjacency, for the main analysis and the four variants.

Procedure (each variant and site; the code path of main/02-04)
* Fit: favela_xai.model.fit_main_model with the configuration in which the model_main keys
  of the variant are replaced (as main/02_classification/01_train_models.py): split drawn
  on the buildings ordered by uID, training rows passed in split order, formal settlement
  fitted as class 1 (model_main.fit_formal_as_class_1), intercept left to xgboost,
  training-median imputation.
* Held-out accuracy on the variant's own 20% test split (SiteModel.test_metrics; SI
  Table S3 columns).
* Tree SHAP of every building (favela_xai.attribution.tree_shap: path-dependent, log-odds,
  negated so that positive values point toward favela; shap.model_output).
* Site importance: mean |SHAP| over the site's favela buildings
  (favela_xai.attribution.favela_importance; Extended Data Fig. 2; importance.population),
  ranked as in main/03_site_signatures/01_importance_tables.py.
* Agreement with the main analysis: Spearman's rho of the 25 per-site values; leading
  indicator kept; at least two, and all three, of the three leading indicators kept
  (spatial_cv.ranking.top_k).
* Cross-site ranking: mean over the sites of the per-site values (the cross-site ranking of
  main/03_site_signatures, the common order of Extended Data Fig. 2), and Spearman's rho
  with the main analysis. The rho of the mean of the site shares (the pooled ranking of SI
  S15) is also written.
* Boundary profiles (Methods 4.8; as main/04_boundary_profiles/01_boundary_profiles.py):
  per-site mean SHAP in 2.5 m bins of signed distance to the official 2013 favela boundary
  (products.signed_distance, joined on uID), cross-site mean over the sites with the bin,
  and |dS|, x0 and w of all 25 indicators (favela_xai.boundary.transition_metrics); x0 of
  classifier_settings_sensitivity.boundary_indicators is reported.

Required settings (checked first, for the configuration and every variant, as in
  main/02_classification/01_train_models.py and main/03_site_signatures/01_importance_tables.py):
  model_main.imputation median, model_main.positive_class favela, shap.explain all_buildings,
  importance.population favela_buildings.

Reference check (run first; no variant is fitted unless it passes; exit code 1 otherwise)
  The main analysis is recomputed with model_main unchanged and compared with the stored
  outputs: held-out set (products.split_csv) and test accuracy (products.site_metrics)
  exactly; Tree SHAP within classifier_settings_sensitivity.reference_shap_tolerance
  (products.shap_gpkg); per-site ranks (products.importance_favela) and cross-site ranks
  (main/03_site_signatures/crosssite_importance.csv) identical; |dS|, x0 and w at the
  precision of SI Table S5 (products.transition_metrics) and the x0 cells of the reported
  indicators in main/04_boundary_profiles/table_S5_transition_metrics.tex.

Inputs   products.features_gpkg, products.signed_distance (main/01_features);
         products.split_csv, products.site_metrics, products.shap_gpkg (main/02_classification);
         products.importance_favela, crosssite_importance.csv (main/03_site_signatures);
         products.transition_metrics, table_S5_transition_metrics.tex (main/04_boundary_profiles)
Outputs  outputs/appendix/S12_spatial_cv/classifier_settings/
           reference_check.csv              every check of the recomputed main analysis
           settings_by_site.csv             variant x site: accuracy, rho, leading indicators
                                            (products.classifier_settings_by_site)
           settings_summary.csv             variant: means, ranges and counts, cross-site rho, x0
                                            (fitting times only in the log)
           settings_importance.csv          variant x site x indicator: mean |SHAP|, share, rank
           settings_crosssite.csv           variant x indicator: cross-site mean, mean share, ranks
           settings_transition_metrics.csv  variant x indicator: dS, |dS|, x0, w, resolved
           in_text_numbers.txt
         outputs/appendix/S12_spatial_cv/tables/table_classifier_settings.tex
         outputs/appendix/S12_spatial_cv/logs/16_classifier_settings.log
Run      python appendix/S12_spatial_cv/16_classifier_settings.py
         (after main/01-04; then 11_ranking_stability.py adds the rows to SI Table S16)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyogrio
import yaml
from scipy.stats import spearmanr

from favela_xai.attribution import favela_importance, shap_frame, tree_shap
from favela_xai.boundary import binned_mean, cross_site_profile, load_signed_distance, profile_edges, transition_metrics
from favela_xai.config import Config
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS, label_of
from favela_xai.io import read_site_features, write_table, write_text
from favela_xai.labels import label_to_int
from favela_xai.model import fit_main_model

import s12_common as C

cfg, s, P = C.setup(__file__)
CS = cfg.section("classifier_settings_sensitivity")
OUT = cfg.product("classifier_settings_by_site", create_parent=True).parent
log = C.log_to(P.logs / "16_classifier_settings.log")
FE = list(MODEL_FEATURE_ORDER)
TOPK = int(cfg.get("spatial_cv.ranking.top_k"))
BOUNDARY = list(CS["boundary_indicators"])
TOL = float(CS["reference_shap_tolerance"])
REF = CS["reference"]["key"]
VARIANTS = [(REF, CS["reference"]["label"], {})] + [(v["key"], v["label"], dict(v["model_main"])) for v in CS["variants"]]
SETTING_KEYS = ["n_estimators", "max_depth", "learning_rate", "random_state", "stratify", "split_random_state"]
EDGES, CENTRES = profile_edges(cfg)                       # 2.5 m bins on [-40, 40] m (Methods 4.8)
MIN_N = int(cfg.get("boundary_profiles.min_buildings_per_site_bin"))


# ============================================================================ configuration of a variant
REQUIRED = {"model_main.imputation": "median", "model_main.positive_class": "favela",
            "shap.explain": "all_buildings", "importance.population": "favela_buildings"}


def require_main_settings(c: Config, what: str) -> None:
    """The settings that main/02 and main/03 require (their check_settings / importance checks)."""
    for key, value in REQUIRED.items():
        if c.get(key) != value:
            raise ValueError(f"{what}: {key} must be '{value}' (as in the main analysis)")


def variant_config(overrides: dict) -> Config:
    """The configuration with the model_main keys of a variant replaced (config.yaml itself is not changed)."""
    data = yaml.safe_load(cfg.source.read_text(encoding="utf-8"))
    unknown = sorted(set(overrides) - set(data["model_main"]))
    if unknown:
        raise ValueError(f"classifier_settings_sensitivity: {unknown} are not keys of model_main")
    data["model_main"].update(overrides)
    vcfg = Config(data, root=cfg.root, source=cfg.source)
    require_main_settings(vcfg, f"classifier_settings_sensitivity variant {overrides}")
    return vcfg


require_main_settings(cfg, "config.yaml")


def stored(path, step: str):
    """A stored output of the main analysis (stop with a pointer to the step that writes it)."""
    if not path.exists():
        raise SystemExit(f"missing {path}\n(run {step} first)")
    return path


def read_stored(path, step: str) -> pd.DataFrame:
    """A stored CSV table, with every number read back exactly as written (round-trip float parsing)."""
    return pd.read_csv(stored(path, step), float_precision="round_trip")


# ============================================================================ inputs and fits
SD = load_signed_distance(cfg)


def site_inputs(site: str):
    """Indicators, labels, uID and signed distance of one site, in the row order of its building table."""
    g = read_site_features(cfg, site, columns=["uID", "label", *FE])
    X = pd.DataFrame(g[FE].astype(float).to_numpy(), columns=FE)       # as main/02_classification/01
    y = label_to_int(g["label"])
    uid = g["uID"].to_numpy()
    dist = SD.loc[SD["site"] == site, ["uID", "is_favela", "signed_dist"]]
    t = pd.DataFrame({"uID": uid}).merge(dist, on="uID", how="left", validate="one_to_one")
    if t["signed_dist"].isna().any():
        raise ValueError(f"{site}: {int(t['signed_dist'].isna().sum())} buildings have no signed distance")
    if not np.array_equal(y.astype(bool), t["is_favela"].astype(bool).to_numpy()):
        raise ValueError(f"{site}: labels of the building table and of the signed-distance product differ")
    return X, y, uid, t["signed_dist"].to_numpy(float)


def importance(shap: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Mean |SHAP| over the favela buildings, share and rank (main/03_site_signatures/01_importance_tables.py)."""
    imp = favela_importance(shap_frame(shap), y.astype(bool))
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    return imp


def leading(imp: pd.DataFrame, k: int) -> list[str]:
    return list(imp.sort_values("rank").index[:k])


def agreement(imp_v: pd.DataFrame, imp_0: pd.DataFrame) -> dict:
    """Agreement of a variant's site ranking with the main analysis (Spearman rho, leading indicators)."""
    a, b = imp_0["mean_abs_shap"].reindex(FE).to_numpy(), imp_v["mean_abs_shap"].reindex(FE).to_numpy()
    la, lb = leading(imp_0, TOPK), leading(imp_v, TOPK)
    k = len(set(la) & set(lb))
    return dict(rho=float(spearmanr(b, a).statistic), leader_reference=la[0], leader_variant=lb[0],
                leader_kept=bool(la[0] == lb[0]), top3_kept=k, top3_overlap=k / TOPK,
                two_of_top3_kept=bool(k >= TOPK - 1), all_top3_kept=bool(k == TOPK),
                top3_reference=";".join(la), top3_variant=";".join(lb))


def crosssite(per_site: dict) -> pd.DataFrame:
    """Cross-site mean of the per-site values and of the shares, and their ranks (main/03 crosssite_importance)."""
    wide = pd.DataFrame({site: per_site[site]["imp"]["mean_abs_shap"] for site in cfg.sites}).reindex(FE)
    share = pd.DataFrame({site: per_site[site]["imp"]["share"] for site in cfg.sites}).reindex(FE)
    out = pd.DataFrame({"crosssite_mean_abs_shap_favela": wide.mean(axis=1),
                        "crosssite_mean_share_favela": share.mean(axis=1)})
    out["crosssite_rank"] = out["crosssite_mean_abs_shap_favela"].rank(ascending=False, method="first").astype(int)
    out["share_rank"] = out["crosssite_mean_share_favela"].rank(ascending=False, method="first").astype(int)
    return out.rename_axis("feature")


def boundary(per_site: dict) -> pd.DataFrame:
    """|dS|, x0 and w of the cross-site profile of every indicator (main/04_boundary_profiles/01)."""
    rows = []
    for j, f in enumerate(FE):
        mu, _ = cross_site_profile(np.vstack([per_site[site]["profiles"][j] for site in cfg.sites]))
        rows.append(dict(feature=f, label=label_of(f), **transition_metrics(mu, CENTRES, cfg)))
    return pd.DataFrame(rows).set_index("feature")


def fit_site(vcfg: Config, X, y, uid, d) -> dict:
    """Main per-site model under a configuration: test metrics, Tree SHAP, importance and boundary profiles."""
    sm = fit_main_model(X, y, vcfg, uid=uid)
    shap = tree_shap(sm.model, sm.X_imputed, negate=sm.formal_as_1, model_output=vcfg.get("shap.model_output"))
    profiles = np.vstack([binned_mean(d, shap[:, j], EDGES, MIN_N) for j in range(len(FE))])
    return dict(sm=sm, shap=shap, metrics=sm.test_metrics(), imp=importance(shap, y), profiles=profiles,
                n_train=len(sm.train_idx))


# ============================================================================ reference check (main analysis)
CHECKS = []


def check(name: str, site: str, passed: bool, recomputed="", stored_value="", max_abs_difference=np.nan):
    CHECKS.append(dict(check=name, site=site, passed=bool(passed), recomputed=recomputed, stored=stored_value,
                       max_abs_difference=max_abs_difference))


def check_site(site: str, r: dict, uid) -> None:
    """Compare the recomputed main analysis of one site with the stored main/02-03 outputs."""
    sp = read_stored(cfg.product("split_csv", site=site), "main/02_classification/01_train_models.py")
    same_rows = np.array_equal(sp["uID"].to_numpy(), uid)
    check("held-out set (split.csv)", site, same_rows and np.array_equal(sp["is_test"].to_numpy(bool), r["sm"].is_test()),
          int(r["sm"].is_test().sum()), int(sp["is_test"].sum()))
    s3 = read_stored(cfg.product("site_metrics"), "main/02_classification/02_table_S3.py").set_index("site")
    acc, acc0 = float(r["metrics"]["accuracy"]), float(s3.loc[site, "accuracy"])
    check("test accuracy (SI Table S3)", site, acc == acc0, repr(acc), repr(acc0), abs(acc - acc0))
    st = pyogrio.read_dataframe(stored(cfg.product("shap_gpkg", site=site), "main/02_classification/01_train_models.py"),
                                columns=["uID"] + SHAP_COLUMNS, read_geometry=False)
    diff = float(np.max(np.abs(st[SHAP_COLUMNS].to_numpy(float) - r["shap"])))
    check(f"Tree SHAP within {TOL:g}", site, np.array_equal(st["uID"].to_numpy(), uid) and diff <= TOL,
          max_abs_difference=diff)
    im = read_stored(cfg.product("importance_favela"), "main/03_site_signatures/01_importance_tables.py")
    im = im[im["site"] == site].set_index("feature").reindex(FE)
    check("per-site ranks (ED Fig. 2)", site, np.array_equal(im["rank"].to_numpy(int), r["imp"]["rank"].reindex(FE).to_numpy()),
          ";".join(leading(r["imp"], TOPK)), ";".join(im.sort_values("rank").index[:TOPK]),
          float(np.max(np.abs(im["mean_abs_shap"].to_numpy(float) - r["imp"]["mean_abs_shap"].reindex(FE).to_numpy()))))


def check_all(cross: pd.DataFrame, trans: pd.DataFrame) -> None:
    """Compare the recomputed cross-site ranking and boundary metrics with main/03 and main/04."""
    cs = read_stored(cfg.output_dir("main/03_site_signatures", create=False) / "crosssite_importance.csv",
                     "main/03_site_signatures/01_importance_tables.py").set_index("feature").reindex(FE)
    check("cross-site ranks (ED Fig. 2 order)", "all sites",
          np.array_equal(cs["crosssite_rank"].to_numpy(int), cross["crosssite_rank"].reindex(FE).to_numpy()),
          ";".join(cross.sort_values("crosssite_rank").index[:TOPK]), ";".join(cs.sort_values("crosssite_rank").index[:TOPK]),
          float(np.max(np.abs(cs["crosssite_mean_abs_shap_favela"].to_numpy(float)
                              - cross["crosssite_mean_abs_shap_favela"].reindex(FE).to_numpy()))))
    tm = read_stored(cfg.product("transition_metrics"), "main/04_boundary_profiles/01_boundary_profiles.py")
    tm = tm.set_index("feature").reindex(FE)
    fmt = {"abs_delta_s": "{:.2f}", "x0": "{:+.1f}", "w": "{:.1f}"}
    for col, f_ in fmt.items():
        a, b = trans[col].reindex(FE).to_numpy(float), tm[col].to_numpy(float)
        same = [f_.format(u) == f_.format(v) for u, v in zip(a, b)]
        check(f"{col} of all 25 indicators at SI Table S5 precision", "all sites", all(same),
              ";".join(f_.format(v) for v in a[[FE.index(f) for f in BOUNDARY]]),
              ";".join(f_.format(v) for v in b[[FE.index(f) for f in BOUNDARY]]), float(np.nanmax(np.abs(a - b))))
    check("resolved flags (|dS| >= 0.1)", "all sites",
          np.array_equal(trans["resolved"].reindex(FE).to_numpy(bool), tm["resolved"].to_numpy(bool)))
    tex = cfg.output_dir("main/04_boundary_profiles", create=False) / "table_S5_transition_metrics.tex"
    lines = stored(tex, "main/04_boundary_profiles/03_table_S5.py").read_text(encoding="utf-8").splitlines()
    for f in BOUNDARY:
        row = next((ln for ln in lines if ln.startswith(label_of(f) + " & ")), None)
        cell = row.split("&")[3].strip() if row else "(row not found)"
        x0 = f"${trans.loc[f, 'x0']:+.1f}$"
        check(f"x0 cell of SI Table S5: {label_of(f)}", "all sites", cell == x0, x0, cell)


# ============================================================================ run
INPUTS = {site: site_inputs(site) for site in cfg.sites}
RES, CROSS, TRANS, SECONDS = {}, {}, {}, {}
for key, label, over in VARIANTS:
    vcfg = variant_config(over)
    m = vcfg.section("model_main")
    log(f"{key}: {label} | " + ", ".join(f"{k} {m[k]}" for k in SETTING_KEYS))
    t0 = time.time()
    per_site = {}
    for site in cfg.sites:
        X, y, uid, d = INPUTS[site]
        r = fit_site(vcfg, X, y, uid, d)
        if key == REF:
            check_site(site, r, uid)
        del r["sm"], r["shap"]                           # drop the fitted model and the SHAP matrix
        per_site[site] = r
        log(f"  {cfg.site_name(site):24s} accuracy {r['metrics']['accuracy']:.4f}  top three "
            + ", ".join(label_of(f) for f in leading(r["imp"], TOPK)))
    RES[key], CROSS[key], TRANS[key] = per_site, crosssite(per_site), boundary(per_site)
    SECONDS[key] = time.time() - t0
    log(f"  {key} fitted in {SECONDS[key]:.0f} s; x0 " + ", ".join(
        f"{label_of(f)} {TRANS[key].loc[f, 'x0']:+.2f} m" for f in BOUNDARY))
    if key == REF:
        check_all(CROSS[key], TRANS[key])
        CK = pd.DataFrame(CHECKS)
        write_table(CK, OUT / "reference_check.csv")
        failed = CK[~CK["passed"]]
        log(f"reference check: {len(CK) - len(failed)} of {len(CK)} checks passed; max |SHAP - stored| "
            f"{CK[CK.check.str.startswith('Tree SHAP')].max_abs_difference.max():.2e}")
        if len(failed):
            for q in failed.itertuples():
                log(f"  FAILED {q.check} [{q.site}]: recomputed {q.recomputed}, stored {q.stored}, "
                    f"max |difference| {q.max_abs_difference}")
            log("the main analysis does not reproduce the stored outputs; no variant is fitted")
            raise SystemExit(1)

# ============================================================================ tables
ref = RES[REF]
site_rows, imp_rows = [], []
for key, label, over in VARIANTS:
    m = variant_config(over).section("model_main")
    for site in cfg.sites:
        r = RES[key][site]
        a = agreement(r["imp"], ref[site]["imp"])
        X, y, uid, d = INPUTS[site]
        site_rows.append(dict(variant=key, label=label, is_reference=key == REF, site=site, site_name=cfg.site_name(site),
                              **{k: m[k] for k in SETTING_KEYS}, n_buildings=len(y), n_favela=int(y.sum()),
                              n_train=r["n_train"], **r["metrics"],
                              accuracy_minus_reference=r["metrics"]["accuracy"] - ref[site]["metrics"]["accuracy"],
                              fav_covered=int(y.sum()), coverage=1.0, **a))
        for f, q in r["imp"].iterrows():
            imp_rows.append(dict(variant=key, site=site, feature=f, label=label_of(f), mean_abs_shap=q["mean_abs_shap"],
                                 share=q["share"], rank=int(q["rank"])))
BS = pd.DataFrame(site_rows)
write_table(BS, cfg.product("classifier_settings_by_site"))
write_table(pd.DataFrame(imp_rows), OUT / "settings_importance.csv")
write_table(pd.concat([CROSS[k].reset_index().assign(variant=k) for k, _, _ in VARIANTS], ignore_index=True)
            [["variant", "feature", "crosssite_mean_abs_shap_favela", "crosssite_mean_share_favela", "crosssite_rank",
              "share_rank"]], OUT / "settings_crosssite.csv")
write_table(pd.concat([TRANS[k].reset_index().assign(variant=k) for k, _, _ in VARIANTS], ignore_index=True)
            [["variant", "feature", "label", "delta_s", "abs_delta_s", "x0", "w", "resolved"]],
            OUT / "settings_transition_metrics.csv")


def names(x: pd.DataFrame) -> str:
    return ";".join(cfg.site_name(v) for v in x["site"])


summary = []
for key, label, over in VARIANTS:
    x = BS[BS.variant == key]
    c, c0 = CROSS[key], CROSS[REF]
    top_c, top_c0 = list(c.sort_values("crosssite_rank").index[:TOPK]), list(c0.sort_values("crosssite_rank").index[:TOPK])
    r = dict(variant=key, label=label, **{k: x[k].iloc[0] for k in SETTING_KEYS}, n_sites=len(x),
             mean_accuracy=x.accuracy.mean(), accuracy_min=x.accuracy.min(), accuracy_max=x.accuracy.max(),
             mean_accuracy_minus_reference=x.accuracy_minus_reference.mean(),
             accuracy_minus_reference_min=x.accuracy_minus_reference.min(),
             accuracy_minus_reference_max=x.accuracy_minus_reference.max(),
             rho_mean=x.rho.mean(), rho_min=x.rho.min(), rho_max=x.rho.max(),
             crosssite_rho=float(spearmanr(c["crosssite_mean_abs_shap_favela"].reindex(FE),
                                           c0["crosssite_mean_abs_shap_favela"].reindex(FE)).statistic),
             crosssite_share_rho=float(spearmanr(c["crosssite_mean_share_favela"].reindex(FE),
                                                 c0["crosssite_mean_share_favela"].reindex(FE)).statistic),
             crosssite_top3=";".join(top_c), crosssite_leader_kept=top_c[0] == top_c0[0],
             crosssite_top3_kept=len(set(top_c) & set(top_c0)),
             sites_leader_kept=int(x.leader_kept.sum()), sites_two_of_top3_kept=int(x.two_of_top3_kept.sum()),
             sites_all_top3_kept=int(x.all_top3_kept.sum()),
             sites_leader_changed=names(x[~x.leader_kept]), sites_fewer_than_two_of_top3=names(x[~x.two_of_top3_kept]))
    for f in BOUNDARY:
        for col in ("x0", "w", "abs_delta_s", "resolved"):
            r[f"{col}_{f}"] = TRANS[key].loc[f, col]
    summary.append(r)
SU = pd.DataFrame(summary)
write_table(SU, OUT / "settings_summary.csv")

# ============================================================================ LaTeX: separate table of the variants


def sx0(v: float, resolved: bool) -> str:
    return f"${v:+.1f}$" if resolved and np.isfinite(v) else "--"


n_sites = len(cfg.sites)
body = ["\\toprule",
        "Settings & Mean accuracy & $\\rho$, sites: mean (range) & $\\rho$, cross-site ranking & Leading indicator kept "
        "& Top three: $\\geq$2 kept & all 3 kept & "
        + " & ".join(f"$x_0$ (m), {label_of(f, tex=True)}" for f in BOUNDARY) + " \\\\",
        "\\midrule"]
for r in SU.itertuples():
    x0s = " & ".join(sx0(getattr(r, f"x0_{f}"), getattr(r, f"resolved_{f}")) for f in BOUNDARY)
    if r.variant == REF:
        body.append(f"{r.label} & {r.mean_accuracy:.3f} & 1 (reference) & 1 (reference) & -- & -- & -- & {x0s} \\\\")
    else:
        body.append(f"{r.label} & {r.mean_accuracy:.3f} & {r.rho_mean:.3f} ({r.rho_min:.3f}--{r.rho_max:.3f}) & "
                    f"{r.crosssite_rho:.3f} & {r.sites_leader_kept}/{n_sites} & {r.sites_two_of_top3_kept}/{n_sites} & "
                    f"{r.sites_all_top3_kept}/{n_sites} & {x0s} \\\\")
body.append("\\bottomrule")
cap = ("Site-level results under alternative classifier settings. Each row refits the main per-site model of every site "
       "with the settings shown, everything else as in the main analysis (class coding, intercept, split drawn on the "
       "buildings ordered by their identifier, training-median imputation, path-dependent Tree SHAP of every building). Mean accuracy: "
       "held-out accuracy on the row's own 20\\% test split, mean over the ten sites. $\\rho$: Spearman correlation of the "
       "25 indicators' mean $|$SHAP$|$ over the site's favela buildings with the main analysis, mean over the sites (range "
       "in parentheses); cross-site ranking: mean of the per-site values over the ten sites (the order of Extended Data "
       "Fig.~2). Leading indicator kept and top three: sites keeping the main analysis's leading indicator, or at least "
       "two or all three of its three leading indicators, in their own top three. $x_0$: sign-change distance of the "
       "cross-site boundary profile (Methods; Supplementary Table~\\ref{tab:spillover_metrics}; $x_0>0$ outside the "
       "official 2013 boundary).")
C.write_tex_table(P.tables / "table_classifier_settings.tex", "16_classifier_settings.py", cap, "tab:classifier_settings",
                  body, cfg, tabular="l" + "r" * (6 + len(BOUNDARY)))

# ============================================================================ in-text numbers
lines = ["Classifier-settings sensitivity (main per-site model refitted; every variant)"]
for r in SU.itertuples():
    x0s = ", ".join(f"x0 {label_of(f)} {getattr(r, f'x0_{f}'):+.1f} m" for f in BOUNDARY)
    lines.append(f"{r.variant} {r.label}: mean accuracy {r.mean_accuracy:.3f} ({r.accuracy_min:.3f}-{r.accuracy_max:.3f}); "
                 f"rho {r.rho_mean:.3f} ({r.rho_min:.3f}-{r.rho_max:.3f}); cross-site rho {r.crosssite_rho:.3f} "
                 f"(mean of shares {r.crosssite_share_rho:.3f}); leading indicator kept {r.sites_leader_kept}/{n_sites}; "
                 f">= {TOPK - 1} of top {TOPK} kept {r.sites_two_of_top3_kept}/{n_sites}; all {TOPK} kept "
                 f"{r.sites_all_top3_kept}/{n_sites}; {x0s}"
                 + (f"; leading indicator changed in: {r.sites_leader_changed}" if r.sites_leader_changed else "")
                 + (f"; fewer than {TOPK - 1} of top {TOPK} kept in: {r.sites_fewer_than_two_of_top3}"
                    if r.sites_fewer_than_two_of_top3 else ""))
write_text("\n".join(lines) + "\n", OUT / "in_text_numbers.txt")
for ln in lines:
    log(ln)
log("done")
