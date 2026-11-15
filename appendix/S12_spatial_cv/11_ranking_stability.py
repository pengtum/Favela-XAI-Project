"""Step 11: stability of the site-level indicator rankings; SI Table S16 and the ranking numbers of SI S12.

Manuscript items
* SI Table S16 -> tables/table_R1_ranking_stability.tex, including the three
  alternative-unit rows of the 200 m group, and, when the output of
  16_classifier_settings.py exists (products.classifier_settings_by_site), the rows
  'Classifier settings' with their caption sentence (also written alone to
  tables/table_R1_ranking_stability_settings_rows.tex).
* The ranking values of SI S12 'Stability of the attributions' and Results 2.2
  (in_text_numbers.txt): Spearman rho with the main-text ranking 0.871, 0.901 and 0.854
  for 250, 500 and 1,000 m blocks with the 200 m buffer, against 0.924 for the
  split-refit null; top three kept; the ladder (no buffer, size-matched control,
  buffer); site-matched contrasts; the three alternative units.
* Inputs of Extended Data Fig. 3 (panels b, c) and Extended Data Fig. 4: ranking_summary.csv,
  ranking_cells.csv, ranking_long.csv, ranking_by_run.csv.

Quantity (SI S12; ED Fig. 3 caption): each run's share of mean |SHAP| over the explained
favela buildings for the 25 indicators; Spearman's rho with the main-text model's
shares on the same buildings; top three = the main-text model's three leading
indicators on those buildings. Per design: mean over the site's runs (grid placements or
seeds), then over sites; 95% interval from 2,000 site-bootstrap resamples; rho minus
null = paired difference from the split-refit null over the same sites.
Scoring (SI S12 'Scoring'): first criterion - only folds whose training set holds at
least 50 buildings of the minority class explain buildings; second criterion - a run is
scored only if these folds explain at least half of the site's favela buildings. A run
has a ranking when it explains at least ``ranking.min_covered_buildings`` buildings
including at least one favela building. Ladder (SI S12): 250 m blocks without buffer
(three placements) -> size-matched control -> 250 m blocks with the 200 m buffer, on the
folds where both the buffered and the control fit pass the first criterion and on the
site-runs scored in both.

Inputs  outputs/appendix/S12_spatial_cv/shap_fold_sums/ (step 10); site tables (step 01);
        optional: classifier_settings/settings_by_site.csv (step 16)
Outputs outputs/appendix/S12_spatial_cv/tables/table_R1_ranking_stability.tex
        (with step 16: tables/table_R1_ranking_stability_settings_rows.tex,
        ranking/ranking_summary_classifier_settings.csv)
        outputs/appendix/S12_spatial_cv/ranking/ ranking_by_run.csv, ranking_vectors.parquet,
          ranking_summary.csv, ranking_cells.csv, ranking_long.csv, ranking_ladder.csv,
          ranking_ladder_steps.csv, site_matched.csv, in_text_numbers.txt
Run     python appendix/S12_spatial_cv/11_ranking_stability.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import s12_common as C

cfg, s, P = C.setup(__file__)
RK = C.section(cfg)["ranking"]
SR = cfg.section("spatial_cv")["ranking"]
OUT = P.out / "ranking"
OUT.mkdir(exist_ok=True)
log = C.log_to(P.logs / "11_ranking_stability.log")
FE = C.FEATURES
ABS_OOF = ["abs_oof_" + f for f in FE]
ABS_MAIN = ["abs_main_" + f for f in FE]
PL = list(range(int(cfg.get("spatial_cv.grid_placements"))))
PRIM_L, PRIM_B = int(cfg.get("spatial_cv.primary.block_m")), int(cfg.get("spatial_cv.primary.buffer_m"))
CTL = f"sq{int(RK['control_design']['block_m'])}_b{int(RK['control_design']['buffer_m'])}"
MIN_COV_B = int(RK["min_covered_buildings"])
TOPK = int(SR["top_k"])
SKF = [f"skf5_s{int(v)}" for v in SR["five_fold_seeds"]]
NULL = [f"null_s{int(v)}" for v in SR["null_split_seeds"]]

FS = pd.concat([pd.read_parquet(f) for f in sorted(P.shap_folds.glob("*.parquet"))], ignore_index=True)
NF = {site: C.load_site(P, site).n_favela for site in cfg.sites}
MAINF = {}                                              # full-site main-model favela |SHAP| means (ED Figs 3b, 4 x axis)
for site in cfg.sites:
    d, m = C.load_site(P, site), C.load_main_shap(P, site)
    MAINF[site] = np.abs(m[d.y == 1]).mean(0)


# ============================================================================ per-run vectors and metrics
def mets(vp, vo) -> dict:
    """Agreement of an out-of-fold vector vo with the main-model vector vp on the same buildings."""
    rp, ro = np.argsort(-vp), np.argsort(-vo)
    sp, so = vp / vp.sum(), vo / vo.sum()
    return dict(rho=spearmanr(vo, vp).statistic, top3_overlap=len(set(rp[:TOPK]) & set(ro[:TOPK])) / TOPK,
                top5_overlap=len(set(rp[:5]) & set(ro[:5])) / 5, top1_same=float(rp[0] == ro[0]),
                share_mad_pp=100 * np.abs(so - sp).mean())


def run_summary(site, run, folds: pd.DataFrame, vec_rows: list):
    """Coverage, scoring and ranking of one run from the fold rows it keeps (first criterion applied)."""
    use = folds[folds.rule1_ok.astype(bool)]
    fitted = folds[folds.status == "ok"]
    n_cov, nf_cov = int(use.n_test.sum()), int(use.n_fav_test.sum())
    r = dict(site=site, run=run, n_folds=len(folds), n_fitted=len(fitted), n_used=len(use),
             with_prediction=len(use) > 0, fitted_any=len(fitted) > 0, covered=n_cov, fav_covered=nf_cov,
             coverage=nf_cov / NF[site], scored=nf_cov / NF[site] >= s.min_favela_coverage)
    r["has_ranking"] = bool(n_cov >= MIN_COV_B and nf_cov > 0)
    if r["has_ranking"]:
        vo = use[ABS_OOF].to_numpy(float).sum(0) / nf_cov
        vp = use[ABS_MAIN].to_numpy(float).sum(0) / nf_cov
        r.update(mets(vp, vo))
        vec_rows.append(pd.DataFrame(dict(site=site, run=run, feature=FE, share_oof=vo / vo.sum(), share_pub=vp / vp.sum())))
    return r


vec_rows, runs = [], []
for (site, run), g in FS.groupby(["site", "run"], sort=False):
    if run.startswith("control_"):
        continue
    runs.append(run_summary(site, run, g, vec_rows))
R = pd.DataFrame(runs)
V = pd.concat(vec_rows, ignore_index=True)
R.to_csv(OUT / "ranking_by_run.csv", index=False)
V.to_parquet(OUT / "ranking_vectors.parquet", index=False)


def version(v):
    """'final' = both criteria (scored runs with a ranking); 'first' = first criterion alone (runs with a ranking)."""
    x = R[R.has_ranking]
    return x[x.scored] if v == "final" else x


# ============================================================================ design groups (rows of SI Table S16)
FB = [int(v) for v in RK["first_placement_blocks_m"]]
FBUF = [int(v) for v in RK["first_placement_buffers_m"]]
TP = {(int(a), int(b)) for a, b in RK["three_placement_designs"]}
ALTB = int(RK["alternative_unit_buffer_m"])
SMALL, MAXB = min(FB), max(FBUF)


def blk(L, b, three=False):
    return [f"sq{L}_b{b}_p{r}" for r in (PL if three else [0])]


def lab(L):
    return f"Blocks {L:,} m"


G = [("ref", "skf5_4seeds", "Random five-fold (four shuffle seeds)", SKF),
     ("ref", "null_3splits", "Split-refit null (three random 80/20 refits)", NULL)]
for L in FB[1:]:
    G.append(("b0", f"sq{L}_b0", lab(L), blk(L, 0)))
    if (L, 0) in TP:
        G.append(("b0", f"sq{L}_b0_3place", f"{lab(L)} (three placements)", blk(L, 0, True)))
G += [("b0", "component_b0", "Held-out favela component", ["component_b0"]),
      ("b0", f"sq{SMALL}_b0", lab(SMALL), blk(SMALL, 0))]
for L in FB[1:]:
    three = (L, PRIM_B) in TP
    G.append(("b200", f"sq{L}_b{PRIM_B}" + ("_3place" if three else ""), f"{lab(L)} + {PRIM_B} m", blk(L, PRIM_B, three)))
G.append(("b200", f"component_b{PRIM_B}", f"Held-out favela component + {PRIM_B} m", [f"component_b{PRIM_B}"]))
ALT_LAB = {"hex": "Hexagons {L} m + {B} m", "kmm": "$k$-means units ($\\approx${L} m) + {B} m",
           "sq": "Blocks {L} m + {B} m, ten folds"}
for name in RK["alternative_unit_designs"]:
    kind = "hex" if name.startswith("hex") else "kmm" if name.startswith("kmm") else "sq"
    L = name[len(kind):].split("_")[0]
    G.append(("b200", f"{name}_b{ALTB}", ALT_LAB[kind].format(L=L, B=ALTB), [f"{name}_b{ALTB}"]))
ALT_GIDS = {f"{name}_b{ALTB}" for name in RK["alternative_unit_designs"]}   # labels written in TeX already (ALT_LAB)
G.append(("stress", f"sq{SMALL}_b{PRIM_B}", f"{lab(SMALL)} + {PRIM_B} m", blk(SMALL, PRIM_B)))
G += [("stress", f"sq{L}_b{MAXB}", f"{lab(L)} + {MAXB} m", blk(L, MAXB)) for L in FB]
GD = {g[1]: g for g in G}


def site_mean(x, col="rho"):
    return x.groupby("site")[col].mean()


def two_plus(x) -> int:
    """Site-runs keeping at least two of the three leading indicators (k - 1 of the top k)."""
    return int(((x.top3_overlap * TOPK).round() >= TOPK - 1).sum())


rows = []
for grp, gid, label, runs_ in G:
    r = dict(group=grp, design=gid, label=label, runs=";".join(runs_), placements_or_seeds=len(runs_))
    for v in ("final", "first"):
        t = version(v)
        t = t[t.run.isin(runs_)]
        ps = t.groupby("site")[["rho", "top3_overlap"]].mean()
        m, lo, hi, n = C.boot_mean(ps.rho.to_numpy(), s)
        nul = site_mean(version(v)[version(v).run.isin(NULL)])
        dm, dlo, dhi, _ = C.boot_mean((ps.rho - nul.reindex(ps.index)).to_numpy(), s) if len(ps) else (np.nan,) * 4
        r.update({f"{v}_sites": n, f"{v}_site_runs": len(t), f"{v}_rho": m, f"{v}_rho_lo": lo, f"{v}_rho_hi": hi,
                  f"{v}_rho_minus_null": dm, f"{v}_rho_minus_null_lo": dlo, f"{v}_rho_minus_null_hi": dhi,
                  f"{v}_top3_two_plus": two_plus(t),
                  f"{v}_top3_all": int((t.top3_overlap == 1).sum()),
                  f"{v}_sites_list": ";".join(ps.index[np.isfinite(ps.rho)])})
    c = R[R.run.isin(runs_) & R.with_prediction]
    r.update(site_runs_with_prediction=len(c), coverage_mean=float(c.coverage.mean()) if len(c) else np.nan,
             coverage_min=float(c.coverage.min()) if len(c) else np.nan,
             coverage_max=float(c.coverage.max()) if len(c) else np.nan)
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(OUT / "ranking_summary.csv", index=False)

# ---------------------------------------------------------------------------- LaTeX (SI Table S16)


def ci(r, v):
    x = r[f"{v}_rho"]
    return "--" if not np.isfinite(x) else f"{x:.3f} ({r[f'{v}_rho_lo']:.3f}--{r[f'{v}_rho_hi']:.3f})"


def sg(x):
    return "--" if not np.isfinite(x) else f"{x:+.3f}".replace("-", "$-$")


def dci(r):
    if r.design == "null_3splits":
        return "0 (reference)"
    if not np.isfinite(r.final_rho_minus_null):
        return "--"
    return f"{sg(r.final_rho_minus_null)} ({sg(r.final_rho_minus_null_lo)} to {sg(r.final_rho_minus_null_hi)})"


def cnt(a, b):
    return f"{int(a)}/{int(b)}" if b else "--"


def tex_row(r) -> str:
    """One row of SI Table S16 from a row of ranking_summary.csv (or of the rows 'Classifier settings')."""
    cov = ("--" if not np.isfinite(r.coverage_mean) else "100\\%" if r.coverage_min == 1 else
           f"{100 * r.coverage_mean:.0f}\\% ({100 * r.coverage_min:.0f}--{100 * r.coverage_max:.0f})")
    fc = "--" if not np.isfinite(r.first_rho) else f"{r.first_rho:.3f} ({int(r.first_sites)}/{int(r.first_site_runs)})"
    label = r.label if r.design in ALT_GIDS else r.label.replace(",", "{,}")   # {,}: thousands and settings labels
    return (f"{label} & {int(r.final_sites)} & {cnt(r.final_site_runs, r.site_runs_with_prediction)} & "
            f"{ci(r, 'final')} & {dci(r)} & {cov} & {cnt(r.final_top3_two_plus, r.final_site_runs)} & "
            f"{cnt(r.final_top3_all, r.final_site_runs)} & {fc} & "
            f"{cnt(r.first_top3_two_plus, r.first_site_runs)} \\\\")


# ---------------------------------------------------------------------------- rows 'Classifier settings' (step 16)
def settings_summary() -> pd.DataFrame:
    """Rows 'Classifier settings' of SI Table S16 from 16_classifier_settings.py (empty when its output is absent).

    Each variant refits the main per-site model of every site and explains every building, so all
    favela buildings are covered and both criteria hold; rho, rho minus null and the top three are
    formed from the per-site values as for the design rows (one run per site).
    """
    path = cfg.product("classifier_settings_by_site")
    if not path.exists():
        return pd.DataFrame()
    Q = pd.read_csv(path)
    # the reference rows must carry the current model_main settings (else the file is stale)
    ref = Q[Q.is_reference.astype(bool)]
    mm = cfg.section("model_main")
    keys = ["n_estimators", "max_depth", "learning_rate", "stratify", "split_random_state"]
    keys += ["random_state"] if "random_state" in Q.columns else []
    if not len(ref) or any(ref[k].iloc[0] != mm[k] for k in keys):
        raise SystemExit(f"{path}: computed for other model_main settings; re-run 16_classifier_settings.py")
    Q = Q[~Q.is_reference.astype(bool)]
    rows = []
    for key, x in Q.groupby("variant", sort=False):
        ps = x.groupby("site")[["rho", "top3_overlap"]].mean()
        r = dict(group="settings", design=f"settings_{key}", label=x.label.iloc[0], runs=key, placements_or_seeds=1)
        for v in ("final", "first"):
            m, lo, hi, n = C.boot_mean(ps.rho.to_numpy(), s)
            nul = site_mean(version(v)[version(v).run.isin(NULL)])
            dm, dlo, dhi, _ = C.boot_mean((ps.rho - nul.reindex(ps.index)).to_numpy(), s)
            r.update({f"{v}_sites": n, f"{v}_site_runs": len(x), f"{v}_rho": m, f"{v}_rho_lo": lo, f"{v}_rho_hi": hi,
                      f"{v}_rho_minus_null": dm, f"{v}_rho_minus_null_lo": dlo, f"{v}_rho_minus_null_hi": dhi,
                      f"{v}_top3_two_plus": two_plus(x), f"{v}_top3_all": int((x.top3_overlap == 1).sum()),
                      f"{v}_sites_list": ";".join(ps.index[np.isfinite(ps.rho)])})
        r.update(site_runs_with_prediction=len(x), coverage_mean=float(x.coverage.mean()),
                 coverage_min=float(x.coverage.min()), coverage_max=float(x.coverage.max()))
        rows.append(r)
    return pd.DataFrame(rows)


TS = settings_summary()
if len(TS):
    TS.to_csv(OUT / "ranking_summary_classifier_settings.csv", index=False)

GL = {"ref": "References (no spatial separation)", "b0": "Spatial blocks and held-out components, no buffer",
      "b200": f"{PRIM_B} m buffer", "stress": "Stress tests",
      "settings": "Classifier settings (main per-site model refitted; no spatial separation)"}
body = ["\\toprule", " & \\multicolumn{7}{c}{Both criteria} & \\multicolumn{2}{c}{First criterion only} \\\\",
        "\\cmidrule(lr){2-8}\\cmidrule(lr){9-10}",
        "Design & Sites & Site-runs scored & $\\rho$ & $\\rho$ minus null & Coverage & Top three: $\\geq$2 kept & all 3 kept & $\\rho$ (sites/site-runs) & Top three: $\\geq$2 kept \\\\",
        "\\midrule"]
for grp in ("ref", "b0", "b200", "stress"):
    body.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{GL[grp]}}}}} \\\\")
    for _, r in T[T.group == grp].iterrows():
        body.append(tex_row(r))
    body.append("\\addlinespace")
SETTINGS_LINES = []
if len(TS):
    SETTINGS_LINES = ([f"\\multicolumn{{10}}{{l}}{{\\textit{{{GL['settings']}}}}} \\\\"]
                      + [tex_row(r) for _, r in TS.iterrows()] + ["\\addlinespace"])
body += SETTINGS_LINES
body.append("\\bottomrule")
# runs with a prediction but no ranking under the first criterion (SI Table S16 caption)
ALT_DESC = {f"{name}_b{ALTB}": ("hexagonal units" if name.startswith("hex") else "$k$-means units"
                                if name.startswith("kmm") else "ten folds") for name in RK["alternative_unit_designs"]}
nr = []
for _, q in R[R.with_prediction & ~R.has_ranking].iterrows():
    g = next((g for g in G if q.run in g[3]), None)
    if g is not None and (ALT_DESC.get(g[1], g[2]), q.site) not in nr:
        nr.append((ALT_DESC.get(g[1], g[2]), q.site))
nr_txt = "".join(f" With {desc} {cfg.site_name(site)} has no ranking even under the first criterion alone, because that "
                 "criterion leaves it no covered favela building, so it is counted among the site-runs with a prediction "
                 "and excluded from $\\rho$ and the top-three counts." for desc, site in nr)
seeds = [str(int(v)) for v in SR["null_split_seeds"]]
cap = ("Stability of the site-level feature rankings under spatial cross-validation"
       + (" and alternative classifier settings" if len(TS) else "") + ". $\\rho$: Spearman correlation with "
       "the main-text model's ranking on the same buildings, mean over the site's runs (grid placements or seeds), then "
       "over sites, with a site-bootstrap 95\\% interval; $\\rho$ minus null: paired difference from the split-refit null "
       "over the same sites. Top three: site-runs keeping at least two, or all three, of the main-text model's three "
       "leading features in their own top three. Scoring: a fold is used only if its training set holds at least 50 "
       "buildings of the minority class (favela or formal settlement); a run is scored only if the used folds explain at "
       "least half of the site's favela buildings. Coverage: favela buildings explained by the used folds, as a share of "
       "the site's favela buildings, mean over site-runs with a prediction (range in parentheses); site-runs scored: "
       "scored/with a prediction. First criterion only: $\\rho$ with the first criterion alone (sites/site-runs), and "
       f"site-runs keeping at least two of the top three. Buffers of {PRIM_B} m use three grid placements, except the "
       "three alternative-unit rows, which use one; rows without a buffer use the first placement, except the row marked "
       f"three placements.{nr_txt} Stress tests (first grid placement): {SMALL} m blocks are smaller than the {PRIM_B} m "
       f"indicator neighbourhood, and a {MAXB} m buffer is twice that neighbourhood; neither is used for the main "
       f"conclusions. The split-refit null uses split seeds {', '.join(seeds[:-1])} and {seeds[-1]}.")
if len(TS):
    MM = cfg.section("model_main")
    SETTINGS_CAPTION = (
        "Classifier settings: the per-site models of Section~\\ref{sec:per_site} refitted with the settings shown and all "
        f"others as in the main analysis ({MM['n_estimators']} trees, maximum depth {MM['max_depth']}, learning rate "
        f"{MM['learning_rate']}, {'stratified' if MM['stratify'] else 'unstratified'} "
        f"{100 * (1 - MM['test_size']):.0f}/{100 * MM['test_size']:.0f} split with seed {MM['split_random_state']}), with "
        "Tree SHAP of every building, so coverage is 100\\%.")
    cap += " " + SETTINGS_CAPTION
    (P.tables / "table_R1_ranking_stability_settings_rows.tex").write_text(
        "\n".join(["% rows 'Classifier settings' of SI Table S16 (tab:ranking_stability), appended before \\bottomrule by",
                    "% appendix/S12_spatial_cv/11_ranking_stability.py from products.classifier_settings_by_site",
                    "% (appendix/S12_spatial_cv/16_classifier_settings.py); sentence appended to the caption:",
                    "% " + SETTINGS_CAPTION, *SETTINGS_LINES, ""]), encoding="utf-8")
C.write_tex_table(P.tables / "table_R1_ranking_stability.tex", "11_ranking_stability.py", cap, "tab:ranking_stability",
                  body, cfg, tabular="lrrlllrrll")

# ============================================================================ cells and long ranks (ED Figs 3-4)


def rank_of(v):
    o = np.argsort(-np.asarray(v, float))
    r = np.empty(len(v), int)
    r[o] = np.arange(1, len(v) + 1)
    return r


fin = version("final")
cells, long_rows = [], []
for site in cfg.sites:
    sp = MAINF[site] / MAINF[site].sum()
    long_rows.append(pd.DataFrame(dict(site=site, design="published", feature=FE, rank_pub=rank_of(sp), rank_run=rank_of(sp),
                                       share_pub=sp, share_run=sp, n_runs=0)))
    for grp, gid, label, runs_ in G:
        rs = R[(R.site == site) & R.run.isin(runs_)]
        rx = fin[(fin.site == site) & fin.run.isin(runs_)]
        fitted = rs[rs.fitted_any]
        if (FS[(FS.site == site) & FS.run.isin(runs_)].status == "skipped_single_group").any():
            status = "one component"
        elif not len(fitted):
            status = "no scored fold"
        elif not len(rx):
            status = "under half of favela buildings explained"
        else:
            status = "ok"
        cell = dict(site=site, site_name=cfg.site_name(site), n_favela=NF[site], design=gid, status=status,
                    runs_final=";".join(rx.run), n_runs_final=len(rx), rho=float(rx.rho.mean()) if len(rx) else np.nan,
                    top3_overlap=float(rx.top3_overlap.mean()) if len(rx) else np.nan,
                    fav_explained_mean_all=float(fitted.coverage.mean()) if len(fitted) else np.nan,
                    fav_explained_by_run=";".join(f"{a}={b:.6f}" for a, b in zip(fitted.run, fitted.coverage)))
        cells.append(cell)
        if len(rx):
            so = np.array([V[(V.site == site) & (V.run == r_)].set_index("feature").share_oof.reindex(FE).to_numpy()
                           for r_ in rx.run])
            ms = so.mean(0)
            long_rows.append(pd.DataFrame(dict(site=site, design=gid, feature=FE, rank_pub=rank_of(sp), rank_run=rank_of(ms),
                                               share_pub=sp, share_run=ms, n_runs=len(rx))))
CE = pd.DataFrame(cells)
CE.to_csv(OUT / "ranking_cells.csv", index=False)
LR = pd.concat(long_rows, ignore_index=True)
LR["site_name"] = LR.site.map(cfg.site_name)
LR.to_csv(OUT / "ranking_long.csv", index=False)

# ============================================================================ ladder (SI S12)
ladder_rows, site_means = [], {}
ctl_runs = {r: f"control_{CTL}_p{r}" for r in PL}
buf_runs = {r: f"{CTL}_p{r}" for r in PL}
lad = []
for site in cfg.sites:
    for r in PL:
        fb = FS[(FS.site == site) & (FS.run == buf_runs[r])].set_index("fold")
        fc = FS[(FS.site == site) & (FS.run == ctl_runs[r])].set_index("fold")
        if not len(fb) or not len(fc):
            continue
        matched = [f for f in fb.index if fb.loc[f, "rule1_ok"] and f in fc.index and fc.loc[f, "rule1_ok"]]
        for tag, src in (("buffered", fb), ("control", fc)):
            q = run_summary(site, f"{tag}_p{r}", src.loc[matched].reset_index(), [])
            lad.append(dict(q, tag=tag, placement=r))
LD = pd.DataFrame(lad)
if len(LD):
    Bm = LD[LD.tag == "buffered"].set_index(["site", "placement"])
    Km = LD[LD.tag == "control"].set_index(["site", "placement"])
    both = Bm.index[(Bm.scored & Bm.has_ranking & Km.reindex(Bm.index).scored.fillna(False).astype(bool)
                     & Km.reindex(Bm.index).has_ranking.fillna(False).astype(bool)).to_numpy(bool)]
    B0_3 = GD[f"sq{int(RK['control_design']['block_m'])}_b0_3place"][3]
    STEPS = {"both criteria": dict(b0=version("final")[version("final").run.isin(B0_3)],
                                   control=Km.loc[both].reset_index(), buffered=Bm.loc[both].reset_index(),
                                   null=version("final")[version("final").run.isin(NULL)],
                                   skf=version("final")[version("final").run.isin(SKF)]),
             "first criterion only": dict(b0=version("first")[version("first").run.isin(B0_3)],
                                          control=Km[Km.has_ranking].reset_index(), buffered=Bm[Bm.has_ranking].reset_index(),
                                          null=version("first")[version("first").run.isin(NULL)],
                                          skf=version("first")[version("first").run.isin(SKF)])}
    cl, cb = int(RK["control_design"]["block_m"]), int(RK["control_design"]["buffer_m"])
    LAB = {"b0": f"{cl} m blocks, no buffer (three grid placements)",
           "control": f"Size-matched control: the {cb} m buffer's number of training buildings removed at random",
           "buffered": f"{cl} m blocks, {cb} m buffer (three grid placements)",
           "null": "Reference: split-refit null (three random 80/20 refits)",
           "skf": "Reference: random five-fold (four shuffle seeds)"}
    for sc, St in STEPS.items():
        sites = sorted(St["buffered"].site.unique())
        for k, x in St.items():
            x = x[x.site.isin(sites)]
            sm = x.groupby("site").rho.mean().reindex(sites)
            site_means[(sc, k)] = sm
            m, lo, hi, n = C.boot_mean(sm.to_numpy(), s)
            ladder_rows.append(dict(scoring=sc, step=LAB[k], sites=n, site_runs=len(x), rho=m, rho_lo=lo, rho_hi=hi,
                                    top3_two_plus=two_plus(x),
                                    top3_all=int((x.top3_overlap == 1).sum())))
    pd.DataFrame(ladder_rows).to_csv(OUT / "ranking_ladder.csv", index=False)
    DIFF = [("control", "b0"), ("buffered", "control"), ("buffered", "b0"), ("control", "null"), ("buffered", "null"),
            ("b0", "null")]
    drows = []
    for a, b in DIFF:
        for sc in STEPS:
            m, lo, hi, n = C.boot_mean((site_means[(sc, a)] - site_means[(sc, b)]).to_numpy(), s)
            drows.append(dict(scoring=sc, difference=f"{a} minus {b}", rho_difference=m, ci_lo=lo, ci_hi=hi, sites=n))
    LS = pd.DataFrame(drows)
    LS.to_csv(OUT / "ranking_ladder_steps.csv", index=False)

# ============================================================================ site-matched contrasts (SI S12)
matched = []
pairs = [(f"sq{L}_b0", f"sq{L}_b{PRIM_B}" + ("_3place" if (L, PRIM_B) in TP else "")) for L in FB[1:]]
pairs.append(("component_b0", f"component_b{PRIM_B}"))
for b0, b2 in pairs:
    a = CE[(CE.design == b0) & (CE.status == "ok")].set_index("site").rho
    c = CE[(CE.design == b2) & (CE.status == "ok")].set_index("site").rho
    com = a.index.intersection(c.index)
    matched.append(dict(no_buffer=b0, buffer=b2, common_sites=len(com), rho_no_buffer=float(a[com].mean()),
                        rho_buffer=float(c[com].mean()), sites=";".join(com)))
SM = pd.DataFrame(matched)
SM.to_csv(OUT / "site_matched.csv", index=False)

# ============================================================================ in-text numbers
Ti = T.set_index("design")
lines = ["SI S12 'Stability of the attributions' (both criteria unless stated)"]
for L in FB[1:]:
    gid = f"sq{L}_b{PRIM_B}" + ("_3place" if (L, PRIM_B) in TP else "")
    r = Ti.loc[gid]
    lines.append(f"{L} m blocks + {PRIM_B} m: rho {r.final_rho:.3f} ({r.final_rho_lo:.3f}-{r.final_rho_hi:.3f}), "
                 f"{int(r.final_sites)} sites; minus null {r.final_rho_minus_null:+.3f} ({r.final_rho_minus_null_lo:+.3f} to "
                 f"{r.final_rho_minus_null_hi:+.3f}); top three >= 2 kept {int(r.final_top3_two_plus)} of "
                 f"{int(r.final_site_runs)}")
r = Ti.loc["null_3splits"]
lines.append(f"split-refit null: rho {r.final_rho:.3f} ({r.final_rho_lo:.3f}-{r.final_rho_hi:.3f})")
lines.append("(manuscript: 0.871 (0.836-0.904), 0.901 (0.858-0.936), 0.854 (0.782-0.919); null 0.924 (0.903-0.943); "
             "250 m minus null -0.052 (-0.078 to -0.027); 27/27, 27/27, 22/22)")
if len(LD):
    Lr = pd.DataFrame(ladder_rows)
    Lb = Lr[Lr.scoring == "both criteria"].set_index("step")
    lines.append("ladder: " + " -> ".join(f"{Lb.loc[LAB[k], 'rho']:.3f}" for k in ("b0", "control", "buffered"))
                 + f" over {int(Lb.loc[LAB['buffered'], 'sites'])} sites (manuscript 0.947 -> 0.904 -> 0.871)")
    q = LS[LS.scoring == "both criteria"].set_index("difference")
    lines.append(f"control minus null {q.loc['control minus null', 'rho_difference']:+.3f} "
                 f"({q.loc['control minus null', 'ci_lo']:+.3f} to {q.loc['control minus null', 'ci_hi']:+.3f}); buffer "
                 f"minus control {q.loc['buffered minus control', 'rho_difference']:+.3f} "
                 f"({q.loc['buffered minus control', 'ci_lo']:+.3f} to {q.loc['buffered minus control', 'ci_hi']:+.3f}) "
                 "(manuscript -0.018 (-0.032 to -0.007); the buffer costs a further 0.033 (0.010-0.055))")
for r in SM.itertuples():
    lines.append(f"site-matched {r.no_buffer} -> {r.buffer}: {r.rho_no_buffer:.3f} -> {r.rho_buffer:.3f} over "
                 f"{r.common_sites} common sites")
lines.append("(manuscript: 0.943 -> 0.871 and 0.949 -> 0.901 over nine sites; 0.874 -> 0.885 over seven; 0.884 -> 0.846)")
for name in RK["alternative_unit_designs"]:
    gid = f"{name}_b{ALTB}"
    r = Ti.loc[gid]
    pal = R[(R.run == gid) & R.with_prediction & ~R.scored]
    lines.append(f"{gid}: rho {r.final_rho:.3f} ({r.final_rho_lo:.3f}-{r.final_rho_hi:.3f}), {int(r.final_sites)} sites; "
                 f"minus null {r.final_rho_minus_null:+.3f} ({r.final_rho_minus_null_lo:+.3f} to "
                 f"{r.final_rho_minus_null_hi:+.3f}); top three >= 2 kept {int(r.final_top3_two_plus)}/"
                 f"{int(r.final_site_runs)}; not scored: "
                 + (", ".join(f"{cfg.site_name(x.site)} ({100 * x.coverage:.0f}%)" for x in pal.itertuples()) or "none"))
lines.append("(manuscript: 0.882 (0.859-0.905), 0.856 (0.811-0.901), 0.927 (0.888-0.959); ten folds minus null +0.004 "
             "(-0.027 to +0.027); Palmeiras 8, 0 and 29%)")
b_sites = [Ti.loc[f"sq{L}_b{PRIM_B}" + ("_3place" if (L, PRIM_B) in TP else "")].final_rho for L in FB[1:]]
lines.append(f"Results 2.2: site mean rho across block sizes {min(b_sites):.2f}-{max(b_sites):.2f} against "
             f"{Ti.loc['null_3splits'].final_rho:.2f} (manuscript 0.85-0.90 against 0.92)")
(OUT / "in_text_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
for ln in lines:
    log(ln)
log("done")
