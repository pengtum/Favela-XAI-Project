"""Step 5: scored accuracy of every design; SI Tables S10-S14 and the accuracy numbers of SI S12.

Manuscript items
* SI Table S10 (the sixteen designs) -> tables/table_R1_spatialcv_designs.tex
* SI Table S11 (square blocks x buffers) -> tables/table_R1_spatialcv_grid.tex
* SI Table S12 (sensitivity designs, grouped hold-outs) -> tables/table_R1_spatialcv_sensitivity.tex
* SI Table S13 (folds and sites removed by the scoring) -> tables/table_R1_spatialcv_coverage.tex
* SI Table S14 (fold-level summary) -> tables/table_R1_spatialcv_folds.tex
* the values plotted in SI Fig. S19 (scored_grid.csv, scored_designs.csv; drawn by 06)
* the accuracy values of SI S12, Results 2.1 and the Discussion (in_text_numbers.txt),
  e.g. balanced accuracy 0.645 (0.584-0.714; nine sites) for 250 m blocks with the
  200 m buffer against 0.895 for the random 80/20 split, the decomposition of that
  decline, the held-out favela component (0.651 / 0.629) and the pooled
  leave-site-out (0.695).

Scoring (Main Methods 'Validation under spatial separation'; SI S12 'Scoring';
``spatial_cv.scoring``): a fold is used only if its training set holds at least 50
buildings of the minority class; a site is scored in a design only if the used
folds give a prediction to at least half of its favela buildings (per grid
placement); reported value = mean over scored sites of the per-site mean over used
folds, 95% interval from 2,000 site-bootstrap resamples. The random 80/20 split
tests 20% of each site by design, so the second criterion is not applied to it.
Matched controls are scored on the site x placement units scored for the blocked
design (plus the first criterion on the control fold itself).

Inputs  outputs/appendix/S12_spatial_cv/folds_all.parquet, summary_loso.csv (step 04); site tables (step 01)
Outputs outputs/appendix/S12_spatial_cv/tables/table_R1_spatialcv_{designs,grid,sensitivity,coverage,folds}.tex
        outputs/appendix/S12_spatial_cv/accuracy/ scored_grid.csv, scored_designs.csv, table_S11_grid.csv,
          table_S12_sensitivity.csv, table_S13_coverage.csv, coverage_by_site_placement.csv,
          table_S14_folds.csv, decomposition.csv, in_text_numbers.txt
Run     python appendix/S12_spatial_cv/05_accuracy_tables.py
"""
import numpy as np
import pandas as pd

from favela_xai import spatial_cv as scv

import s12_common as C

cfg, s, P = C.setup(__file__)
SCV = cfg.section("spatial_cv")
OUT = P.out / "accuracy"
OUT.mkdir(exist_ok=True)
log = C.log_to(P.logs / "05_accuracy_tables.log")

BLOCKS = [int(v) for v in SCV["block_sizes_m"]]
BUFFERS = [int(v) for v in SCV["buffers_m"]]
PRIM_L, PRIM_B = int(SCV["primary"]["block_m"]), int(SCV["primary"]["buffer_m"])     # 250 m blocks, 200 m buffer
COMP_B = [int(v) for v in SCV["component_buffers_m"]]
LAB_L = {L: f"{L:,} m" for L in BLOCKS}

F = C.read_folds(cfg)
TOT = C.favela_totals(cfg, P)
LOSO = pd.read_csv(C.require(P.out / "summary_loso.csv", "04_collect_folds.py")) if len(cfg.sites) > 1 else None


def rows_of(design, buffer=None, control=False, placement=None):
    x = F[(F.design == design) & (F.control == control)]
    if buffer is not None:
        x = x[x.buffer == buffer]
    if placement is not None:
        x = x[x["repeat"] == placement]
    return x


def flat(prefix: str, summary: dict) -> dict:
    return {f"{prefix}{k}": v for k, v in summary.items()}


def cell_record(label, x, ctl=None, unit=("site", "repeat"), random_split=False, **extra):
    """Score one design cell; return (flat record, score dict)."""
    o = C.score_cell(x, TOT, s, ctl, unit=unit, random_split=random_split)
    r = dict(design=label, **extra, **flat("", o["both"]), **flat("first_", o["first"]), **o["counts"])
    if "ctl_both" in o:
        r.update(flat("control_", o["ctl_both"]))
        r.update(flat("control_first_", o["ctl_first"]))
    return r, o


def unscored_names(cov: pd.DataFrame) -> str:
    """Names of the sites that the second criterion removes in at least one placement."""
    return ", ".join(cfg.site_name(site) for site, g in cov.groupby("site") if (~g.scored).any())


covs = []
# ============================================================================ SI Table S11: square blocks x buffers
rows_A, recs_C = [], []
for L in BLOCKS:
    for b in BUFFERS:
        x, c = rows_of(f"sq{L}", b), rows_of(f"sq{L}", b, control=True)
        lab = f"Square {LAB_L[L]}, three placements"
        r, o = cell_record(lab, x, c if len(c) else None, block_m=L, buffer_m=b)
        rows_A.append(r)
        covs.append(o["cov"].assign(design=lab, buffer_m=b))
        recs_C.append(dict(r, unscored_names=unscored_names(o["cov"])))
for d, lab in (("random_80_20", "Random 80/20 split"), ("random_skf5", "Random five-fold")):
    r, o = cell_record(lab, rows_of(d), random_split=(d == "random_80_20"), block_m=np.nan, buffer_m=0)
    rows_A.append(r)
A = pd.DataFrame(rows_A)
A.to_csv(OUT / "table_S11_grid.csv", index=False)

body = ["\\toprule",
        "Buffer (m) & Sites & Folds & Prev. & Accuracy & Balanced accuracy & AUC & F1 (favela) & Matched control, bal.\\ acc. & First criterion only, bal.\\ acc. \\\\",
        "\\midrule"]


def first_txt(r):
    v = r["first_balanced_accuracy"]
    return "--" if not np.isfinite(v) else f"{v:.3f} ({int(r['first_n_sites'])})"


for L in BLOCKS:
    body.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{Square blocks, {LAB_L[L].replace(',', '{,}')}}}}} \\\\")
    for _, r in A[A.block_m == L].iterrows():
        ctl = C.ci3(r, "balanced_accuracy", "control_") if r.buffer_m > 0 else "--"
        body.append(f"{int(r.buffer_m)} & {int(r.n_sites)} & {int(r.n_used_final)}/{int(r.n_fold_rows)} & "
                    f"{C.f2(r.prev_test_mean)} & {C.ci3(r, 'accuracy')} & {C.ci3(r, 'balanced_accuracy')} & "
                    f"{C.ci3(r, 'auc')} & {C.ci3(r, 'f1_favela')} & {ctl} & {first_txt(r)} \\\\")
    body.append("\\addlinespace")
body.append("\\multicolumn{10}{l}{\\textit{Random references (no spatial separation)}} \\\\")
for _, r in A[A.block_m.isna()].iterrows():
    body.append(f"{r.design} & {int(r.n_sites)} & {int(r.n_used_final)}/{int(r.n_fold_rows)} & {C.f2(r.prev_test_mean)} & "
                f"{C.ci3(r, 'accuracy')} & {C.ci3(r, 'balanced_accuracy')} & {C.ci3(r, 'auc')} & "
                f"{C.ci3(r, 'f1_favela')} & -- & {first_txt(r)} \\\\")
body.append("\\bottomrule")
C.write_tex_table(
    P.tables / "table_R1_spatialcv_grid.tex", "05_accuracy_tables.py",
    "Within-site classification accuracy under buffered square spatial blocks. Sites: sites scored, of ten. Folds: "
    "used/fold rows over the three placements; the second scoring criterion is applied per placement. Prev.: mean "
    "favela share of the test set. First criterion only: balanced accuracy when only the first criterion is applied "
    "(number of sites in parentheses). Intervals: site-bootstrap 95\\%.",
    "tab:spatialcv_grid", body, cfg, tabular="rrrrlllll l")

# ============================================================================ SI Table S12: sensitivity designs
SENS = [("sq250", None, "Square 250 m, three placements (primary)"),
        ("sq250", 0, "Square 250 m, first placement"),
        ("sq500", 0, "Square 500 m, first placement"),
        ("hex250", 0, "Hexagon, 250 m"), ("hex500", 0, "Hexagon, 500 m"),
        ("kmm250", 0, "k-means, many clusters ($\\approx$250 m)"),
        ("kmm500", 0, "k-means, many clusters ($\\approx$500 m)"),
        ("km5", 0, "k-means, 5 clusters (= folds)"),
        ("sq250_K10", 0, "Square 250 m, $K$ = 10"), ("sq500_K10", 0, "Square 500 m, $K$ = 10")]
rows_B = []
for d, rep, lab in SENS:
    for b in (0, PRIM_B):
        x, c = rows_of(d, b, placement=rep), rows_of(d, b, control=True, placement=rep)
        units = x.drop_duplicates(["site", "repeat"]).n_units.mean() if x.n_units.notna().any() else np.nan
        r, o = cell_record(lab, x, c if (len(c) and b == PRIM_B) else None, key=d, placement=rep, buffer_m=b,
                           units_per_site=units)
        rows_B.append(r)
        if rep is not None:                               # the three-placement rows are listed with Table S11
            covs.append(o["cov"].assign(design=lab, buffer_m=b))
            recs_C.append(dict(r, unscored_names=unscored_names(o["cov"])))
comp_notes = {}
for b in COMP_B:
    x = rows_of("component", b)
    r, o = cell_record("Held-out favela component", x, unit=("site",), key="component", buffer_m=b, units_per_site=np.nan)
    rows_B.append(r)
    cov = o["cov"].assign(repeat=0)
    covs.append(cov.assign(design="Held-out favela component", buffer_m=b))
    recs_C.append(dict(r, unscored_names=unscored_names(cov)))
    # sentence of the Table S12 caption on the site that the second criterion removes
    notes = []
    for _, cv in cov[~cov.scored].iterrows():
        f = x[x.site == cv.site]
        excl = f[f.status.isin(scv.SCORED_STATUS) & ~scv.rule1_mask(f, s)]
        if len(excl) == 1 and len(f) == 2:
            e = excl.iloc[0]
            fav = int(round(e.n_train * e.prev_train))
            frm = int(e.n_train - fav)
            cls, n_min = ("formal-settlement", frm) if frm < fav else ("favela", fav)
            notes.append(f"{cfg.site_name(cv.site)} is not scored because one of its two folds keeps only {n_min} "
                         f"{cls} training buildings, so the remaining fold gives a prediction to "
                         f"{100 * cv.coverage:.1f}\\% of its favela buildings")
        else:
            notes.append(f"{cfg.site_name(cv.site)} is not scored (the used folds give a prediction to "
                         f"{100 * cv.coverage:.1f}\\% of its favela buildings)")
    comp_notes[b] = notes
n_testable = int(rows_of("component", COMP_B[0]).site.nunique())
if LOSO is not None:
    r, o = cell_record("Pooled leave-site-out", rows_of("loso_pooled"), key="loso_pooled", buffer_m=np.nan,
                       units_per_site=np.nan)
    rows_B.append(r)
    covs.append(o["cov"].assign(design="Pooled leave-site-out", buffer_m=np.nan))
B = pd.DataFrame(rows_B)
B.to_csv(OUT / "table_S12_sensitivity.csv", index=False)

body = ["\\toprule",
        "Design & Buffer (m) & Units & Sites & Folds & Prev. & Acc. & Balanced accuracy & AUC & F1 (fav.) & Matched control, bal.\\ acc. & First criterion only, bal.\\ acc. \\\\",
        "\\midrule"]
for _, r in B.iterrows():
    bt = "held-out site" if not np.isfinite(r.buffer_m) else str(int(r.buffer_m))
    u = "--" if not np.isfinite(r.units_per_site) else f"{r.units_per_site:.1f}"
    ctl = C.ci3(r, "balanced_accuracy", "control_") if np.isfinite(r.get("control_balanced_accuracy", np.nan)) else "--"
    body.append(f"{r.design} & {bt} & {u} & {int(r.n_sites)} & {int(r.n_used_final)}/{int(r.n_fold_rows)} & "
                f"{C.f2(r.prev_test_mean)} & {C.f3(r.accuracy)} & {C.ci3(r, 'balanced_accuracy')} & {C.ci3(r, 'auc')} & "
                f"{C.f3(r.f1_favela)} & {ctl} & {first_txt(r)} \\\\")
    if not np.isfinite(r.buffer_m) or r.buffer_m == PRIM_B:
        body.append("\\addlinespace[2pt]")
body.append("\\bottomrule")
note200 = "; ".join(comp_notes.get(PRIM_B, []))
cap_S12 = ("Sensitivity of the spatial cross-validation to block shape, block-assignment rule and number of folds, and two "
           "grouped hold-outs. Designs other than the primary one use the first grid placement. Values as in "
           "Table~\\ref{tab:spatialcv_grid} (site-bootstrap 95\\% intervals). Units: spatial units per site. Held-out "
           f"favela component: {n_testable} sites with at least two groups"
           + (f"; with the {PRIM_B} m buffer, {note200}." if note200 else "."))
C.write_tex_table(P.tables / "table_R1_spatialcv_sensitivity.tex", "05_accuracy_tables.py", cap_S12,
                  "tab:spatialcv_sens", body, cfg, tabular="lrrrrrrlllll")

# ============================================================================ SI Table S13: folds and sites removed
Cc = pd.DataFrame(recs_C)
prim_lab = f"Square {LAB_L[PRIM_L]}, three placements"
listed = Cc[(Cc.n_excluded_first > 0) | (Cc.unscored_names != "") | ((Cc.design == prim_lab) & (Cc.buffer_m == PRIM_B))]
Cc.to_csv(OUT / "table_S13_coverage.csv", index=False)
body = ["\\toprule",
        "Design & Buffer (m) & Fold rows & Not fitted & Excluded by first criterion & Used & Sites not scored & Test buildings scored \\\\",
        "\\midrule"]
for _, r in listed.iterrows():
    body.append(f"{r.design} & {int(r.buffer_m)} & {int(r.n_fold_rows)} & {int(r.n_not_fitted)} & "
                f"{int(r.n_excluded_first)} & {int(r.n_used_final)} & {r.unscored_names or '--'} & "
                f"{100 * r.share_test_scored:.1f}\\% \\\\")
body.append("\\bottomrule")
C.write_tex_table(
    P.tables / "table_R1_spatialcv_coverage.tex", "05_accuracy_tables.py",
    "Folds and sites removed by the two scoring criteria. Folds not fitted: no test building, fewer than 50 training "
    "buildings, or a single class in the training set after the buffer. Sites not scored: the sites that the second "
    "criterion removes in that design and buffer (for designs with three grid placements, per placement), listed by "
    "name. "
    "Test buildings scored: share of all test buildings, over all placements, that enter the scored values. Design "
    "cells not listed lose no fold and no site; the matched controls are scored on the same site placements.",
    "tab:spatialcv_coverage", body, cfg, tabular="lrrrrrp{4.5cm}r")
COV = pd.concat(covs, ignore_index=True)
COV["site_name"] = COV.site.map(cfg.site_name)
COV["grid_placement"] = COV["repeat"].fillna(0).astype(int) + 1
COV[["design", "buffer_m", "site_name", "grid_placement", "fav_covered", "fav_total", "coverage", "scored"]].rename(
    columns={"fav_covered": "favela_buildings_with_prediction", "fav_total": "favela_buildings_in_site",
             "coverage": "favela_coverage", "scored": "site_scored"}).to_csv(OUT / "coverage_by_site_placement.csv",
                                                                            index=False)

# ============================================================================ SI Table S14: fold-level summary
Bl = F[~F.control]
STRIPS = [("Random five-fold", Bl[Bl.design == "random_skf5"], ("site", "repeat")),
          (f"Blocks {PRIM_L}\\,m, no buffer", Bl[(Bl.design == f"sq{PRIM_L}") & (Bl.buffer == 0)], ("site", "repeat")),
          (f"Blocks {PRIM_L}\\,m + {PRIM_B}\\,m buffer (primary design)",
           Bl[(Bl.design == f"sq{PRIM_L}") & (Bl.buffer == PRIM_B)], ("site", "repeat")),
          (f"Blocks 1{{,}}000\\,m + {PRIM_B}\\,m buffer", Bl[(Bl.design == "sq1000") & (Bl.buffer == PRIM_B)], ("site", "repeat")),
          ("Held-out favela component, no buffer", Bl[(Bl.design == "component") & (Bl.buffer == 0)], ("site",)),
          (f"Held-out favela component + {PRIM_B}\\,m buffer", Bl[(Bl.design == "component") & (Bl.buffer == PRIM_B)],
           ("site",)),
          ("Held-out site (pooled leave-site-out)", Bl[Bl.design == "loso_pooled"], ("site", "repeat"))]
rows_D = []
for lab, g, unit in STRIPS:
    if not len(g):
        continue
    kept, _ = scv.apply_rules(g, TOT, s, unit=unit)
    rows_D.append(dict(design=lab, scored=len(kept), not_scored=len(g) - len(kept), sites=int(kept.site.nunique()),
                       mean_site_means=float(kept.groupby("site").balanced_accuracy.mean().mean()),
                       median_fold=float(np.median(kept.balanced_accuracy)),
                       median_prev_test=float(np.median(kept.prev_test))))
D = pd.DataFrame(rows_D)
D.to_csv(OUT / "table_S14_folds.csv", index=False)
body = ["\\toprule",
        " & \\multicolumn{2}{c}{Folds} & \\multicolumn{2}{c}{Balanced accuracy} & Median test favela \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Design & Scored & Not scored & Mean of site means & Median fold & prevalence \\\\",
        "\\midrule"]
body += [f"{r.design} & {r.scored} & {r.not_scored} & {C.tex_dot(r.mean_site_means)} & {C.tex_dot(r.median_fold)} & "
         f"{C.tex_dot(r.median_prev_test)} \\\\" for r in D.itertuples()]
body.append("\\bottomrule")
C.write_tex_table(
    P.tables / "table_R1_spatialcv_folds.tex", "05_accuracy_tables.py",
    "\\textbf{Fold-level balanced accuracy of the spatial validation designs.} One row per design; a\nfold is one "
    "held-out unit of that design. Block designs contribute the folds of their three grid placements.\n\\emph{Folds "
    "scored} counts the folds that pass both scoring criteria and \\emph{not scored} those removed\nby them, including "
    "folds that could not be fitted. \\emph{Mean of site means} is the mean over scored sites\nof each site's mean over "
    "its used folds; it weights sites equally rather than folds. \\emph{Median fold} is\nthe median over the scored "
    "folds themselves, which weights folds equally instead; the two differ where the\nfold distribution is skewed or "
    "where sites contribute unequal numbers of folds. \\emph{Median test favela\nprevalence} is the median over the "
    "scored folds of the share of the\nfold's test buildings labelled favela in the official 2013 boundaries; balanced "
    "accuracy is insensitive to it by\nconstruction, and 0.5 is its chance level at any prevalence. "
    "The individual folds, with their metrics, prevalence and sample sizes, are\nlisted in Supplementary Data~1 and~2.",
    "tab:spatialcv_folds", body, cfg, size="\\small", tabular="@{}lrrrrr@{}")

# ============================================================================ SI Fig. S19 values
grid = []
for r in rows_A:
    base = dict(design="square" if np.isfinite(r["block_m"]) else ("random_80_20" if r["design"].startswith("Random 80") else
                                                                  "random_skf5"),
                L=r["block_m"], buffer=r["buffer_m"])
    grid.append(dict(base, control=False, n_fits=r["n_fold_rows"], **{k: v for k, v in r.items()
                     if not k.startswith(("control_", "first_")) and k not in ("design", "block_m", "buffer_m")}))
    if "control_balanced_accuracy" in r and np.isfinite(r.get("control_balanced_accuracy", np.nan)):
        grid.append(dict(base, control=True, **{k[len("control_"):]: v for k, v in r.items()
                         if k.startswith("control_") and not k.startswith("control_first_")}))
G = pd.DataFrame(grid)
G.to_csv(OUT / "scored_grid.csv", index=False)
KEY = {("sq250", None): "main250", ("sq500", 0): "sq500_r0_reference", ("hex250", 0): "hex250", ("hex500", 0): "hex500",
       ("kmm250", 0): "kmm250", ("kmm500", 0): "kmm500", ("sq250_K10", 0): "sq250_K10", ("sq500_K10", 0): "sq500_K10",
       ("km5", 0): "km5", ("component", None): "component", ("loso_pooled", None): "loso"}
des = []
for r in rows_B:
    k = KEY.get((r["key"], r.get("placement")))
    if k is None:
        continue
    b = 0 if not np.isfinite(r["buffer_m"]) else int(r["buffer_m"])
    des.append(dict(row_key=k, buffer=b, control=False, n_sites=r["n_sites"], n_folds=r["n_folds"],
                    balanced_accuracy=r["balanced_accuracy"], balanced_accuracy_ci_lo=r["balanced_accuracy_ci_lo"],
                    balanced_accuracy_ci_hi=r["balanced_accuracy_ci_hi"]))
    if np.isfinite(r.get("control_balanced_accuracy", np.nan)):
        des.append(dict(row_key=k, buffer=b, control=True, n_sites=r["control_n_sites"], n_folds=r["control_n_folds"],
                        balanced_accuracy=r["control_balanced_accuracy"],
                        balanced_accuracy_ci_lo=r["control_balanced_accuracy_ci_lo"],
                        balanced_accuracy_ci_hi=r["control_balanced_accuracy_ci_hi"]))
pd.DataFrame(des).to_csv(OUT / "scored_designs.csv", index=False)

# ============================================================================ SI Table S10: overview of the designs
Ai = A[A.block_m.notna()].set_index(["block_m", "buffer_m"])   # unique index: the random references have no block size
Bi = {(r["key"], r.get("placement"), r["buffer_m"] if np.isfinite(r["buffer_m"]) else None): r for r in rows_B}
bufs = f"{min(BUFFERS)}--{max(BUFFERS)} ({PRIM_B})"
sens_bufs = ", ".join(str(int(v)) for v in SCV["sensitivity_buffers_m"]) + f" ({PRIM_B})"
comp_bufs = ", ".join(str(v) for v in COMP_B) + f" ({PRIM_B})"
ctl_bufs = f"{min(b for b in BUFFERS if b > 0)}--{max(BUFFERS)} ({PRIM_B})"


def s10_row(name, held, buf, folds, r, pre=""):
    ba = (f"{C.tex_dot(r[pre + 'balanced_accuracy'])} ({C.tex_dot(r[pre + 'balanced_accuracy_ci_lo'])}--"
          f"{C.tex_dot(r[pre + 'balanced_accuracy_ci_hi'])})")
    return [name, f"  & {held}", f"  & {buf}", f"  & {folds}", f"  & {int(r[pre + 'n_sites'])}", f"  & {ba} \\\\"]


groups = []
prim = []
for L in BLOCKS:
    r = Ai.loc[(L, PRIM_B)]
    name = f"Square blocks, {LAB_L[L].replace(',', '{,}').replace(' m', '')}\\,m" + (" (primary)" if L == PRIM_L else "")
    prim.append(s10_row(name, "Square grid block", bufs, int(r.n_fold_rows), r))
r = Bi[("component", None, PRIM_B)]
prim.append(s10_row("Held-out favela component", "A group of neighbouring 2013 favela polygons", comp_bufs,
                    int(r["n_fold_rows"]), r))
if LOSO is not None:
    r = Bi[("loso_pooled", None, None)]
    prim.append(s10_row("Pooled leave-site-out", "An entire study site", "---", int(r["n_fold_rows"]), r))
groups.append((f"Primary ({len(prim)})", prim))
held = {"hex": "Hexagon of the area of a {L}\\,m square", "kmm": "$k$-means cluster of building centroids",
        "km5": "One of five $k$-means clusters, used as a fold", "sq": "Square grid block"}
sens = []
for d, lab in (("hex250", "Hexagon, 250\\,m"), ("hex500", "Hexagon, 500\\,m"),
               ("kmm250", "$k$-means, many clusters ($\\approx$250\\,m)"),
               ("kmm500", "$k$-means, many clusters ($\\approx$500\\,m)"), ("km5", "$k$-means, 5 clusters (= folds)"),
               ("sq250_K10", "Square 250\\,m, $K$ = 10"), ("sq500_K10", "Square 500\\,m, $K$ = 10")):
    r = Bi[(d, 0, PRIM_B)]
    h = held["km5" if d == "km5" else d.rstrip("0123456789_K")].format(L=d[3:])
    sens.append(s10_row(lab, h, sens_bufs, int(r["n_fold_rows"]), r))
groups.append((f"Sensitivity ({len(sens)})", sens))
r = Ai.loc[(PRIM_L, PRIM_B)]
groups.append(("Control (1)", [s10_row("Matched control", "Same test blocks; buffer buildings removed at random",
                                       ctl_bufs, "---", r, pre="control_")]))
refs = [s10_row("Random 80/20 split", "20\\% of each site's buildings, at random", "---",
                int(A[A.design == "Random 80/20 split"].iloc[0].n_fold_rows), A[A.design == "Random 80/20 split"].iloc[0]),
        s10_row("Random five-fold", "One of five random stratified folds", "---",
                int(A[A.design == "Random five-fold"].iloc[0].n_fold_rows), A[A.design == "Random five-fold"].iloc[0])]
groups.append(("Reference, no spatial separation (2)", refs))
body = ["\\toprule", "Design & Held out & Buffer (m) & Folds & Sites scored & Balanced accuracy (95\\% CI) \\\\", "\\midrule"]
for gi, (glab, rr) in enumerate(groups):
    if gi:
        body.append("\\midrule")
    body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{glab}}}}} \\\\")
    for i, lines in enumerate(rr):
        body += lines
        if i < len(rr) - 1:
            body.append("\\addlinespace[0.5pt]")
body.append("\\bottomrule")
n_designs = sum(len(rr) for _, rr in groups)
km = (f"{C.tex_dot(LOSO.nearest_train_site_dist.min() / 1000, 1)}--{C.tex_dot(LOSO.nearest_train_site_dist.max() / 1000, 1)}"
      if LOSO is not None else "--")
cap = ("\\textbf{The spatial validation designs.} All sixteen designs used in this section, grouped by role. "
       "\\emph{Buffer}: distances at which the design was run, the one reported here in parentheses; the pooled "
       f"leave-site-out needs no buffer, because held-out sites lie {km}\\,km from the nearest training site. "
       "\\emph{Folds}: fold rows contributed; the matched control re-uses the folds of the design it matches. "
       "\\emph{Sites scored}: of ten, at the buffer reported here. \\emph{Balanced accuracy}: site-bootstrap 95\\% "
       "interval in parentheses.")
spec = ("@{}>{\\raggedright\\arraybackslash}p{3.5cm}>{\\raggedright\\arraybackslash}p{4.55cm}"
        ">{\\raggedright\\arraybackslash}p{1.75cm}>{\\raggedright\\arraybackslash}p{0.95cm}"
        ">{\\raggedright\\arraybackslash}p{1.2cm}>{\\raggedright\\arraybackslash}p{3.95cm}@{}")
C.write_tex_table(P.tables / "table_R1_spatialcv_designs.tex", "05_accuracy_tables.py", cap, "tab:spatialcv_designs",
                  body, cfg, resize=False, preamble=["\\singlespacing", "\\setlength{\\tabcolsep}{3pt}"], tabular=spec,
                  requires="booktabs,array,setspace")
log(f"SI Table S10: {n_designs} designs (manuscript: sixteen)")

# ============================================================================ decomposition (SI S12 'Classification accuracy')
b200 = rows_of(f"sq{PRIM_L}", PRIM_B)
ctl200 = rows_of(f"sq{PRIM_L}", PRIM_B, control=True)
b0 = rows_of(f"sq{PRIM_L}", 0)
rnd = rows_of("random_80_20")
kept200, cov200 = scv.apply_rules(b200, TOT, s)
units = cov200.loc[cov200.scored, ["site", "repeat"]]
SETS = {"both criteria (primary)": dict(R=rnd[scv.rule1_mask(rnd, s)],
                                        B0=scv.apply_rules(b0, TOT, s)[0].merge(units, on=["site", "repeat"]),
                                        C=scv.apply_control(ctl200, cov200, s), B200=kept200),
        "first criterion only (sensitivity)": dict(R=rnd[scv.rule1_mask(rnd, s)], B0=b0[scv.rule1_mask(b0, s)],
                                                   C=ctl200[scv.rule1_mask(ctl200, s)], B200=b200[scv.rule1_mask(b200, s)])}
dec = []
for scoring, S_ in SETS.items():
    sites = sorted(set(S_["B200"].site))
    for col in ("balanced_accuracy", "accuracy", "auc"):
        sm = {k: v.groupby("site")[col].mean().reindex(sites) for k, v in S_.items()}
        comp = {"block_hold_out": sm["B0"] - sm["R"], "smaller_training_set": sm["C"] - sm["B0"],
                "lost_neighbouring_context": sm["B200"] - sm["C"], "total": sm["B200"] - sm["R"]}
        bt = {k: C.boot_mean(v.to_numpy(), s) for k, v in comp.items()}
        r = dict(scoring=scoring, metric=col, n_sites=len(sites), sites=";".join(sites),
                 **{f"step_{k}": float(sm[k].mean()) for k in ("R", "B0", "C", "B200")},
                 **{f"folds_{k}": int(S_[k].site.isin(sites).sum()) for k in ("R", "B0", "C", "B200")})
        for k, t in bt.items():
            r.update({k: t[0], f"{k}_ci_lo": t[1], f"{k}_ci_hi": t[2]})
        r["spatial_share"] = (bt["block_hold_out"][0] + bt["lost_neighbouring_context"][0]) / bt["total"][0]
        r["smaller_training_set_share"] = bt["smaller_training_set"][0] / bt["total"][0]
        dec.append(r)
DEC = pd.DataFrame(dec)
DEC.to_csv(OUT / "decomposition.csv", index=False)

# ============================================================================ in-text numbers
Ai = A[A.block_m.notna()].set_index(["block_m", "buffer_m"])   # unique index: the random references have no block size
p0, p2 = Ai.loc[(PRIM_L, 0)], Ai.loc[(PRIM_L, PRIM_B)]
rA = A[A.design == "Random 80/20 split"].iloc[0]
rF = A[A.design == "Random five-fold"].iloc[0]
d = DEC[(DEC.scoring.str.startswith("both")) & (DEC.metric == "balanced_accuracy")].iloc[0]
at200 = [r for r in rows_B if r["buffer_m"] == PRIM_B and r["key"] not in ("component",)]
at200 += [dict(Ai.loc[(L, PRIM_B)]) for L in BLOCKS if L >= PRIM_L and L != PRIM_L]
vals = [r["balanced_accuracy"] for r in at200]
ctls = [r["control_balanced_accuracy"] for r in at200 if np.isfinite(r.get("control_balanced_accuracy", np.nan))]
cp = COV[(COV.design == f"Square {LAB_L[PRIM_L]}, three placements") & (COV.buffer_m == PRIM_B) & ~COV.scored]
comp = {b: Bi[("component", None, b)] for b in COMP_B}
lines = [
    "SI S12 in-text numbers (both scoring criteria unless stated)",
    f"250 m blocks, no buffer: {C.ci3(p0, 'balanced_accuracy')}; {int(p0.n_sites)} sites, {int(p0.n_used_final)} folds "
    "(manuscript 0.772 (0.727-0.822); ten sites, 150 folds)",
    f"250 m blocks, 200 m buffer: {C.ci3(p2, 'balanced_accuracy')}; {int(p2.n_sites)} sites, {int(p2.n_used_final)} of "
    f"{int(p2.n_fold_rows)} folds; first criterion alone {p2.first_balanced_accuracy:.3f} over {int(p2.first_n_sites)} sites "
    "(manuscript 0.645 (0.584-0.714); nine sites, 129 of 150; 0.643 over ten)",
    f"sites not scored under the primary design: {', '.join(sorted(set(cp.site_name)))}; favela coverage per placement "
    + ", ".join(f"{100 * v:.0f}%" for v in cp.sort_values('grid_placement').coverage) + " (manuscript Palmeiras 4-11%)",
    f"random 80/20 split (refitted): {C.ci3(rA, 'balanced_accuracy')}, AUC {rA.auc:.3f}; random five-fold "
    f"{C.ci3(rF, 'balanced_accuracy')}, AUC {rF.auc:.3f} (manuscript 0.895 / 0.966, 0.904 / 0.970)",
    f"other block sizes >= {PRIM_L} m, shapes and rules at {PRIM_B} m: {min(vals):.3f}-{max(vals):.3f}; matched controls "
    f"{min(ctls):.3f}-{max(ctls):.3f} (manuscript 0.645-0.699 against 0.708-0.789)",
    f"decomposition over {int(d.n_sites)} sites: {d.step_R:.3f} -> {d.step_B0:.3f} -> {d.step_C:.3f} -> {d.step_B200:.3f}; "
    f"smaller training set {d.smaller_training_set:+.3f} ({d.smaller_training_set_ci_lo:+.3f} to "
    f"{d.smaller_training_set_ci_hi:+.3f}); spatial share {100 * d.spatial_share:.0f}% of the total {d.total:+.3f}; "
    f"block hold-out {d.block_hold_out:+.3f}; lost neighbouring context {d.lost_neighbouring_context:+.3f} "
    "(manuscript 0.903 -> 0.785 -> 0.779 -> 0.645; -0.006 (-0.010 to -0.003); 98% of 0.258; -0.118; -0.134)",
] + [f"held-out favela component, {b} m buffer: {C.ci3(comp[b], 'balanced_accuracy')}; {int(comp[b]['n_sites'])} sites, "
     f"{int(comp[b]['n_used_final'])} groups" for b in COMP_B] + [
    "(manuscript 0.651 (0.578-0.729), eight sites, 58 groups; 0.629 (0.577-0.679), seven sites, 56 groups)"]
if LOSO is not None:
    rl = Bi[("loso_pooled", None, None)]
    lines.append(f"pooled leave-site-out: {C.ci3(rl, 'balanced_accuracy')}, AUC {rl['auc']:.3f} "
                 "(manuscript 0.695 (0.618-0.781), AUC 0.781)")
(OUT / "in_text_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
for ln in lines:
    log(ln)
log("done")
