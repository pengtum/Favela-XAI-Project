"""Step 7: distance between test and training buildings; SI Table S15 and the data of SI Fig. S20.

Manuscript items
* SI Table S15 -> tables/table_R1_spatialcv_distances.tex:
  distance from each test building to the nearest building of the training set used
  in its own fit, first grid placement of each block design, distances after the
  buffer, sites pooled with equal weight per site; scoring as in the accuracy tables
  (both criteria), with the median under the first criterion alone beside it.
* SI S12 'Distance between test and training buildings' in-text numbers: site-weighted
  median 5.8 m (random 80/20 split), 51.6 m (250 m blocks, no buffer), 279.5 m (primary
  design; 5th-95th percentile 209-453 m, minimum 200 m), 5.1 km (held-out site).
* SI Fig. S20 data (drawn by 08):
  a  site-weighted cumulative distributions for the designs random five-fold, 100-1,000 m
     blocks without buffer, 250 m blocks with the 200 m buffer and five k-means folds
     (scored folds only), and three prediction tasks, which are not scored: T1 the test
     set of a random 80/20 split, here the split of the main per-site models (Main Methods
     4.4: stratified, seed 42), T2 a held-out favela component (every group), T3 a
     held-out site with the other nine sites as training data;
  b  error rate of the out-of-fold predictions by distance to the nearest training
     building (random five-fold; 100-1,000 m blocks without and with the 200 m buffer;
     first grid placement), buildings pooled over sites, Wilson 95% intervals and a
     site-bootstrap 95% interval (2,000 resamples) for the unbuffered block designs
     pooled; distance bins of ``spatial_cv.distances.bin_edges_m``.

Inputs  outputs/appendix/S12_spatial_cv/folds_all.parquet, predictions_oof.parquet (step 04);
        site_tables/ (step 01); component_groups/ (step 02);
        outputs/main/02_classification/<site>_split.csv (products.split_csv; task T1)
Outputs outputs/appendix/S12_spatial_cv/tables/table_R1_spatialcv_distances.tex
        outputs/appendix/S12_spatial_cv/distances/ distances_by_design.csv, ecdf_distances.parquet,
          error_by_distance.csv, error_by_distance_site.csv, in_text_numbers.txt
Run     python appendix/S12_spatial_cv/07_distances.py
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from favela_xai import spatial_cv as scv
from favela_xai.model import main_split
from favela_xai.stats import pooled_rate_bootstrap, site_bootstrap_draws, wilson_interval

import s12_common as C

cfg, s, P = C.setup(__file__)
SCV = cfg.section("spatial_cv")
OUT = P.out / "distances"
OUT.mkdir(exist_ok=True)
log = C.log_to(P.logs / "07_distances.log")
BLOCKS = [int(v) for v in SCV["block_sizes_m"]]
PRIM_L, PRIM_B = int(SCV["primary"]["block_m"]), int(SCV["primary"]["buffer_m"])
SITES = list(cfg.sites)

F = C.read_folds(cfg)
TOT = C.favela_totals(cfg, P)
PR = C.read_predictions(P, columns=["site", "design", "repeat", "buffer", "fold", "row_id", "d_nn_train", "misclassified"])
F0 = F[(F["repeat"] == 0) & (~F.control)]


def fold_keys(design, buffer=None, unit=("site", "repeat")):
    """(site, fold) pairs used under both criteria and under the first criterion (first grid placement)."""
    x = F0[F0.design == design]
    if buffer is not None:
        x = x[x.buffer == buffer]
    kept, _ = scv.apply_rules(x, TOT, s, unit=list(unit))
    first = x[scv.rule1_mask(x, s)]
    if design in ("random_80_20", "loso_pooled"):     # one split per site / every site covered: criterion 2 does not apply
        kept = first
    return set(zip(kept.site, kept.fold)), set(zip(first.site, first.fold)), x


def pred_rows(design, buffer=None):
    q = PR[(PR.design == design) & (PR["repeat"] == 0)]
    if buffer is not None:
        q = q[q.buffer == buffer]
    return q


def in_keys(q, keys):
    return np.array([(a, b) in keys for a, b in zip(q.site, q.fold)], dtype=bool)


def wstats(site, d) -> dict:
    """Statistics of distances pooled over sites with equal weight per site (each building weighted 1/n_site)."""
    df = pd.DataFrame({"site": np.asarray(site), "d": np.asarray(d, float)})
    parts = [g.d.to_numpy() for _, g in df.groupby("site") if len(g)]
    if not parts:
        return dict(n_sites=0)
    dd = np.concatenate(parts)
    w = np.concatenate([np.full(len(p), 1.0 / len(p)) for p in parts])
    o = np.argsort(dd)
    v = dd[o]
    cw = np.cumsum(w[o]) / w.sum()

    def q(x):
        return float(v[min(np.searchsorted(cw, x), len(v) - 1)])

    return dict(n_sites=len(parts), median_m=q(0.5), p05_m=q(0.05), p95_m=q(0.95), min_m=float(dd.min()),
                share_le_25m=float(w[dd <= 25].sum() / w.sum()), share_lt_200m=float(w[dd < 200].sum() / w.sum()))


# ============================================================================ SI Table S15
ROWS = [("No spatial separation", "Random 80/20 split", "random_80_20", 0),
        ("No spatial separation", "Random five-fold", "random_skf5", 0)]
ROWS += [("Spatial blocks, no buffer", f"Blocks {L:,} m", f"sq{L}", 0) for L in BLOCKS]
ROWS += [("Spatial blocks, no buffer", "Five k-means clusters", "km5", 0)]
for b in (PRIM_B, max(int(v) for v in SCV["buffers_m"])):
    ROWS += [(f"Spatial blocks, {b} m buffer",
              f"Blocks {L:,} m + {b} m buffer" + (" (primary)" if (L, b) == (PRIM_L, PRIM_B) else ""), f"sq{L}", b)
             for L in BLOCKS]
ROWS += [("Grouped hold-outs", "Held-out favela component", "component", 0),
         ("Grouped hold-outs", f"Held-out favela component + {PRIM_B} m buffer", "component", PRIM_B)]
if len(SITES) > 1:
    ROWS += [("Grouped hold-outs", "Held-out site (pooled leave-site-out)", "loso_pooled", None)]
rows = []
for grp, lab, design, b in ROWS:
    unit = ("site",) if design == "component" else ("site", "repeat")
    kk, k1, x = fold_keys(design, b if design != "loso_pooled" else None, unit)
    q = pred_rows(design, b if design != "loso_pooled" else None)
    both, first = q[in_keys(q, kk)], q[in_keys(q, k1)]
    st, s1 = wstats(both.site, both.d_nn_train), wstats(first.site, first.d_nn_train)
    rows.append(dict(group=grp, design=lab, key=design, buffer=b, **st, first_n_sites=s1.get("n_sites", 0),
                     first_median_m=s1.get("median_m", np.nan), share_test_buildings_scored=len(both) / max(x.n_test.sum(), 1)))
T = pd.DataFrame(rows)
T.to_csv(OUT / "distances_by_design.csv", index=False)


def f1(v):
    return "--" if not np.isfinite(v) else f"{v:,.1f}".replace(",", "{,}")


def f0(v):
    return "--" if not np.isfinite(v) else f"{v:,.0f}".replace(",", "{,}")


body = ["\\toprule",
        "Design & Sites & Median (m) & 5th--95th percentile (m) & Minimum (m) & $\\leq$25 m & $<$200 m & First criterion only, median (m) \\\\",
        "\\midrule"]
for grp, g in T.groupby("group", sort=False):
    body.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{grp}}}}} \\\\")
    for r in g.itertuples():
        lab = r.design.replace(",", "{,}")
        fc = f"{f1(r.first_median_m)} ({int(r.first_n_sites)})"
        if r.n_sites == 0:
            body.append(f"{lab} & 0 & -- & -- & -- & -- & -- & {fc} \\\\")
            continue
        body.append(f"{lab} & {int(r.n_sites)} & {f1(r.median_m)} & {f1(r.p05_m)}--{f0(r.p95_m)} & {f1(r.min_m)} & "
                    f"{100 * r.share_le_25m:.1f}\\% & {100 * r.share_lt_200m:.1f}\\% & {fc} \\\\")
    body.append("\\addlinespace")
body.append("\\bottomrule")
C.write_tex_table(
    P.tables / "table_R1_spatialcv_distances.tex", "07_distances.py",
    "Distance from each test building to the nearest training building used in its fit. First grid placement of each "
    "block design; distances after the buffer. Statistics pool the sites with equal weight per site. Scoring as in the "
    "accuracy tables. First criterion only: median when only the first criterion is applied (number of sites in "
    "parentheses).",
    "tab:spatialcv_distances", body, cfg, tabular="lrrrrrrl")

# ============================================================================ SI Fig. S20a: distances per curve
curves = []
ECDF = [("random5_strat", "random_skf5", 0)] + [(f"grid{L}_b0", f"sq{L}", 0) for L in BLOCKS] + \
       [(f"grid{PRIM_L}_b{PRIM_B}", f"sq{PRIM_L}", PRIM_B), ("kmeans5_b0", "km5", 0)]
for key, design, b in ECDF:
    kk, _, _ = fold_keys(design, b)
    q = pred_rows(design, b)
    q = q[in_keys(q, kk)]
    curves.append(pd.DataFrame({"site": q.site.to_numpy(), "curve": key, "d": q.d_nn_train.to_numpy(float)}))
# prediction tasks (geometry only, every building of the task; not scored)
xy_all = {site: C.load_site(P, site).xy for site in SITES}
for site in SITES:
    d = C.load_site(P, site)
    # T1: test set of the 80/20 split of the main per-site model (main/02_classification, products.split_csv;
    # recomputed with the Methods 4.4 rule if that file is not there)
    split_file = cfg.product("split_csv", site=site)
    if split_file.exists():
        is_test = pd.read_csv(split_file).set_index("uID").is_test.reindex(d.table.uID).to_numpy(bool)
        tr, te = np.where(~is_test)[0], np.where(is_test)[0]
    else:
        tr, te = main_split(d.y, cfg)
    curves.append(pd.DataFrame({"site": site, "curve": "T1_random80_20",
                                "d": cKDTree(d.xy[tr]).query(d.xy[te], k=1)[0]}))
    splits, codes, _ = C.component_splits(C.load_groups(P, site))  # T2: every held-out favela component
    for _, trn, tst in splits:
        curves.append(pd.DataFrame({"site": site, "curve": "T2_component_holdout",
                                    "d": cKDTree(d.xy[trn]).query(d.xy[tst], k=1)[0]}))
    if len(SITES) > 1:                                              # T3: held-out site, nine training sites
        other = np.vstack([xy_all[o] for o in SITES if o != site])
        curves.append(pd.DataFrame({"site": site, "curve": "T3_leave_site_out",
                                    "d": cKDTree(other).query(d.xy, k=1)[0]}))
E = pd.concat(curves, ignore_index=True)
E.to_parquet(OUT / "ecdf_distances.parquet", index=False)
log("sites per curve: " + ", ".join(f"{k} {v}" for k, v in E.groupby("curve", sort=False).site.nunique().items()))

# ============================================================================ SI Fig. S20b: error rate by distance
edges = [float(v) for v in SCV["distances"]["bin_edges_m"]] + [np.inf]
labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-2], edges[1:-1])] + [f">{edges[-2]:g}"]
DECAY = [("random_skf5", "random_skf5", 0)] + [(f"sq{L}_b{b}", f"sq{L}", b) for b in (0, PRIM_B) for L in BLOCKS]
parts = []
for key, design, b in DECAY:
    kk, _, _ = fold_keys(design, b)
    q = pred_rows(design, b)
    q = q[in_keys(q, kk)]
    parts.append(pd.DataFrame({"dkey": key, "site": q.site.to_numpy(), "d": q.d_nn_train.to_numpy(float),
                               "err": q.misclassified.to_numpy(bool)}))
Q = pd.concat(parts, ignore_index=True)
Q["bin"] = pd.cut(Q.d, edges, right=False, labels=labels)
cnt = Q.groupby(["dkey", "site", "bin"], observed=False).agg(n=("err", "size"), err=("err", "sum")).reset_index()
cnt.to_csv(OUT / "error_by_distance_site.csv", index=False)
B_, SEED = s.bootstrap_resamples, s.bootstrap_seed          # 2,000 site resamples (spatial_cv.scoring)


def decay_rows(c, scope):
    """Pooled error rate per bin with Wilson and site-bootstrap intervals (fresh draws per design)."""
    N = c.pivot_table(index="site", columns="bin", values="n", aggfunc="sum", observed=False).reindex(SITES).fillna(0)
    K = c.pivot_table(index="site", columns="bin", values="err", aggfunc="sum", observed=False).reindex(SITES).fillna(0)
    N, K = N[labels].to_numpy(float), K[labels].to_numpy(float)
    br = pooled_rate_bootstrap(N, K, site_bootstrap_draws(len(SITES), B_, SEED))
    n, k = N.sum(0), K.sum(0)
    wlo, whi = wilson_interval(k, n)
    out = []
    for j, bl in enumerate(labels):
        ok = np.isfinite(br[:, j])
        out.append(dict(design=scope, bin=bl, d_lo=edges[j], d_hi=edges[j + 1], n=int(n[j]), n_err=int(k[j]),
                        error_rate=k[j] / n[j] if n[j] else np.nan, wilson_lo=wlo[j], wilson_hi=whi[j],
                        boot_lo=float(np.percentile(br[ok, j], 2.5)) if ok.any() and n[j] else np.nan,
                        boot_hi=float(np.percentile(br[ok, j], 97.5)) if ok.any() and n[j] else np.nan,
                        n_sites=int((N[:, j] > 0).sum())))
    return out


drows = []
for dk, c in cnt.groupby("dkey", sort=False):
    drows += decay_rows(c, dk)
drows += decay_rows(cnt[cnt.dkey.str.startswith("sq") & cnt.dkey.str.endswith("_b0")], "pooled_square_b0")
pd.DataFrame(drows).to_csv(OUT / "error_by_distance.csv", index=False)

# ============================================================================ in-text numbers
Ti = T.set_index("design")
prim = f"Blocks {PRIM_L} m + {PRIM_B} m buffer (primary)"
lines = ["SI S12 'Distance between test and training buildings' (site-weighted, both criteria)",
         f"random 80/20 split: median {Ti.loc['Random 80/20 split', 'median_m']:.1f} m (manuscript 5.8 m)",
         f"250 m blocks, no buffer: median {Ti.loc[f'Blocks {PRIM_L} m', 'median_m']:.1f} m (manuscript 51.6 m)",
         f"primary design: median {Ti.loc[prim, 'median_m']:.1f} m, 5th-95th percentile {Ti.loc[prim, 'p05_m']:.0f}-"
         f"{Ti.loc[prim, 'p95_m']:.0f} m, minimum {Ti.loc[prim, 'min_m']:.0f} m (manuscript 279.5 m, 209-453 m, 200 m)"]
if "Held-out site (pooled leave-site-out)" in Ti.index:
    lines.append(f"held-out site: median {Ti.loc['Held-out site (pooled leave-site-out)', 'median_m'] / 1000:.1f} km "
                 "(manuscript 5.1 km)")
(OUT / "in_text_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
for ln in lines:
    log(ln)
log("done")
