"""Step 5 of SI Section S13: the real 2013 boundary against twenty shifted boundaries.

Manuscript items (SI Section S13 'Shifted boundaries'; data of Extended Data
Fig. 5, drawn by 07_figure_ED5_shifted_boundaries.py)
* plateau contrast |dS| and maximum slope within +/-20 m on the real line
  (polygon side, as for the shifted lines) against the largest value over the
  twenty shifted boundaries (placebo_summary.csv, level 'pooled',
  real_geom_value and placebo_max; ED Fig. 5a-c);
* the same for the label step (label - 0.5, no attributions; ED Fig. 5c);
* transition width w with each building on its labelled side
  (smoothness_vs_label.csv, boundary 'real_label'; ED Fig. 5d);
* within-class gradient G on the real line against the twenty shifted lines,
  in the attributions and in the oriented raw values (placebo_summary.csv,
  metrics G_fav / G_non; ED Fig. 5e).
The values quoted in SI S13 are printed at the end of the run.

Design (SI S13; ED Fig. 5)
* Buildings, labels, the 25 indicators and the attributions of the main model
  are held fixed; only the line moves.
* Real line: the 2013 polygons near the site (01_prepare_inputs.py), with each
  building's side set by polygon membership of its centroid ('real_geom',
  "polygon side") or by its label ('real_label', "label side", the convention
  of main-text Figure 4).
* Shifted boundaries: all polygons translated rigidly by L ~ U(150, 400) m in a
  direction theta ~ U(0, 2 pi), twenty times (``boundary_robustness``:
  shift_range_m, n_shifted_boundaries); shift k uses numpy
  default_rng(shift_seed_base + k), L drawn before theta, and the same vector
  is applied at every site. Side set by polygon membership.
* Profiles and metrics: robustness_common (2.5 m bins, >= 20 buildings per bin
  and site, mean of the per-site profiles; |dS|, x0, w as in Figure 4; maximum
  slope of the smoothed profile within +/-20 m).
* Quantities: SHAP of the five leading indicators, the label step, and the raw
  indicators standardised within each site as (value - median)/IQR and oriented
  so that positive values are favela-like.
* Within-class gradient: robustness_common.within_class_gradient (bands within
  10 m and at 25-40 m of the line, >= 20 buildings per band and site; pooled =
  mean over sites). It is computed with |distance|, so it is the same for the
  polygon and the label side of the real line.
* Percentile of the real value among the shifted boundaries =
  100 x (number below + 0.5 x ties) / number valid; one-sided empirical
  p = (1 + number at least as extreme) / (1 + number valid). "Extreme" is
  higher for |dS|, max slope, dS and G, lower for |x0| and w; a line without a
  sign change counts as |x0| = infinity.

Inputs: cache/<site>_buildings.parquet, cache/<site>_polygons.parquet
(01_prepare_inputs.py); outputs/main/04_boundary_profiles/transition_metrics.csv
(``products.transition_metrics``, SI Table S5; optional, for the check in sanity.csv).

Outputs (outputs/appendix/S13_boundary_robustness/)
* placebo_translation.csv: every line (real_label, real_geom, placebo 0-19) x
  quantity, pooled and per site: x0, w, dS, |dS|, |x0|, max slope, bins,
  G_fav, G_non, G_mean.
* placebo_summary.csv: pooled and per-site distribution of each metric over the
  shifted boundaries and the percentile / p of the real line.
* placebo_vectors.csv: the twenty shift vectors.
* smoothness_vs_label.csv: metrics of every quantity on the real line (both sides).
* sanity.csv: label-side and polygon-side x0, w, dS, max slope of the SHAP
  profiles of the five leading indicators, with the SI Table S5 values.
* profiles_translation.npz: pooled profiles on the real line (both sides) and
  on each shifted line, per-site profiles on the real line (read by the figure).

Run from the repository root:
    python appendix/S13_boundary_robustness/05_shifted_boundaries.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import robustness_common as rc
from favela_xai.io import write_table

# Metrics compared between the real line and the shifted lines, and the
# direction in which a value is more distinctive than the shifted lines.
SUMMARY_METRICS = [("abs_dS", "high"), ("maxslope", "high"), ("abs_x0", "low"), ("w", "low"), ("dS", "high"),
                   ("G_fav", "high"), ("G_non", "high"), ("G_mean", "high")]


def line_rows(tag: str, li, per_site: dict, pooled: dict, grads: dict, sd: dict, extra: dict) -> list[dict]:
    """Metric rows of one line: pooled and per site, for every quantity."""
    rows = []
    for q in rc.QUANTITIES:
        kind, ind = rc.quantity_parts(q)
        g = rc.pooled_profile(grads[q])            # mean over sites of [G_fav, G_non]
        m = rc.profile_metrics(pooled[q])
        rows.append(dict(boundary=tag, level="pooled", site="pooled", quantity=kind, indicator=ind, **extra, **m,
                         G_fav=g[0], G_non=g[1], G_mean=0.5 * (g[0] + g[1])))
        for i, s in enumerate(li.sites):
            m = rc.profile_metrics(per_site[q][i])
            n_within = int(np.sum((sd[s] >= -rc.HALF) & (sd[s] <= rc.HALF)))
            rows.append(dict(boundary=tag, level="site", site=s, quantity=kind, indicator=ind,
                             n_within_40m=n_within, **extra, **m,
                             G_fav=grads[q][i, 0], G_non=grads[q][i, 1],
                             G_mean=0.5 * (grads[q][i, 0] + grads[q][i, 1])))
    return rows


def summarise(pt: pd.DataFrame) -> pd.DataFrame:
    """Distribution of each metric over the shifted lines and the position of the real line."""
    out = []
    for (level, site, kind, ind), g in pt.groupby(["level", "site", "quantity", "indicator"], sort=False):
        shifted = g[g["boundary"] == "placebo"]
        for metric, better in SUMMARY_METRICS:
            v = shifted[metric].to_numpy(float)
            if metric == "abs_x0":
                v = np.where(np.isnan(v), np.inf, v)          # no sign change = infinitely far
            v = v[~np.isnan(v)]
            rec = dict(placebo_set="common", level=level, site=site, quantity=kind, indicator=ind, metric=metric,
                       n_placebo_valid=len(v),
                       n_placebo_no_crossing=int(shifted["x0"].isna().sum()) if metric == "abs_x0" else np.nan,
                       placebo_median=float(np.median(v)) if len(v) else np.nan,
                       placebo_min=float(np.min(v)) if len(v) else np.nan,
                       placebo_max=float(np.max(v)) if len(v) else np.nan)
            for tag in ("real_geom", "real_label"):
                r = float(g.loc[g["boundary"] == tag, metric].to_numpy()[0])
                if len(v) and np.isfinite(r):
                    pct = 100.0 * (np.sum(v < r) + 0.5 * np.sum(v == r)) / len(v)
                    extreme = np.sum(v >= r) if better == "high" else np.sum(v <= r)
                    p = (1 + extreme) / (1 + len(v))
                else:
                    pct, p = np.nan, np.nan
                rec[f"{tag}_value"], rec[f"{tag}_percentile"], rec[f"{tag}_p_emp"] = r, pct, p
            rec["more_distinctive_if"] = better
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    cfg = rc.cfg
    li = rc.line_inputs(cfg.sites)
    print("raw orientation (+1: larger raw value = favela-like):", li.orientation, flush=True)

    # ------------------------------------------------------------ real line
    sd_geom, sd_label = rc.real_line(li)
    ps_label, pool_label = rc.line_profiles(sd_label, li, rc.QUANTITIES)
    ps_geom, pool_geom = rc.line_profiles(sd_geom, li, rc.QUANTITIES)
    grad_real = rc.line_gradients(sd_geom, li, rc.QUANTITIES)       # identical for both sides (|d|)
    none = dict(placebo_id=-1, shift_m=0.0, shift_dir_deg=np.nan)
    rows = line_rows("real_label", li, ps_label, pool_label, grad_real, sd_label, none)
    rows += line_rows("real_geom", li, ps_geom, pool_geom, grad_real, sd_geom, none)

    # ------------------------------------------------------------ twenty shifted boundaries
    lo, hi = (float(a) for a in rc.BR["shift_range_m"])
    seed0 = int(rc.BR["shift_seed_base"])
    shifted_pooled = {q: [] for q in rc.QUANTITIES}
    vectors = []
    for k in range(int(rc.BR["n_shifted_boundaries"])):
        rng = np.random.default_rng(seed0 + k)
        length = float(rng.uniform(lo, hi))
        theta = float(rng.uniform(0, 2 * np.pi))
        dx, dy = length * np.cos(theta), length * np.sin(theta)
        vectors.append(dict(placebo_id=k, seed=seed0 + k, shift_m=length, dir_deg=np.degrees(theta), dx=dx, dy=dy))
        sd = {s: rc.polygon_side_distance(rc.translate(li.polygons[s], dx, dy), li.x[s], li.y[s]) for s in li.sites}
        per_site, pooled = rc.line_profiles(sd, li, rc.QUANTITIES)
        grads = rc.line_gradients(sd, li, rc.QUANTITIES)
        for q in rc.QUANTITIES:
            shifted_pooled[q].append(pooled[q])
        rows += line_rows("placebo", li, per_site, pooled, grads, sd,
                          dict(placebo_id=k, shift_m=length, shift_dir_deg=np.degrees(theta)))
        print(f"shifted boundary {k:2d}: L = {length:6.1f} m, direction {np.degrees(theta):6.1f} deg", flush=True)

    pt = pd.DataFrame(rows)
    pt["abs_dS"] = pt["dS"].abs()
    pt["abs_x0"] = pt["x0"].abs()
    write_table(pt, rc.OUT / "placebo_translation.csv")
    write_table(pd.DataFrame(vectors), rc.OUT / "placebo_vectors.csv")
    summary = summarise(pt)
    write_table(summary, rc.OUT / "placebo_summary.csv")

    # ------------------------------------------------------------ real line: both sides
    smooth = []
    for tag, pooled in (("real_label", pool_label), ("real_geom", pool_geom)):
        for q in rc.QUANTITIES:
            kind, ind = rc.quantity_parts(q)
            m = rc.profile_metrics(pooled[q])
            smooth.append(dict(boundary=tag, quantity=kind, indicator=ind, x0=m["x0"], w=m["w"], dS=m["dS"],
                               maxslope=m["maxslope"]))
    write_table(pd.DataFrame(smooth), rc.OUT / "smoothness_vs_label.csv")

    # ------------------------------------------------------------ sanity: label side = main Fig. 4 / SI Table S5
    t5_path = cfg.product("transition_metrics")
    t5 = pd.read_csv(t5_path).set_index("feature") if t5_path.exists() else None
    san = []
    for f in rc.KEY5:
        ml, mg = rc.profile_metrics(pool_label["shap_" + f]), rc.profile_metrics(pool_geom["shap_" + f])
        rec = dict(indicator=f, x0_label_sign=ml["x0"], w_label_sign=ml["w"], dS_label_sign=ml["dS"],
                   maxslope_label_sign=ml["maxslope"], x0_geom_sign=mg["x0"], w_geom_sign=mg["w"],
                   dS_geom_sign=mg["dS"], maxslope_geom_sign=mg["maxslope"],
                   x0_change_geom_minus_label=mg["x0"] - ml["x0"])
        if t5 is not None:
            rec.update(x0_table_S5=t5.loc[f, "x0"], w_table_S5=t5.loc[f, "w"], dS_table_S5=t5.loc[f, "delta_s"])
        san.append(rec)
    san = pd.DataFrame(san)
    write_table(san, rc.OUT / "sanity.csv")
    if t5 is None:
        print(f"note: {t5_path} not found (main/04_boundary_profiles); sanity.csv has no SI Table S5 columns")
    elif not np.allclose(san["x0_label_sign"], san["x0_table_S5"], atol=1e-3, equal_nan=True):
        print("WARNING: label-side x0 differs from SI Table S5 (check the polygon repair in 01_prepare_inputs.py)")

    # ------------------------------------------------------------ profiles for the figure
    npz = {"cen": rc.CEN, "sites": np.array(li.sites)}
    for q in rc.QUANTITIES:
        npz[f"real_label_{q}"] = pool_label[q]
        npz[f"real_geom_{q}"] = pool_geom[q]
        npz[f"placebo_{q}"] = np.vstack(shifted_pooled[q])
        npz[f"real_label_site_{q}"] = ps_label[q]
        npz[f"real_geom_site_{q}"] = ps_geom[q]
    np.savez(rc.OUT / "profiles_translation.npz", **npz)

    # ------------------------------------------------------------ values quoted in SI S13 / ED Fig. 5
    pd.set_option("display.width", 220)
    pooled_sum = summary[summary["level"] == "pooled"].set_index(["quantity", "indicator", "metric"])
    print("\nReal line (polygon side) vs the largest shifted-boundary value, pooled:")
    for q in ["shap_" + f for f in rc.KEY5] + ["label"]:
        kind, ind = rc.quantity_parts(q)
        a, b = pooled_sum.loc[(kind, ind, "abs_dS")], pooled_sum.loc[(kind, ind, "maxslope")]
        print(f"  {kind:5s} {ind:28s} |dS| {a.real_geom_value:.3f} vs {a.placebo_max:.3f}   "
              f"max slope {b.real_geom_value:.4f} vs {b.placebo_max:.4f}")
    sm = pd.DataFrame(smooth).set_index(["boundary", "quantity", "indicator"])
    print("Transition width w, label side: label step {:.1f} m, mean interbuilding distance {:.1f} m, "
          "adjacency {:.1f} m".format(sm.loc[("real_label", "label", "label"), "w"],
                                      sm.loc[("real_label", "shap", "mean_interbuilding_distance"), "w"],
                                      sm.loc[("real_label", "shap", "adjacency"), "w"]))
    print("Within-class gradient, real line vs shifted lines (percentile of the real value):")
    for kind in ("shap", "raw"):
        for f in rc.KEY5:
            gf, gn = pooled_sum.loc[(kind, f, "G_fav")], pooled_sum.loc[(kind, f, "G_non")]
            print(f"  {kind:4s} {f:28s} G_fav {gf.real_geom_value:+.3f} ({gf.real_geom_percentile:5.1f})   "
                  f"G_non {gn.real_geom_value:+.3f} ({gn.real_geom_percentile:5.1f})")
    print("Reported in SI S13 / ED Fig. 5: |dS| 2.39 vs 1.29 and 1.71 vs 0.31; max slope 0.137 vs 0.073 and 0.082 vs 0.029; "
          "label step |dS| 1.00 vs 0.17, max slope 0.093 vs 0.013; w 4.1 / 16.7 / 15.1 m.")


if __name__ == "__main__":
    main()
