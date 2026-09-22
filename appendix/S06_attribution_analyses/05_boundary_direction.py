"""SI Section S6.1: direction of the boundary transition (values of SI Fig. S9).

Measures, along each site's favela-formal axis in attribution space, how far
the buildings near the official 2013 favela boundary are displaced from their
own reference class toward the other class; SI Fig. S9 is drawn from the table
written here by 06_figure_boundary_direction.py.

Design (SI S6.1; config.yaml ``boundary_direction`` and
``appendix_S06_attribution_analyses.direction``)
With d the signed distance to the boundary (negative on the favela side;
products.signed_distance):
* reference classes: deep favela d < -50 m, deep formal d > +50 m;
* boundary zone |d| <= 20 m, split by label into a near-boundary favela group
  and a near-boundary formal group; at least 30 buildings per group (checked);
* C_G: mean 25-dimensional Tree-SHAP vector of group G (main per-site model);
  site axis v = C_formal - C_favela; t(G) = (C_G - C_favela) . v / (v . v),
  so t = 0 at the favela pole and 1 at the formal pole;
* inward displacement = t(near favela), outward displacement =
  1 - t(near formal), in per cent of the site's deep-to-deep contrast;
  net skew = outward - inward;
* pooled values: unweighted mean over the sites; +- = standard deviation of
  the per-site net skews (ddof ``direction.net_skew_sd_ddof``);
* sensitivity: reference threshold ``sweep_deep_threshold_m`` x boundary zone
  ``sweep_boundary_zone_m``;
* ``direction.sweep_design``: grid (every combination) | one_at_a_time (each
  list with the other setting at its primary value);
* ``direction.min_group_rule``: flag (keep and flag a site below the minimum
  group size) | exclude (leave it out of that design point).

Inputs
* products.shap_gpkg (main/02_classification): label and 25 Tree-SHAP columns;
* products.signed_distance (main/01_features).
Outputs (outputs/appendix/S06_attribution_analyses/)
* boundary_direction_by_site.csv: per site the group sizes, t of the two
  near-boundary groups, inward and outward displacement and net skew (per
  cent), plus a final row 'all_sites' with the means and the s.d. of the net
  skew (primary design);
* boundary_direction_sensitivity.csv: the same per site and design point of the
  sweep, with the pooled values and the sign of every site's skew relative to
  the primary design;
* boundary_direction_summary.txt: the values reported in SI S6.1.

Run from the repository root:
    python appendix/S06_attribution_analyses/05_boundary_direction.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from favela_xai.config import ConfigError, load_config
from favela_xai.features import SHAP_COLUMNS
from favela_xai.io import write_table, write_text

from attribution_common import add_signed_distance, settings, shap_table, site_order

BY_SITE = "boundary_direction_by_site.csv"
SWEEP = "boundary_direction_sensitivity.csv"
SUMMARY = "boundary_direction_summary.txt"
GROUPS = ["deep_favela", "deep_formal", "near_favela", "near_formal"]


def group_masks(d: np.ndarray, is_favela: np.ndarray, deep_m: float, zone_m: float) -> dict[str, np.ndarray]:
    """The two reference classes and the two near-boundary groups of SI S6.1."""
    near = np.abs(d) <= zone_m
    return {
        "deep_favela": d < -deep_m,
        "deep_formal": d > deep_m,
        "near_favela": near & is_favela,
        "near_formal": near & ~is_favela,
    }


def site_direction(t: pd.DataFrame, deep_m: float, zone_m: float, min_n: int) -> dict:
    """Displacements of one site for one design point (percentages)."""
    X = t[SHAP_COLUMNS].to_numpy(float)
    d = t["signed_dist"].to_numpy(float)
    fav = t["is_favela"].to_numpy(bool)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(d)
    masks = group_masks(d, fav, deep_m, zone_m)
    res = {f"n_{g}": int((m & ok).sum()) for g, m in masks.items()}
    res["meets_min_group"] = all(res[f"n_{g}"] >= min_n for g in GROUPS)
    if any(res[f"n_{g}"] == 0 for g in GROUPS):
        res.update(t_near_favela=np.nan, t_near_formal=np.nan, inward_pct=np.nan, outward_pct=np.nan,
                   net_skew_pct=np.nan)
        return res

    C = {g: X[m & ok].mean(axis=0) for g, m in masks.items()}      # group centroids in SHAP space
    v = C["deep_formal"] - C["deep_favela"]                        # the site's favela-formal axis
    vv = float(v @ v)
    if vv == 0:
        raise ConfigError("the two reference centroids coincide; the site axis is undefined")

    def t_of(g: str) -> float:
        return float((C[g] - C["deep_favela"]) @ v / vv)

    res["t_near_favela"] = t_of("near_favela")
    res["t_near_formal"] = t_of("near_formal")
    res["inward_pct"] = 100 * res["t_near_favela"]                 # inside, toward formal
    res["outward_pct"] = 100 * (1 - res["t_near_formal"])          # outside, toward favela
    res["net_skew_pct"] = res["outward_pct"] - res["inward_pct"]
    return res


def run_design(tables: dict[str, pd.DataFrame], deep_m: float, zone_m: float, min_n: int, rule: str) -> pd.DataFrame:
    """Per-site results of one design point; sites below the minimum group size are
    flagged (rule 'flag') or set to missing (rule 'exclude')."""
    rows = []
    for site, t in tables.items():
        r = site_direction(t, deep_m, zone_m, min_n)
        if rule == "exclude" and not r["meets_min_group"]:
            for k in ("t_near_favela", "t_near_formal", "inward_pct", "outward_pct", "net_skew_pct"):
                r[k] = np.nan
        rows.append(dict(site=site, deep_threshold_m=deep_m, boundary_zone_m=zone_m, **r))
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    b = cfg.section("boundary_direction")
    s = settings(cfg)["direction"]
    if b.get("pooled_statistic") != "mean_over_sites":
        raise ConfigError("only boundary_direction.pooled_statistic: mean_over_sites is implemented")
    rule = s["min_group_rule"]
    if rule not in ("flag", "exclude"):
        raise ConfigError("direction.min_group_rule must be 'flag' or 'exclude'")
    deep0, zone0 = float(b["deep_threshold_m"]), float(b["boundary_zone_m"])
    min_n = int(b["min_buildings_per_group"])
    ddof = int(s["net_skew_sd_ddof"])
    sites = site_order(cfg)

    tables = {}
    for site in sites:
        t = add_signed_distance(cfg, shap_table(cfg, site))
        tables[site] = t[["is_favela", "signed_dist"] + SHAP_COLUMNS]

    # ---- primary design (SI S6.1: 50 m reference threshold, 20 m boundary zone) ----
    prim = run_design(tables, deep0, zone0, min_n, rule)
    prim.insert(1, "site_name", [cfg.site_name(x) for x in prim["site"]])
    for _, r in prim.iterrows():
        if not r["meets_min_group"]:
            sizes = ", ".join(f"{g} {r['n_' + g]}" for g in GROUPS)
            print(f"  WARNING {r['site_name']}: fewer than {min_n} buildings in a group ({sizes})")
    mean_in, mean_out = prim["inward_pct"].mean(), prim["outward_pct"].mean()
    mean_net, sd_net = prim["net_skew_pct"].mean(), prim["net_skew_pct"].std(ddof=ddof)
    pooled = dict(site="all_sites", site_name="All sites (mean)", deep_threshold_m=deep0, boundary_zone_m=zone0,
                  inward_pct=mean_in, outward_pct=mean_out, net_skew_pct=mean_net, net_skew_sd_pct=sd_net,
                  meets_min_group=bool(prim["meets_min_group"].all()))
    by_site = pd.concat([prim, pd.DataFrame([pooled])], ignore_index=True)
    write_table(by_site, out / BY_SITE)

    # ---- sensitivity sweep ---------------------------------------------------------
    deeps = [float(x) for x in b["sweep_deep_threshold_m"]]
    zones = [float(x) for x in b["sweep_boundary_zone_m"]]
    if s["sweep_design"] == "grid":
        points = list(itertools.product(deeps, zones))
    elif s["sweep_design"] == "one_at_a_time":
        points = sorted({(x, zone0) for x in deeps} | {(deep0, z) for z in zones})
    else:
        raise ConfigError("direction.sweep_design must be 'grid' or 'one_at_a_time'")
    sign0 = dict(zip(prim["site"], np.sign(prim["net_skew_pct"])))
    parts = []
    for deep_m, zone_m in points:
        r = run_design(tables, deep_m, zone_m, min_n, rule)
        # 1 = same sign of the net skew as in the primary design, 0 = sign change,
        # missing where either value is missing
        r["same_sign_as_primary"] = [float(np.sign(v) == sign0[x]) if np.isfinite(v) and np.isfinite(sign0[x])
                                     else np.nan for x, v in zip(r["site"], r["net_skew_pct"])]
        r["pooled_inward_pct"] = r["inward_pct"].mean()
        r["pooled_outward_pct"] = r["outward_pct"].mean()
        r["pooled_net_skew_pct"] = r["net_skew_pct"].mean()
        parts.append(r)
    sweep = pd.concat(parts, ignore_index=True)
    write_table(sweep, out / SWEEP)

    # ---- numbers quoted in SI S6.1 -------------------------------------------------
    per_point = sweep.groupby(["deep_threshold_m", "boundary_zone_m"])["pooled_net_skew_pct"].first()
    flips = sweep[sweep["same_sign_as_primary"] == 0]
    below_min = sweep[~sweep["meets_min_group"].astype(bool)]
    order = prim.sort_values("net_skew_pct", ascending=False)
    lines = [
        f"Primary design: deep favela d < -{deep0:g} m, deep formal d > +{deep0:g} m, boundary zone |d| <= {zone0:g} m",
        f"Sites: {len(prim)}; all groups >= {min_n} buildings: {bool(prim['meets_min_group'].all())}",
        f"Pooled (mean over sites): inside {mean_in:.0f}% toward formal, outside {mean_out:.0f}% toward favela",
        f"Net skew across sites: {mean_net:+.0f} +- {sd_net:.0f} percentage points (s.d., ddof {ddof})",
        "Per-site net skew (outward - inward, percentage points):",
        *[f"  {r.site_name:24s} {r.net_skew_pct:+5.0f}   (inward {r.inward_pct:4.0f}, outward {r.outward_pct:4.0f})"
          for r in order.itertuples()],
        f"Outside dominates in {int((prim['net_skew_pct'] > 0).sum())} sites, inside in "
        f"{int((prim['net_skew_pct'] < 0).sum())}",
        f"Sweep ({s['sweep_design']}; reference threshold {deeps[0]:g}-{deeps[-1]:g} m, boundary zone "
        f"{zones[0]:g}-{zones[-1]:g} m, {len(points)} design points):",
        f"  pooled net skew within {per_point.min():+.0f} to {per_point.max():+.0f} points",
        f"  sites changing the sign of their skew: {sorted(set(flips['site']))}",
        f"  site x design points below {min_n} buildings in a group: {len(below_min)}"
        + (f" ({sorted(set(below_min['site']))})" if len(below_min) else ""),
    ]
    text = "\n".join(lines) + "\n"
    write_text(text, out / SUMMARY)
    print(text)
    print(f"[saved] {out / BY_SITE}\n[saved] {out / SWEEP}\n[saved] {out / SUMMARY}")


if __name__ == "__main__":
    main()
