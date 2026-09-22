"""Step 3 of SI Section S13: boundary profiles of the raw indicators beside their SHAP profiles.

Manuscript items
* SI Section S13 'Raw indicators and attributions' and SI Fig. S22: raw and
  SHAP plateau contrasts of the five leading indicators with site-bootstrap
  95% intervals, and the number of sites with a positive / negative contrast
  (raw_profiles_metrics.csv: raw_dS_IQR, raw_dS_ci_lo/hi, raw_dS_ci_excludes_0;
  raw_profiles_persite_summary.csv: n_sites_raw_pos/neg, n_sites_shap_pos).
* Data of SI Fig. S22 (04_figure_S22_raw_vs_shap.py).

Method (SI S13; SI Fig. S22)
* Each raw indicator is standardised within each site as (value - median)/IQR
  (``boundary_robustness.raw_standardisation``; robust_z in
  robustness_common.py). Not oriented: the sign is that of the raw value.
* Profiles as in main-text Figure 4: 2.5 m bins, at least 20 buildings per bin
  and site, mean of the per-site profiles; plateau contrast dS = mean of bins
  <= -25 m minus mean of bins >= +25 m; the half-way midpoint is where the
  smoothed profile first crosses the level half-way between the plateaus.
* Site bootstrap: ``boundary_robustness.raw_bootstrap_resamples`` (500)
  resamples of the sites, numpy default_rng(``raw_bootstrap_seed`` = 42), the
  same resamples for every indicator; the sites are indexed in
  ``appendix_S13_boundary_robustness.raw_bootstrap_site_order``.
* Per-site contrasts: the plateau contrast of each site's own profile; a site
  is evaluable when both plateaus hold at least one bin with 20 buildings.

Inputs: cache/<site>_buildings.parquet (01_prepare_inputs.py).
Outputs (outputs/appendix/S13_boundary_robustness/)
* raw_profiles_metrics.csv: per indicator, raw and SHAP dS with bootstrap
  intervals, plateaus, w, half-way midpoints, correlation of the two normalised
  profiles.
* raw_profiles_persite.csv: per site and indicator, raw and SHAP plateau contrasts.
* raw_profiles_persite_summary.csv: per indicator, numbers of sites with a
  positive / negative contrast.
* raw_profiles.npz: pooled profiles and bootstrap bands (read by the figure).

Run from the repository root:
    python appendix/S13_boundary_robustness/03_raw_profiles.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import robustness_common as rc
from favela_xai.config import ConfigError
from favela_xai.io import write_table


def pearson_normalised(a, b) -> float:
    """Pearson correlation of two normalised profiles over the bins finite in both."""
    na, nb = rc.normalise(a), rc.normalise(b)
    ok = np.isfinite(na) & np.isfinite(nb)
    return pearsonr(na[ok], nb[ok])[0] if ok.sum() >= 3 else np.nan


def main() -> None:
    if rc.BR["raw_standardisation"] != "site_median_iqr":
        raise ConfigError("only raw_standardisation: site_median_iqr is implemented (SI Fig. S22 caption)")
    n_boot = int(rc.BR["raw_bootstrap_resamples"])
    rng = np.random.default_rng(int(rc.BR["raw_bootstrap_seed"]))
    sites = rc.bootstrap_site_order(rc.cfg.sites)   # row order of the per-site profile matrices

    # ------------------------------------------------------------ per-site profiles
    raw_ps = {f: [] for f in rc.INDICATORS}          # raw (robust z) profiles, sites x bins
    shap_ps = {f: [] for f in rc.INDICATORS}         # SHAP profiles
    fallback_sites = {f: [] for f in rc.INDICATORS}
    persite = []
    for s in sites:
        df = rc.read_buildings(s, columns=["signed_dist"] + rc.INDICATORS + rc.SHAP)
        d = df["signed_dist"].to_numpy()
        for f in rc.INDICATORS:
            z, fallback = rc.robust_z(df[f].to_numpy())
            if fallback:
                fallback_sites[f].append(s)
            raw_prof = rc.site_profile(d, z)
            shap_prof = rc.site_profile(d, df["shap_" + f].to_numpy())
            raw_ps[f].append(raw_prof)
            shap_ps[f].append(shap_prof)
            ri, ro = rc.plateaus(raw_prof)
            si, so = rc.plateaus(shap_prof)
            persite.append(dict(site=s, feature=f, raw_dS_IQR=ri - ro, shap_dS=si - so))
    raw_ps = {f: np.vstack(v) for f, v in raw_ps.items()}
    shap_ps = {f: np.vstack(v) for f, v in shap_ps.items()}

    # the same site resamples for every indicator
    draws = rng.integers(0, len(sites), size=(n_boot, len(sites)))

    # ------------------------------------------------------------ pooled profiles + bootstrap
    rows, npz = [], {"cen": rc.CEN}
    for f in rc.INDICATORS:
        mr, ms = rc.pooled_profile(raw_ps[f]), rc.pooled_profile(shap_ps[f])
        raw_m, shap_m = rc.profile_metrics(mr), rc.profile_metrics(ms)
        raw_mid, raw_ncross = rc.midpoint(mr)
        shap_mid, shap_ncross = rc.midpoint(ms)
        nr, ns = rc.normalise(mr), rc.normalise(ms)
        ok = np.isfinite(nr) & np.isfinite(ns)
        r, p = pearsonr(nr[ok], ns[ok]) if ok.sum() >= 3 else (np.nan, np.nan)

        boot = np.full((n_boot, 5), np.nan)       # raw dS, raw midpoint, SHAP dS, SHAP x0, r
        bands = {k: np.full((n_boot, len(rc.CEN)), np.nan) for k in ("raw", "shap", "raw_norm", "shap_norm")}
        for b in range(n_boot):
            br, bs = rc.pooled_profile(raw_ps[f][draws[b]]), rc.pooled_profile(shap_ps[f][draws[b]])
            bands["raw"][b], bands["shap"][b] = br, bs
            bands["raw_norm"][b], bands["shap_norm"][b] = rc.normalise(br), rc.normalise(bs)
            mb_raw, mb_shap = rc.profile_metrics(br), rc.profile_metrics(bs)
            boot[b] = [mb_raw["dS"], rc.midpoint(br)[0], mb_shap["dS"], mb_shap["x0"], pearson_normalised(br, bs)]

        def ci(k):
            return np.nanpercentile(boot[:, k], [2.5, 97.5])

        inner, outer = rc.plateaus(mr)
        rows.append(dict(
            feature=f, leading=f in rc.KEY5,
            raw_dS_IQR=raw_m["dS"], raw_dS_ci_lo=ci(0)[0], raw_dS_ci_hi=ci(0)[1],
            raw_dS_ci_excludes_0=bool(ci(0)[0] > 0 or ci(0)[1] < 0),
            raw_inner_plateau=inner, raw_outer_plateau=outer, raw_w=raw_m["w"],
            raw_midpoint=raw_mid, raw_mid_n_crossings=raw_ncross, raw_mid_ci_lo=ci(1)[0], raw_mid_ci_hi=ci(1)[1],
            raw_mid_boot_n_defined=int(np.isfinite(boot[:, 1]).sum()),
            shap_dS=shap_m["dS"], shap_dS_ci_lo=ci(2)[0], shap_dS_ci_hi=ci(2)[1],
            shap_x0=shap_m["x0"], shap_x0_ci_lo=ci(3)[0], shap_x0_ci_hi=ci(3)[1], shap_w=shap_m["w"],
            shap_midpoint=shap_mid, shap_mid_n_crossings=shap_ncross, mid_diff_raw_minus_shap=raw_mid - shap_mid,
            pearson_r_normalised=r, pearson_p=p, pearson_r_ci_lo=ci(4)[0], pearson_r_ci_hi=ci(4)[1],
            n_bins_compared=int(ok.sum()), same_direction=bool(np.sign(raw_m["dS"]) == np.sign(shap_m["dS"])),
            iqr_fallback_sites=";".join(fallback_sites[f])))
        npz[f + "__raw"], npz[f + "__shap"] = mr, ms
        for k, arr in bands.items():
            npz[f"{f}__{k}_band"] = np.nanpercentile(arr, [2.5, 97.5], axis=0)

    res = pd.DataFrame(rows)
    write_table(res, rc.OUT / "raw_profiles_metrics.csv")
    np.savez(rc.OUT / "raw_profiles.npz", **npz)

    # ------------------------------------------------------------ per-site contrasts
    ps = pd.DataFrame(persite)
    write_table(ps, rc.OUT / "raw_profiles_persite.csv")
    summary = ps.groupby("feature").agg(
        raw_dS_mean=("raw_dS_IQR", "mean"),
        n_sites_evaluable_raw=("raw_dS_IQR", lambda v: int(v.notna().sum())),
        n_sites_raw_pos=("raw_dS_IQR", lambda v: int((v > 0).sum())),
        n_sites_raw_neg=("raw_dS_IQR", lambda v: int((v < 0).sum())),
        shap_dS_mean=("shap_dS", "mean"),
        n_sites_evaluable_shap=("shap_dS", lambda v: int(v.notna().sum())),
        n_sites_shap_pos=("shap_dS", lambda v: int((v > 0).sum())),
        n_sites_shap_neg=("shap_dS", lambda v: int((v < 0).sum()))).loc[rc.INDICATORS]
    write_table(summary.reset_index(), rc.OUT / "raw_profiles_persite_summary.csv")

    pd.set_option("display.width", 200)
    key = res.set_index("feature").loc[rc.KEY5]
    print(key[["raw_dS_IQR", "raw_dS_ci_lo", "raw_dS_ci_hi", "raw_dS_ci_excludes_0", "shap_dS", "shap_x0",
               "pearson_r_normalised"]].round(3).to_string())
    print(summary.loc[rc.KEY5].to_string())
    print("\nReported in SI S13 / Fig. S22: squareness +0.91 (+0.50 to +1.51), 9 of 9 sites positive; "
          "avg weighted distance -0.47 (-0.75 to -0.20), 8 of 9 negative; SHAP dS positive in 9 of 9 for all five.")


if __name__ == "__main__":
    main()
