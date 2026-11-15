"""Step 4: misclassification rate by signed distance to the 2013 favela boundary.

Manuscript items
  * Results 2.1: 26.8% of the buildings within 6 m of the official favela
    polygon misclassified against 3.7% of those 90 m or more away (ratio 7.3,
    95% confidence interval 3.9-19.5); the near rate exceeds the far rate in
    all ten sites.
  * Methods 4.7: rates in 6 m signed-distance bins with Wilson 95% intervals
    per bin and site-bootstrap intervals for contrasts between distance bands.
  * SI Section S5: the same numbers and, by side, 25.6% against 3.6%
    (favela-labelled buildings, false negatives) and 40.1% against 3.7%
    (formal-settlement buildings, false positives). SI Table S4 and Extended
    Data Fig. 1 are drawn from the tables written here
    (appendix/S05_boundary_errors).

Definitions (config.yaml, boundary_errors)
  * error: out-of-fold favela probability >= 0.5 differs from the label (03_oof_predictions.py);
  * side: signed distance < 0 = favela-labelled side (inside; errors are false negatives),
    >= 0 = formal side (outside; errors are false positives);
  * 6 m bins from -90 to +90 m plus one open end bin per side (bin_width_m, half_range_m);
  * distance bands on |signed distance|: < 6, 6-30, 30-90, >= 90 m (band_edges_m);
    near/far ratio = rate in the first band / rate in the last band (near_far_bands);
  * Wilson score interval, z = 1.96 (wilson_z);
  * site bootstrap: 2,000 resamples of the sites with replacement (seed 42), buildings
    pooled within each resample; one draw matrix shared by all bins, bands and sides;
    percentile 2.5 / 97.5 (bootstrap_resamples, bootstrap_seed).

Inputs
  outputs/main/02_classification/oof_predictions.parquet   products.oof_predictions

Outputs (outputs/main/02_classification/)
  boundary_error_bins.csv            pooled 6 m bins: buildings, errors, rate, Wilson and
                                     site-bootstrap intervals, share of all errors
  boundary_error_bands.csv           pooled distance bands by side (inside, outside, both) with
                                     the near/far ratio and its site-bootstrap interval
                                     (SI Table S4)
  boundary_error_bins_by_site.csv    the 6 m bins of every site
  boundary_error_bands_by_site.csv   the distance bands of every site
  site_counts_near_far.csv           per site and side: near and far rates, near > far,
                                     Wilson intervals separated, ratio
  boundary_error_summary.txt         the numbers quoted in Results 2.1 and SI S5

Run from the repository root:
  python main/02_classification/04_boundary_error_rates.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from favela_xai.boundary import band_index, bin_index, error_bins
from favela_xai.config import load_config
from favela_xai.io import write_table, write_text
from favela_xai.stats import pooled_rate_bootstrap, site_bootstrap_draws, wilson_interval

SIDE_ERROR = {"inside": "false negative (favela predicted formal)",
              "outside": "false positive (formal predicted favela)",
              "both": "false negative + false positive"}


def band_labels(edges) -> list[str]:
    """'< 6 m', '6-30 m', '30-90 m', '>= 90 m' for edges [0, 6, 30, 90]."""
    e = [float(v) for v in edges]
    labels = [f"< {e[1]:g} m"] + [f"{a:g}-{b:g} m" for a, b in zip(e[1:-1], e[2:])] + [f">= {e[-1]:g} m"]
    return labels


def boot_interval(rates: np.ndarray):
    """Percentile 95% interval over resamples (NaN resamples ignored) and the number of valid resamples."""
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(rates, [2.5, 97.5], axis=0)
    return lo, hi, np.isfinite(rates).sum(axis=0)


def counts(site_idx, cell_idx, err, n_sites, n_cells, mask=None):
    """(sites x cells) arrays of buildings and errors."""
    m = np.ones(len(err), bool) if mask is None else mask
    N = np.zeros((n_sites, n_cells))
    K = np.zeros((n_sites, n_cells))
    np.add.at(N, (site_idx[m], cell_idx[m]), 1)
    np.add.at(K, (site_idx[m], cell_idx[m]), err[m])
    return N, K


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    be = cfg.section("boundary_errors")
    z = float(be["wilson_z"])

    D = pd.read_parquet(cfg.require_product("oof_predictions"))
    sites = [s for s in cfg.sites if s in set(D["site"])]        # config order = bootstrap order
    S = {s: i for i, s in enumerate(sites)}
    si = D["site"].map(S).to_numpy()
    sd = D["signed_dist"].to_numpy(float)
    er = D["error"].to_numpy().astype(float)
    inside = sd < 0
    if not np.array_equal(inside, D["is_favela"].to_numpy().astype(bool)):
        print("note: the side of some buildings differs from their label (signed distance of exactly 0)")
    n_tot, e_tot = len(D), float(er.sum())

    bins = error_bins(float(be["bin_width_m"]), float(be["half_range_m"]), tails=True)
    bidx = bin_index(sd, bins)
    assert (bidx >= 0).all()
    blabels = band_labels(be["band_edges_m"])
    bnd = band_index(np.abs(sd), be["band_edges_m"])
    assert (bnd >= 0).all()
    near, far = [int(v) for v in be["near_far_bands"]]

    draws = site_bootstrap_draws(len(sites), int(be["bootstrap_resamples"]), int(be["bootstrap_seed"]))

    # ---- pooled 6 m bins
    Nb, Kb = counts(si, bidx, er, len(sites), len(bins))
    n, k = Nb.sum(0), Kb.sum(0)
    wlo, whi = wilson_interval(k, n, z)
    blo, bhi, bval = boot_interval(pooled_rate_bootstrap(Nb, Kb, draws))
    rows = []
    for i, (lo, hi, lab) in enumerate(bins):
        side = "inside" if hi <= 0 else "outside"
        sel = bidx == i
        rows.append(dict(bin=lab, lower_m=lo, upper_m=hi,
                         bin_centre_m=(lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                         median_signed_dist_m=float(np.median(sd[sel])) if sel.any() else np.nan,
                         side=side, error_type=SIDE_ERROR[side], n_buildings=int(n[i]), n_errors=int(k[i]),
                         error_rate=k[i] / n[i] if n[i] else np.nan, wilson_lo=wlo[i], wilson_hi=whi[i],
                         boot_lo=blo[i], boot_hi=bhi[i], boot_valid_resamples=int(bval[i]),
                         share_of_all_errors=k[i] / e_tot))
    write_table(pd.DataFrame(rows), out_dir / "boundary_error_bins.csv")

    # ---- pooled distance bands by side, with the near/far ratio
    sides = {"inside": inside, "outside": ~inside, "both": np.ones(n_tot, bool)}
    band_counts, rows, ratios = {}, [], {}
    for side, mask in sides.items():
        N, K = counts(si, bnd, er, len(sites), len(blabels), mask)
        band_counts[side] = (N, K)
        n, k = N.sum(0), K.sum(0)
        wlo, whi = wilson_interval(k, n, z)
        boot = pooled_rate_bootstrap(N, K, draws)
        blo, bhi, bval = boot_interval(boot)
        with np.errstate(invalid="ignore", divide="ignore"):
            rb = boot[:, near] / boot[:, far]
        rb[~np.isfinite(rb)] = np.nan
        ratio = (k[near] / n[near]) / (k[far] / n[far])
        rlo, rhi = np.nanpercentile(rb, [2.5, 97.5])
        ratios[side] = (ratio, rlo, rhi)
        for j, lab in enumerate(blabels):
            rows.append(dict(side=side, band=lab, error_type=SIDE_ERROR[side], n_buildings=int(n[j]),
                             n_errors=int(k[j]), error_rate=k[j] / n[j] if n[j] else np.nan,
                             wilson_lo=wlo[j], wilson_hi=whi[j], boot_lo=blo[j], boot_hi=bhi[j],
                             boot_valid_resamples=int(bval[j]), share_of_all_errors=k[j] / e_tot,
                             ratio_near_far=ratio, ratio_boot_lo=rlo, ratio_boot_hi=rhi,
                             ratio_boot_valid_resamples=int(np.isfinite(rb).sum())))
    bands = pd.DataFrame(rows)
    write_table(bands, out_dir / "boundary_error_bands.csv")

    # ---- per site: 6 m bins and bands
    rows = []
    for s in sites:
        i = S[s]
        n, k = Nb[i], Kb[i]
        wlo, whi = wilson_interval(k, n, z)
        for b, (lo, hi, lab) in enumerate(bins):
            side = "inside" if hi <= 0 else "outside"
            rows.append(dict(site=s, bin=lab, lower_m=lo, upper_m=hi, side=side, n_buildings=int(n[b]),
                             n_errors=int(k[b]), error_rate=k[b] / n[b] if n[b] else np.nan,
                             wilson_lo=wlo[b], wilson_hi=whi[b]))
    write_table(pd.DataFrame(rows), out_dir / "boundary_error_bins_by_site.csv")

    rows, near_far = [], []
    for side, (N, K) in band_counts.items():
        wlo_all, whi_all = wilson_interval(K, N, z)
        for s in sites:
            i = S[s]
            for j, lab in enumerate(blabels):
                rows.append(dict(site=s, side=side, band=lab, n_buildings=int(N[i, j]), n_errors=int(K[i, j]),
                                 error_rate=K[i, j] / N[i, j] if N[i, j] else np.nan,
                                 wilson_lo=wlo_all[i, j], wilson_hi=whi_all[i, j]))
            nn, nf = N[i, near], N[i, far]
            evaluable = bool(nn > 0 and nf > 0)
            rn = K[i, near] / nn if nn else np.nan
            rf = K[i, far] / nf if nf else np.nan
            near_far.append(dict(side=side, site=s, n_near=int(nn), n_far=int(nf), rate_near=rn, rate_far=rf,
                                 evaluable=evaluable, near_exceeds_far=bool(rn > rf) if evaluable else np.nan,
                                 wilson_separated=bool(wlo_all[i, near] > whi_all[i, far]) if evaluable else np.nan,
                                 ratio=rn / rf if evaluable and rf > 0 else np.nan))
    write_table(pd.DataFrame(rows), out_dir / "boundary_error_bands_by_site.csv")
    nf = pd.DataFrame(near_far)
    write_table(nf, out_dir / "site_counts_near_far.csv")

    # ---- summary numbers (Results 2.1, SI Section S5)
    lines = [f"buildings {n_tot:,}, misclassified {int(e_tot):,} ({100 * e_tot / n_tot:.2f}%), sites {len(sites)}"]
    for side in ["both", "inside", "outside"]:
        t = bands[bands.side == side].set_index("band")
        ratio, rlo, rhi = ratios[side]
        ev = nf[(nf.side == side) & nf.evaluable]
        kk, mm = int(ev.near_exceeds_far.sum()), len(ev)
        p = binomtest(kk, mm, 0.5).pvalue if mm else np.nan
        lines.append(f"{side:8s}: {100 * t.loc[blabels[near], 'error_rate']:.1f}% within {blabels[near]} against "
                     f"{100 * t.loc[blabels[far], 'error_rate']:.1f}% at {blabels[far]}; ratio {ratio:.2f} "
                     f"(site bootstrap 95% CI {rlo:.2f}-{rhi:.2f}); near > far in {kk} of {mm} evaluable sites "
                     f"(two-sided sign test p = {p:.4f})")
    lines.append("Results 2.1 / SI S5: both sides 26.8% vs 3.7%, ratio 7.3 (3.9-19.5), 10 of 10 sites; "
                 "inside 25.6% vs 3.6%; outside 40.1% vs 3.7%")
    t = bands[bands.side == "both"].set_index("band")
    close = t.loc[blabels[:2]]
    lines.append(f"share of all errors within |d| < {be['band_edges_m'][2]} m: "
                 f"{100 * close['share_of_all_errors'].sum():.1f}% (buildings: {100 * close['n_buildings'].sum() / n_tot:.1f}%)")
    write_text("\n".join(lines) + "\n", out_dir / "boundary_error_summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
