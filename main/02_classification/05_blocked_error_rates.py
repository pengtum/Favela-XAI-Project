"""Step 5: misclassification rate by distance to the boundary when whole neighbourhoods are withheld.

Manuscript items
  * Results 2.1, SI Sections S5 and S12: with whole neighbourhoods withheld
    from training (250 m blocks, 200 m buffer) the misclassification rate
    within 6 m of the official boundary is 51.6% against 15.5% at 90 m or
    more (ratio 3.3, 95% confidence interval 2.0-6.5), higher near the line
    than far from it in all nine scored sites.
  * Methods 4.7: the same conditional rates under the spatially blocked
    designs of Methods 4.5.
  The folds, the buffer and both scoring criteria come from the shared spatial cross-validation
  code (favela_xai.spatial_cv, also used by appendix/S12_spatial_cv): the same fold maker, seeds,
  buffer and scoring rules as SI Section S12.

Design (config.yaml)
  boundary_errors.blocked_design   250 m square blocks, first grid placement (placement 0), with a
                                   200 m training buffer; the same blocks without a buffer are
                                   reported alongside
  spatial_cv                       whole blocks assigned to five folds by a greedy rule that balances
                                   building count and favela share (fold seed 42 + placement); a
                                   buffer removes every training building within the buffer
                                   distance of any test building
  model_cv                         XGBoost 100 trees, seed 432, training-median imputation (Methods 4.5)
  spatial_cv.scoring               criterion 1: a fold is used only if its training set holds at
                                   least 50 buildings of the minority class; criterion 2: a site is
                                   scored only if at least half of its favela buildings received a
                                   prediction (Methods 4.5; SI Section S12). Primary = both
                                   criteria; sensitivity = criterion 1 only.
  boundary_errors                  distance bands, Wilson z, site bootstrap (2,000 resamples, seed 42,
                                   a fresh draw per design and criterion over the scored sites)

Inputs
  outputs/main/01_features/<site>_buildings_features.gpkg   products.features_gpkg
  outputs/main/01_features/signed_distance.parquet          products.signed_distance

Outputs (outputs/main/02_classification/)
  blocked_folds.csv      one row per site x buffer x fold: training / test sizes, class counts,
                         fit status, test metrics, criterion 1
  blocked_coverage.csv   per site and buffer: share of favela buildings with a prediction after
                         criterion 1, scored (criterion 2)
  blocked_bands.csv      design x criterion x side x band: buildings, errors, rate, Wilson and
                         site-bootstrap intervals
  blocked_ratios.csv     design x criterion x side: near and far rates, ratio with site-bootstrap
                         interval, sites with near > far
  blocked_sites.csv      design x criterion x side x site: near and far counts and rates

Run from the repository root (independent of steps 1-4; about 100 model fits):
  python main/02_classification/05_blocked_error_rates.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from favela_xai import spatial_cv as scv
from favela_xai.boundary import band_index, load_signed_distance
from favela_xai.config import load_config
from favela_xai.features import MODEL_FEATURE_ORDER
from favela_xai.io import read_site_features, write_table
from favela_xai.labels import label_to_int
from favela_xai.stats import pooled_rate_bootstrap, site_bootstrap_draws, wilson_interval

CRITERIA = {"both": "both criteria (primary)", "first_only": "first criterion only (sensitivity)"}


def band_labels(edges) -> list[str]:
    e = [float(v) for v in edges]
    return [f"< {e[1]:g} m"] + [f"{a:g}-{b:g} m" for a, b in zip(e[1:-1], e[2:])] + [f">= {e[-1]:g} m"]


def run_site(cfg, s: scv.CVSettings, site: str, signed: pd.DataFrame, block_m: float, buffers, placement: int):
    """Fold rows and test predictions of the blocked design for one site."""
    g = read_site_features(cfg, site, columns=["uID", "label"] + MODEL_FEATURE_ORDER)
    X = g[MODEL_FEATURE_ORDER].to_numpy(dtype=float)
    y = label_to_int(g["label"])
    c = g.geometry.centroid
    xy = np.column_stack([c.x.to_numpy(), c.y.to_numpy()])
    sd = signed[signed["site"] == site].set_index("uID").reindex(g["uID"].to_numpy())["signed_dist"].to_numpy(float)
    if np.isnan(sd).any():
        raise SystemExit(f"{site}: buildings without a signed distance; re-run main/01_features/05_signed_distance.py")

    splits, meta = scv.design_square(site, y, xy, block_m, s, n_folds=s.n_folds, placement=placement)
    rows, preds = scv.run_splits(X, y, xy, splits, buffers, meta, cfg, s, placement=placement, controls=False,
                                 pred_buffers=tuple(buffers))
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if len(pred):
        pred["site"] = site
        pred["uID"] = g["uID"].to_numpy()[pred["row_id"].to_numpy()]
        pred["signed_dist"] = sd[pred["row_id"].to_numpy()]
    return pd.DataFrame(rows), pred, int(y.sum())


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    s = scv.CVSettings.from_config(cfg)
    be = cfg.section("boundary_errors")
    bd = be["blocked_design"]
    block_m, placement = float(bd["block_m"]), int(bd["grid_placement"])
    buffers = [0.0, float(bd["buffer_m"])]
    design_name = {b: f"{int(block_m)} m blocks, " + ("no buffer" if b == 0 else f"{int(b)} m buffer") for b in buffers}
    z = float(be["wilson_z"])
    blabels = band_labels(be["band_edges_m"])
    near, far = [int(v) for v in be["near_far_bands"]]
    print(f"design: {int(block_m)} m square blocks, grid placement {placement}, {s.n_folds} folds, "
          f"buffers {buffers} m; XGBoost {cfg.get('model_cv.n_estimators')} trees, seed {cfg.get('model_cv.random_state')}")

    signed = load_signed_distance(cfg, cfg.sites)
    folds, preds, total_fav = [], [], {}
    for site in cfg.sites:
        f, p, n_fav = run_site(cfg, s, site, signed, block_m, buffers, placement)
        folds.append(f)
        preds.append(p)
        total_fav[site] = n_fav
        print(f"  {cfg.site_name(site):24s} folds fitted: " + ", ".join(
            f"buffer {int(b)} m {int((f[f.buffer == b].status == 'ok').sum())}/{s.n_folds}" for b in buffers))
    F = pd.concat(folds, ignore_index=True)
    P = pd.concat(preds, ignore_index=True)
    F["criterion1"] = scv.rule1_mask(F, s)
    write_table(F, out_dir / "blocked_folds.csv")
    tot_fav = pd.Series(total_fav)

    band_rows, ratio_rows, site_rows, cov_rows = [], [], [], []
    for b in buffers:
        fb = F[F.buffer == b]
        kept, cov = scv.apply_rules(fb, tot_fav, s)                  # both criteria (per site x placement)
        cov_rows.append(cov.assign(buffer=b))
        r1 = fb[fb.criterion1]
        used = {"both": {(x, int(f)) for x, f in zip(kept.site, kept.fold)},
                "first_only": {(x, int(f)) for x, f in zip(r1.site, r1.fold)}}
        pb = P[P.buffer == b]
        for crit, clab in CRITERIA.items():
            keep = np.array([(si, int(fo)) in used[crit] for si, fo in zip(pb.site, pb.fold)], bool)
            q = pb[keep]
            sites = [x for x in cfg.sites if x in set(q.site)]
            S = {x: i for i, x in enumerate(sites)}
            si = q.site.map(S).to_numpy()
            sd = q.signed_dist.to_numpy(float)
            er = q.misclassified.to_numpy().astype(float)
            bi = band_index(np.abs(sd), be["band_edges_m"])
            inside = sd < 0
            draws = site_bootstrap_draws(len(sites), int(be["bootstrap_resamples"]), int(be["bootstrap_seed"]))
            for side, m in {"inside": inside, "outside": ~inside, "both": np.ones(len(sd), bool)}.items():
                N = np.zeros((len(sites), len(blabels)))
                K = np.zeros((len(sites), len(blabels)))
                np.add.at(N, (si[m], bi[m]), 1)
                np.add.at(K, (si[m], bi[m]), er[m])
                n, k = N.sum(0), K.sum(0)
                wlo, whi = wilson_interval(k, n, z)
                boot = pooled_rate_bootstrap(N, K, draws)
                with np.errstate(invalid="ignore", divide="ignore"):
                    blo, bhi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                    rb = boot[:, near] / boot[:, far]
                rb[~np.isfinite(rb)] = np.nan
                for j, lab in enumerate(blabels):
                    band_rows.append(dict(design=design_name[b], buffer_m=b, criterion=clab, n_sites=len(sites),
                                          side=side, band=lab, n_buildings=int(n[j]), n_errors=int(k[j]),
                                          error_rate=k[j] / n[j] if n[j] else np.nan, wilson_lo=wlo[j],
                                          wilson_hi=whi[j], boot_lo=blo[j], boot_hi=bhi[j]))
                # per site, near against far
                with np.errstate(invalid="ignore", divide="ignore"):
                    rn, rf = K[:, near] / N[:, near], K[:, far] / N[:, far]
                wl, _ = wilson_interval(K[:, near], N[:, near], z)
                _, wh = wilson_interval(K[:, far], N[:, far], z)
                ev = (N[:, near] > 0) & (N[:, far] > 0)
                for x in sites:
                    i = S[x]
                    site_rows.append(dict(design=design_name[b], buffer_m=b, criterion=clab, side=side, site=x,
                                          n_near=int(N[i, near]), errors_near=int(K[i, near]), n_far=int(N[i, far]),
                                          errors_far=int(K[i, far]), rate_near=rn[i], rate_far=rf[i],
                                          evaluable=bool(ev[i]),
                                          near_exceeds_far=bool(rn[i] > rf[i]) if ev[i] else np.nan,
                                          wilson_separated=bool(wl[i] > wh[i]) if ev[i] else np.nan))
                kk, mm = int((rn[ev] > rf[ev]).sum()), int(ev.sum())
                rlo, rhi = np.nanpercentile(rb, [2.5, 97.5])
                ratio_rows.append(dict(design=design_name[b], buffer_m=b, criterion=clab, n_sites=len(sites),
                                       side=side, rate_near=k[near] / n[near], rate_far=k[far] / n[far],
                                       ratio_near_far=(k[near] / n[near]) / (k[far] / n[far]),
                                       ratio_boot_lo=rlo, ratio_boot_hi=rhi,
                                       ratio_boot_valid_resamples=int(np.isfinite(rb).sum()),
                                       sites_near_exceeds_far=kk, sites_evaluable=mm,
                                       sign_test_p=binomtest(kk, mm, 0.5).pvalue if mm else np.nan,
                                       sites_not_scored=", ".join(x for x in cfg.sites if x not in S)))

    write_table(pd.concat(cov_rows, ignore_index=True), out_dir / "blocked_coverage.csv")
    write_table(pd.DataFrame(band_rows), out_dir / "blocked_bands.csv")
    R = pd.DataFrame(ratio_rows)
    write_table(R, out_dir / "blocked_ratios.csv")
    write_table(pd.DataFrame(site_rows), out_dir / "blocked_sites.csv")

    for r in R[R.side == "both"].itertuples():
        print(f"{r.design:28s} {r.criterion:36s} {100 * r.rate_near:5.1f}% within 6 m vs {100 * r.rate_far:5.1f}% "
              f"at 90 m or more; ratio {r.ratio_near_far:.2f} ({r.ratio_boot_lo:.2f}-{r.ratio_boot_hi:.2f}); "
              f"near > far in {r.sites_near_exceeds_far} of {r.sites_evaluable} sites ({r.n_sites} scored)")
    print("Results 2.1 / SI S5 / SI S12 (250 m blocks, 200 m buffer, both criteria): 51.6% vs 15.5%, "
          "ratio 3.3 (2.0-6.5), nine scored sites")


if __name__ == "__main__":
    main()
