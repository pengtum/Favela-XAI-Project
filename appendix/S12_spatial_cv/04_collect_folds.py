"""Step 4: collect the fold shards of step 3 into the fold record and the prediction table.

Manuscript items: the fold record behind every accuracy value of SI Section S12
(SI Tables S10-S15, SI Figs S19-S21) and Supplementary Data 1-2; the distance
of each held-out site to the nearest training site quoted in the SI Table S10
caption ("held-out sites lie 3.2-12.0 km from the nearest training site").

Checks: every expected shard is present (10 sites x 1 random, 1 leave-site-out,
4 block sizes x 3 placements, 7 sensitivity designs, 1 component job); the
buffered folds keep no training building within the buffer of a test building
(median test-to-training distance above the buffer). With the ten sites the
accuracy designs give 8,020 fold rows.

Inputs  outputs/appendix/S12_spatial_cv/fold_shards/*.parquet, prediction_shards/*.parquet (step 03)
Outputs outputs/appendix/S12_spatial_cv/folds_all.parquet (products.spatial_cv_folds) and folds_all.csv
          one row per fold: site, design, block_rule, L, K, n_units, repeat (grid placement,
          0-based), fold, buffer, control, n_train, n_test, n_dropped, prev_train, prev_test,
          status, accuracy, balanced_accuracy, f1_favela, f1_formal, auc, brier,
          d_test_train_median, d_test_train_p90 (+ offsets, clusters, component group codes,
          leave-site-out distances where they apply)
        outputs/appendix/S12_spatial_cv/predictions_oof.parquet
        outputs/appendix/S12_spatial_cv/summary_loso.csv
Run     python appendix/S12_spatial_cv/04_collect_folds.py
"""
import numpy as np
import pandas as pd

import s12_common as C

cfg, s, P = C.setup(__file__)
SCV = cfg.section("spatial_cv")
log = C.log_to(P.logs / "04_collect_folds.log")

# ----------------------------------------------------------------------------- completeness
n_sens = len(SCV["hexagon_sizes_m"]) + len(SCV["kmeans_unit_sizes_m"]) + 1 + len(SCV["ten_fold"]["block_sizes_m"])
expected = []
for site in cfg.sites:
    expected += [f"random__{site}", f"component__{site}"]
    expected += [f"loso__{site}"] if len(cfg.sites) > 1 else []
    expected += [f"main__{site}__sq{int(L)}__p{r}" for L in SCV["block_sizes_m"] for r in range(int(SCV["grid_placements"]))]
missing = [n for n in expected if not (P.fold_shards / f"{n}.parquet").exists()]
sens_files = sorted(P.fold_shards.glob("sens__*.parquet"))
log(f"expected shards (without sensitivity designs) {len(expected)}, missing {len(missing)} {missing[:5]}; "
    f"sensitivity shards {len(sens_files)} of {len(cfg.sites) * n_sens}")
if missing or len(sens_files) < len(cfg.sites) * n_sens:
    raise SystemExit("step 03 is not complete; run 03_run_folds.py again")

# ----------------------------------------------------------------------------- fold record
F = pd.concat([pd.read_parquet(f) for f in sorted(P.fold_shards.glob("*.parquet"))], ignore_index=True)
F["control"] = F["control"].astype(bool)
order = {site: i for i, site in enumerate(cfg.all_sites)}
F = F.sort_values(["design", "site", "repeat", "buffer", "control", "fold"],
                  key=lambda c: c.map(order) if c.name == "site" else c, kind="stable").reset_index(drop=True)
accuracy_designs = ~F.design.isin(["component", "random_80_20_stratified"])
log(f"fold rows {len(F):,} (accuracy designs {int(accuracy_designs.sum()):,}; expected with ten sites: 8,020); "
    f"held-out component {int((F.design == 'component').sum())}; stratified reference {int((F.design == 'random_80_20_stratified').sum())}")
log("status by design:\n" + F.groupby(["design", "status"]).size().unstack(fill_value=0).to_string())

# buffered folds: the buffer must separate training and test buildings
blk = F[F.status.eq("ok") & ~F.control & (F.buffer > 0) & F.block_rule.isin(["square", "hexagon", "kmeans_many",
                                                                              "kmeans5", "component_group"])]
bad = int((blk.d_test_train_median <= blk.buffer).sum())
log(f"buffered fits with a median test-to-training distance not above the buffer: {bad} of {len(blk)}")

path = cfg.product("spatial_cv_folds", create_parent=True)
F.to_parquet(path, index=False)
F.to_csv(path.with_suffix(".csv"), index=False)
log(f"wrote {path}")

# ----------------------------------------------------------------------------- predictions
Pr = pd.concat([pd.read_parquet(f) for f in sorted(P.pred_shards.glob("*.parquet"))], ignore_index=True)
Pr["buffer"] = Pr["buffer"].astype(np.float32)
Pr.to_parquet(P.out / "predictions_oof.parquet", index=False)
log(f"predictions {len(Pr):,} rows: " + ", ".join(f"{k} {v:,}" for k, v in Pr.groupby("design").size().items()))

# ----------------------------------------------------------------------------- pooled leave-site-out summary
if (F.design == "loso_pooled").any():                      # needs at least two sites
    Lo = F[F.design == "loso_pooled"].set_index("site").reindex(cfg.sites)
    cols = ["status", "n_train", "n_test", "prev_train", "prev_test", "accuracy", "balanced_accuracy", "auc", "f1_favela",
            "d_test_train_min", "d_test_train_median", "nearest_train_site", "nearest_train_site_dist"]
    out = Lo[cols].reset_index()
    out.to_csv(P.out / "summary_loso.csv", index=False)
    km = out.nearest_train_site_dist / 1000
    log(f"pooled leave-site-out: nearest training site {km.min():.1f}-{km.max():.1f} km "
        "(SI Table S10 caption: 3.2-12.0 km); "
        f"balanced accuracy site mean {out.balanced_accuracy.mean():.3f}")
log("done")
