"""Raw-indicator benchmark quoted in Results 2.4 (read from the SI S9 per-site protocol).

Manuscript items
* Results 2.4 and SI Section S7: a two-component summary of the raw indicators
  separates the classes less well (mean per-site accuracy 0.782 and AUC 0.821,
  against 0.932 and 0.974 for the 25 indicators).
* SI Table S6: rows Feature-Basic (PC1+PC2 of the raw 25) and Morph-25 (the raw
  25 indicators).

The benchmark is computed in appendix/S09_block_complexity_k/03_persite_protocol.py
with the per-site protocol of SI Section S9 (buildings with a valid
block-complexity index k and complete indicators and attributions; stratified
80/20 split, seed 42; XGBoost with 200 trees, max depth 6, learning rate 0.1).
Feature-Basic is a two-component PCA of the 25 raw indicators fitted within
each site without standardisation (config.yaml 'pca.raw_benchmark'); it is
distinct from the pooled, standardised attribution PCA of 01_attribution_pca.py.
This script reads that summary and prints the Results 2.4 numbers. The
25-indicator reference (0.932 / 0.974) is the Morph-25 row of SI Table S6 on the
SI S9 evaluation subset, not the all-building accuracy of SI Table S3.

Inputs
  outputs/appendix/S09_block_complexity_k/k_persite_summary.csv
      (main_05_attribution_plane.raw_benchmark_summary in config.yaml)

Outputs (outputs/main/05_attribution_plane/)
  raw_benchmark_summary.csv   the Feature-Basic and Morph-25 rows: mean and s.d.
                              across the sites of test accuracy and AUC

Run (from the repository root), after appendix/S09_block_complexity_k/03_persite_protocol.py
  python main/05_attribution_plane/03_raw_benchmark_numbers.py
"""
from __future__ import annotations

import pandas as pd

from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_table

# Feature sets of SI Table S6 quoted in Results 2.4
QUOTED = {"feature_basic": "two-component PCA of the raw indicators (Feature-Basic)",
          "morph25": "the 25 raw indicators (Morph-25)"}


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    path = cfg.outputs_root / cfg.get("main_05_attribution_plane.raw_benchmark_summary")
    if not path.exists():
        raise ConfigError(f"{path} does not exist; run appendix/S09_block_complexity_k/03_persite_protocol.py first")
    summary = pd.read_csv(path)
    rows = summary[summary["feature_set"].isin(QUOTED)].copy()
    if len(rows) != len(QUOTED):
        raise ConfigError(f"{path}: expected the rows {sorted(QUOTED)} in column feature_set")
    rows = rows.set_index("feature_set").loc[list(QUOTED)].reset_index()
    write_table(rows, out / "raw_benchmark_summary.csv")

    print("Results 2.4 / SI S7 (mean over the sites of the per-site test values):")
    for r in rows.itertuples():
        print(f"  {QUOTED[r.feature_set]:55s} accuracy {r.acc_mean:.3f} (s.d. {r.acc_sd:.3f}), "
              f"AUC {r.auc_mean:.3f} (s.d. {r.auc_sd:.3f}), {int(r.n_sites)} sites")
    print("  Results 2.4: 0.782 / 0.821 against 0.932 / 0.974")
    print(f"[saved] {out / 'raw_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
