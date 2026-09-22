"""Distribution of k by label, pooled and per site (SI S9.2; data of SI Fig. S13).

Manuscript items
* SI S9.2 'Per-building distribution of k': number of buildings with a valid k;
  pooled mean, median and p90 of k for favela and formal-settlement buildings;
  pooled Cohen's d and Mann-Whitney U test; per-site Cohen's d and the number
  of sites in which favela buildings have the higher mean k.
* SI S9: per-site AUC magnitude of k, and the share of buildings outside every
  polygonised block (also Methods 4.9).
* Statistics quoted in the SI Fig. S13 caption; the figure itself is drawn by
  06_figure_S13_k_distribution.py from the files written here.

Statistics (settings in config.yaml ``appendix_S09_block_complexity_k``)
* Buildings: the valid-k subset of every site (k >= 1, complete Tree-SHAP
  attributions, label present; k_common.valid_k).
* Cohen's d = (mean favela - mean formal) / pooled s.d. (ddof 1); Mann-Whitney
  U test, two-sided (scipy); per-site statistics for sites with at least five
  buildings of each class (distribution_min_group).
* AUC of k per site: roc_auc_score with favela as the positive class and k as
  the score; magnitude max(AUC, 1 - AUC); direction "favela deeper" when
  AUC >= 0.5.
* Histogram of SI Fig. S13a: building counts by class for k = 1 ... 12, with
  k above the clip value counted in the last bin (figure_k_clip = 12).

Inputs
* outputs/appendix/S09_block_complexity_k/block_complexity_k.csv and
  block_coverage.csv (02)
* products.features_gpkg (main/01_features), products.shap_gpkg (main/02_classification)

Outputs (outputs/appendix/S09_block_complexity_k/)
* k_distribution.csv     one row per site and one row 'pooled': n by class, mean,
                         s.d., median and p90 of k by class, Cohen's d,
                         Mann-Whitney U and p, AUC of k (raw, magnitude, direction)
* k_histogram.csv        k (1-12, last bin = 12 or more), buildings by class
* k_in_text_numbers.txt  the numbers quoted in Methods 4.9, SI S9 and S9.2

Run (from the repository root), after 02_block_complexity_k.py:
    python appendix/S09_block_complexity_k/05_k_distribution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_table, write_text
from k_common import read_k, site_table, valid_k


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Standardised mean difference with the pooled s.d. (ddof 1)."""
    nx, ny = len(x), len(y)
    sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sd if sd > 0 else np.nan


def describe(k: np.ndarray, y: np.ndarray, site: str) -> dict:
    """Distribution statistics of k for favela (y = 1) and formal-settlement (y = 0) buildings."""
    kf, kn = k[y == 1].astype(float), k[y == 0].astype(float)
    u, p = mannwhitneyu(kf, kn, alternative="two-sided")
    auc = roc_auc_score(y, k) if 0 < y.sum() < len(y) else np.nan
    return dict(site=site, n_favela=len(kf), n_formal=len(kn),
                k_mean_favela=kf.mean(), k_mean_formal=kn.mean(),
                k_sd_favela=kf.std(ddof=1), k_sd_formal=kn.std(ddof=1),
                k_median_favela=np.median(kf), k_median_formal=np.median(kn),
                k_p90_favela=np.percentile(kf, 90), k_p90_formal=np.percentile(kn, 90),
                cohens_d=cohens_d(kf, kn), mannwhitney_u=u, mannwhitney_p=p,
                auc_k=auc, auc_k_magnitude=max(auc, 1 - auc), favela_deeper=bool(auc >= 0.5))


def fmt_p(p: float) -> str:
    return "< 1e-300" if p < 1e-300 else f"= {p:.2e}"


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    s9 = cfg.section("appendix_S09_block_complexity_k")
    min_group = int(s9["distribution_min_group"])
    kmax = int(cfg.get("block_complexity.figure_k_clip"))
    k = read_k(out)
    cov_path = out / "block_coverage.csv"
    if not cov_path.exists():
        raise ConfigError(f"{cov_path} not found; run 02_block_complexity_k.py first")
    coverage = pd.read_csv(cov_path)

    parts = []
    for site in cfg.sites:
        tab = site_table(cfg, site, k)
        parts.append(tab.loc[valid_k(tab), ["site", "uID", "y", "k"]])
    valid = pd.concat(parts, ignore_index=True)

    rows = []
    for site in cfg.sites:
        sub = valid[valid.site == site]
        y = sub["y"].to_numpy()
        if (y == 1).sum() < min_group or (y == 0).sum() < min_group:
            print(f"{site}: fewer than {min_group} buildings in a class, no per-site statistics")
            continue
        rows.append(describe(sub["k"].to_numpy(), y, site))
    rows.append(describe(valid["k"].to_numpy(), valid["y"].to_numpy(), "pooled"))
    dist = pd.DataFrame(rows)
    write_table(dist, out / "k_distribution.csv")

    # Histogram counts of SI Fig. S13a (k clipped at kmax)
    kc = np.clip(valid["k"].to_numpy(dtype=int), 1, kmax)
    yv = valid["y"].to_numpy()
    hist = pd.DataFrame({"k": np.arange(1, kmax + 1)})
    hist["n_favela"] = [int(((kc == v) & (yv == 1)).sum()) for v in hist.k]
    hist["n_formal"] = [int(((kc == v) & (yv == 0)).sum()) for v in hist.k]
    write_table(hist, out / "k_histogram.csv")

    # ------------------------------------------------------------------ in-text numbers
    pooled = dist[dist.site == "pooled"].iloc[0]
    per_site = dist[dist.site != "pooled"].sort_values("cohens_d", ascending=False)
    share = 100 * coverage.set_index("site").loc[cfg.sites, "share_outside_blocks"]
    lines = [
        "SI S9 / Main Methods: buildings outside every polygonised block, per site: "
        f"mean {share.mean():.0f}%, range {share.min():.0f}-{share.max():.0f}% ("
        + ", ".join(f"{cfg.site_name(s)} {v:.1f}%" for s, v in share.items()) + ")",
        f"SI S9.2 / Fig. S13: buildings with a valid k: {len(valid):,} "
        f"({int(pooled.n_favela):,} favela, {int(pooled.n_formal):,} formal settlement)",
        f"pooled mean k: favela {pooled.k_mean_favela:.2f} (median {pooled.k_median_favela:.0f}; "
        f"p90 {pooled.k_p90_favela:.0f}); formal settlement {pooled.k_mean_formal:.2f} "
        f"(median {pooled.k_median_formal:.0f}; p90 {pooled.k_p90_formal:.0f})",
        f"pooled Cohen's d = {pooled.cohens_d:.2f}; Mann-Whitney U p {fmt_p(pooled.mannwhitney_p)}",
        "per-site Cohen's d: " + "; ".join(f"{cfg.site_name(r.site)} {r.cohens_d:+.2f}" for r in per_site.itertuples()),
        f"sites where favela buildings have the higher mean k: {(per_site.cohens_d > 0).sum()} of {len(per_site)}",
        f"SI S9: per-site AUC magnitude of k {per_site.auc_k_magnitude.min():.3f}-{per_site.auc_k_magnitude.max():.3f} ("
        + ", ".join(f"{cfg.site_name(r.site)} {r.auc_k_magnitude:.3f} {'favela deeper' if r.favela_deeper else 'favela shallower'}"
                    for r in per_site.itertuples()) + ")",
    ]
    text = "\n".join(lines) + "\n"
    write_text(text, out / "k_in_text_numbers.txt")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
