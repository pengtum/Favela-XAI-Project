"""What goes with successful transfer: seven between-site measures, Mantel tests, BH q values.

Mantel correlation of seven between-site measures with the transfer
balanced-accuracy matrix of 01_transfer_matrix.py, with Benjamini-Hochberg
correction across the seven measures (SI Fig. S12, SI Section S8).

Construction (SI S8)
* The 25 indicators are standardised once on the pooled buildings of all ten
  sites: (x_k - m_k) / q_k with m_k the pooled median and q_k the pooled
  interquartile range (1 where it is zero); missing values are ignored.
* Site descriptors: favela fabric f_s and formal fabric n_s (standardised
  medians of the site's favela and formal-settlement buildings); attribution
  composition a_s (mean |Tree SHAP| of the site's main per-site model over its
  favela buildings, scaled to sum to one); favela share pi_s of the site's buildings.
* Seven measures for an ordered pair (a, b): favela-formal contrast similarity
  cos(f_a - n_a, f_b - n_b); same dominant indicator [argmax a_a = argmax a_b];
  attribution-composition similarity cos(a_a, a_b); full attribution ranking
  (Spearman correlation of a_a and a_b over the 25 indicators); favela-fabric
  and formal-fabric distance (mean absolute difference of f, of n);
  difference in favela share |pi_a - pi_b|.
* Mantel test: Spearman correlation between the 90 off-diagonal cells of the
  transfer matrix and of the measure matrix; the ten site identities are
  permuted on rows and columns of the measure matrix; two-sided on |r|;
  p = (c + 1) / (P + 1) with P = 9,999 permutations (seed 42; the same
  permutation sequence for every measure).
* Benjamini-Hochberg correction across the seven measures. The smallest
  attainable p is 1 / (P + 1) = 1e-4, so the smallest q is a floor as well and
  is reported as "q < 0.001".
* Fig. S12b colours the bars with an uncorrected p < 0.05.

Inputs
* outputs/appendix/S08_transfer/transfer_balanced_accuracy_matrix.csv (01_transfer_matrix.py)
* products.features_gpkg (main/01_features): label and the 25 indicators
* products.shap_gpkg (main/02_classification): label and the 25 Tree-SHAP
  columns (positive toward favela) of the main per-site models

Outputs (outputs/appendix/S08_transfer/)
* transfer_predictors.csv   seven measures: Mantel r, p, BH q (rows ordered by p)
* site_descriptors.csv      f_s, n_s, a_s per site and indicator; pi_s per site
* Figure_R1_transfer_similarity.pdf   SI Fig. S12

Run (from the repository root), after 01_transfer_matrix.py:
    python appendix/S08_transfer/02_transfer_similarity.py
"""
from __future__ import annotations

import sys
import warnings

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from favela_xai.attribution import favela_importance  # noqa: E402
from favela_xai.config import ConfigError, load_config  # noqa: E402
from favela_xai.features import MODEL_FEATURE_ORDER, SHAP_COLUMNS  # noqa: E402
from favela_xai.io import read_site_features, read_site_shap, write_table  # noqa: E402
from favela_xai.labels import label_to_int  # noqa: E402
from favela_xai.stats import benjamini_hochberg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATS = MODEL_FEATURE_ORDER


def is_favela(values) -> np.ndarray:
    """1 for favela, 0 for formal settlement (label strings or 1/0 codes)."""
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values):
        return (values.to_numpy(dtype=float) == 1).astype(int)
    return label_to_int(values)


# -----------------------------------------------------------------------------
# Site descriptors
# -----------------------------------------------------------------------------
def site_descriptors(cfg, sites: list[str]) -> dict:
    """f_s, n_s (standardised medians), a_s (favela-only |SHAP| composition) and pi_s."""
    X, y = {}, {}
    for s in sites:
        gdf = read_site_features(cfg, s, columns=["label", *FEATS])
        X[s] = gdf[FEATS].to_numpy(float)
        y[s] = is_favela(gdf["label"])
    pooled = np.vstack([X[s] for s in sites])
    med = np.nanmedian(pooled, axis=0)
    iqr = np.nanpercentile(pooled, 75, axis=0) - np.nanpercentile(pooled, 25, axis=0)
    iqr = np.where(iqr == 0, 1.0, iqr)                       # an IQR of zero is taken as 1

    fav_fabric, formal_fabric, composition, share = {}, {}, {}, {}
    for s in sites:
        fav_fabric[s] = (np.nanmedian(X[s][y[s] == 1], axis=0) - med) / iqr
        formal_fabric[s] = (np.nanmedian(X[s][y[s] == 0], axis=0) - med) / iqr
        share[s] = float(y[s].mean())
        sh = read_site_shap(cfg, s, columns=["label", *SHAP_COLUMNS])
        imp = favela_importance(pd.DataFrame(sh[SHAP_COLUMNS]), is_favela(sh["label"]).astype(bool), FEATS)
        composition[s] = imp["share"].to_numpy()             # mean |SHAP| over favela buildings, sums to 1
    return dict(f=fav_fabric, n=formal_fabric, a=composition, pi=share, median=med, iqr=iqr)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def between_site_measures(d: dict, sites: list[str]) -> dict:
    """The seven 10 x 10 measure matrices of SI S8, keyed by measure name."""
    f, n_, a, pi = d["f"], d["n"], d["a"], d["pi"]

    def matrix(fun):
        return np.array([[fun(p, q) for q in sites] for p in sites], dtype=float)

    return {
        "favela-formal contrast similarity": matrix(lambda p, q: cosine(f[p] - n_[p], f[q] - n_[q])),
        "same dominant indicator": matrix(lambda p, q: float(np.argmax(a[p]) == np.argmax(a[q]))),
        "attribution-composition similarity": matrix(lambda p, q: cosine(a[p], a[q])),
        "full attribution ranking": matrix(lambda p, q: spearmanr(a[p], a[q]).statistic),
        "favela-fabric distance": matrix(lambda p, q: np.mean(np.abs(f[p] - f[q]))),
        "formal-fabric distance": matrix(lambda p, q: np.mean(np.abs(n_[p] - n_[q]))),
        "difference in favela share": matrix(lambda p, q: abs(pi[p] - pi[q])),
    }


# Row labels of Fig. S12b
FIGURE_LABEL = {
    "favela-formal contrast similarity": "favela–formal contrast",
    "same dominant indicator": "same dominant indicator",
    "attribution-composition similarity": "attribution composition",
    "full attribution ranking": "full attribution ranking",
    "favela-fabric distance": "favela-fabric distance",
    "formal-fabric distance": "formal-fabric distance",
    "difference in favela share": "difference in favela share",
}


# -----------------------------------------------------------------------------
# Mantel test (site-identity permutation)
# -----------------------------------------------------------------------------
def mantel(transfer: np.ndarray, measure: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    """Spearman r between off-diagonal cells; p from permuting site identities (two-sided on |r|)."""
    n = transfer.shape[0]
    off = ~np.eye(n, dtype=bool)
    a = rankdata(transfer[off])
    observed = np.corrcoef(a, rankdata(measure[off]))[0, 1]
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        r = np.corrcoef(a, rankdata(measure[np.ix_(perm, perm)][off]))[0, 1]
        if abs(r) >= abs(observed) - 1e-12:
            count += 1
    return float(observed), (count + 1) / (permutations + 1)


# -----------------------------------------------------------------------------
# Figure (SI Fig. S12)
# -----------------------------------------------------------------------------
# Axes boxes (figure fractions) of SI Fig. S12 on its 183 mm x 2.9 in canvas:
# panel a (scatter) and panel b (bars).
AXES_BOXES = [(0.06943746, 0.15354406, 0.3482083, 0.74789272),
              (0.61876653, 0.15354406, 0.36561872, 0.74789272)]
ORANGE, GREY = "#E2700E", "#BDBDBD"


def make_figure(BAL, contrast, same, results: pd.DataFrame, chance: float, alpha: float, path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "font.size": 7.5, "axes.linewidth": 0.7,
                         "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})
    n = BAL.shape[0]
    off = ~np.eye(n, dtype=bool)
    fig, ax = plt.subplots(1, 2, figsize=(183 / 25.4, 2.9))

    # a: transfer against contrast similarity; orange = same dominant indicator
    x, yv, sm = contrast[off], BAL[off], same[off] > 0
    ax[0].scatter(x[~sm], yv[~sm], s=13, color=GREY, lw=0, label="different dominant indicator")
    ax[0].scatter(x[sm], yv[sm], s=13, color=ORANGE, lw=0, label="same dominant indicator")
    ax[0].axhline(chance, color="#888", ls=":", lw=.7)
    ax[0].set_xlabel("similarity of favela–formal contrast (cosine)")
    ax[0].set_ylabel("transfer balanced accuracy")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_title("a", loc="left", fontweight="bold")
    ax[0].set_xticks([-.5, -.25, 0, .25, .5, .75, 1])
    ax[0].set_yticks([.4, .5, .6, .7, .8, .9])

    # b: Mantel r of the seven measures, sorted by r; orange = uncorrected p < alpha
    T = results.sort_values("mantel_r")
    ax[1].barh(range(len(T)), T.mantel_r, color=[ORANGE if p < alpha else GREY for p in T.p_mantel])
    ax[1].set_yticks(range(len(T)))
    ax[1].set_yticklabels([FIGURE_LABEL[k] for k in T.measure], fontsize=6.8)
    ax[1].axvline(0, color="k", lw=.6)
    # fixed x range, widened only if a value falls outside it
    ax[1].set_xlim(min(-.62, T.mantel_r.min() - .05), max(.9, T.mantel_r.max() + .05))
    ax[1].set_xticks([-.5, -.25, 0, .25, .5, .75])
    ax[1].set_xlabel("Mantel r with transfer balanced accuracy")
    ax[1].set_title("b", loc="left", fontweight="bold")

    for a_, box in zip(ax, AXES_BOXES):
        a_.set_position(box)
    fig.savefig(path)
    plt.close(fig)


# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    t = cfg.section("transfer")
    s8 = cfg.section("appendix_S08_transfer")
    sites = cfg.sites
    if len(sites) < 3:
        sys.exit("Cross-site transfer needs the ten sites; it is not meaningful in demo mode.")
    if t["indicator_standardisation"] != "pooled_median_iqr" or t["multiple_testing"] != "benjamini_hochberg":
        raise ValueError("SI S8 states the pooled median/IQR standardisation and the Benjamini-Hochberg correction")

    matrix_path = out / "transfer_balanced_accuracy_matrix.csv"
    if not matrix_path.exists():
        raise ConfigError(f"{matrix_path} not found; run appendix/S08_transfer/01_transfer_matrix.py first")
    BAL = pd.read_csv(matrix_path, index_col=0).loc[sites, sites].to_numpy(float)

    d = site_descriptors(cfg, sites)
    measures = between_site_measures(d, sites)
    P, seed = int(t["mantel_permutations"]), int(t["mantel_seed"])
    rows = []
    for name, M in measures.items():
        r, p = mantel(BAL, M, P, seed)
        rows.append(dict(measure=name, mantel_r=r, p_mantel=p))
    res = pd.DataFrame(rows)
    res["q_bh"] = benjamini_hochberg(res["p_mantel"].to_numpy())
    res["p_at_permutation_floor"] = res["p_mantel"] <= 1.0 / (P + 1) + 1e-12
    res["significant_q05"] = res["q_bh"] < float(s8["significance_level"])
    res = res.sort_values("p_mantel", kind="stable").reset_index(drop=True)
    write_table(res, out / "transfer_predictors.csv")

    # descriptors, for transparency
    desc = pd.concat([pd.DataFrame({"site": s, "indicator": FEATS, "favela_fabric": d["f"][s],
                                    "formal_fabric": d["n"][s], "attribution_share": d["a"][s],
                                    "favela_share": d["pi"][s]}) for s in sites], ignore_index=True)
    write_table(desc, out / "site_descriptors.csv")

    make_figure(BAL, measures["favela-formal contrast similarity"], measures["same dominant indicator"], res,
                float(s8["chance_level"]), float(s8["significance_level"]),
                out / "Figure_R1_transfer_similarity.pdf")
    off = ~np.eye(len(sites), dtype=bool)
    print(f"transfer balanced accuracy: within-site mean {np.diag(BAL).mean():.3f}; "
          f"cross-site mean {BAL[off].mean():.3f}")
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
