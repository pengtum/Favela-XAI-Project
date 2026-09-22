"""Cross-site transfer of the per-site classifiers: the 10 x 10 transfer matrix.

Applies the main per-site model of each site, without refitting, to the
held-out test split of every site and scores balanced accuracy, accuracy and
AUC. Produces SI Fig. S11, the Supplementary Data table of all 100 pairs and
the transfer values of SI Section S8, the Discussion and the Methods
('Cross-site classifier transferability'): within- and cross-site mean
balanced accuracy, site-bootstrap 95% confidence interval, range, pairs below
chance and favela share of the target test splits.

Design (SI S8; config.yaml ``transfer`` and ``appendix_S08_transfer``)
* Row i = training site (classifier applied); column j = target site (test
  split scored); the diagonal holds the within-site test results (SI Table S3).
* Classifiers: the main per-site models of main/02_classification (XGBoost,
  200 trees, max depth 6, learning rate 0.1, seed 42).
* Target test split: stratified 80/20 split (seed 42) of the target site,
  re-derived with favela_xai.model.main_split as in main/02_classification.
* ``transfer.target_imputation``: target_training_medians (missing values of
  the target test rows filled with the target site's training medians) |
  source_training_medians (with those of the applied model's site).
* Favela predicted when P(favela) >= 0.5; balanced accuracy = mean of favela
  recall and formal-settlement recall; AUC with favela as the positive class.
* Site bootstrap of the cross-site mean: the ten sites drawn with replacement,
  cells of the same drawn site index dropped; 10,000 resamples (seed 20260917).

Inputs
* products.features_gpkg (main/01_features): label and the 25 indicators of
  every building of each site.
* products.model_json (main/02_classification): the main per-site model of each site.
* products.site_metrics (main/02_classification), optional: SI Table S3, only
  used to check the diagonal against the within-site accuracy.

Outputs (outputs/appendix/S08_transfer/)
* transfer_pairwise_metrics.csv    100 pairs: accuracy, balanced accuracy, AUC,
                                   target test size and favela share
* Supplementary_Data_transfer.csv  the same table with the column headers of the
                                   released Supplementary Data (UTF-8 with BOM)
* transfer_balanced_accuracy_matrix.csv   10 x 10 balanced accuracy, full
                                   precision, site keys (read by 02_transfer_similarity.py)
* transfer_summary.log             within- and cross-site means, bootstrap interval,
                                   range, pairs below chance, favela shares
* Figure_R1_transfer_balanced_accuracy.pdf   SI Fig. S11

Run (from the repository root, after main/02_classification):
    python appendix/S08_transfer/01_transfer_matrix.py
"""
from __future__ import annotations

import sys
import warnings

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from favela_xai.config import load_config  # noqa: E402
from favela_xai.features import MODEL_FEATURE_ORDER  # noqa: E402
from favela_xai.io import read_site_features, write_table, write_text  # noqa: E402
from favela_xai.labels import label_to_int  # noqa: E402
from favela_xai.model import main_split  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# Column headers of the released Supplementary Data file (same order as the
# machine-readable table).
SUPPLEMENTARY_HEADERS = {
    "training_site": "Training site",
    "target_site": "Target site",
    "pair": "Pair type",
    "n_target_test_buildings": "Target test buildings (n)",
    "n_target_test_favela": "Target favela test buildings (n)",
    "target_favela_share": "Target favela share",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "auc": "AUC",
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def is_favela(values) -> np.ndarray:
    """1 for favela, 0 for formal settlement (label strings or 1/0 codes)."""
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values):
        return (values.to_numpy(dtype=float) == 1).astype(int)
    return label_to_int(values)


def load_test_split(cfg, site: str) -> dict:
    """Held-out test split of one site, raw and imputed with the site's training medians.

    The split and the medians are those of favela_xai.model.fit_main_model:
    main_split on the rows of the per-site building table, and a median
    SimpleImputer fitted on the training rows (kept for all-missing columns).
    """
    gdf = read_site_features(cfg, site, columns=["label", *MODEL_FEATURE_ORDER])
    X = pd.DataFrame(gdf[MODEL_FEATURE_ORDER].astype(float).to_numpy(), columns=MODEL_FEATURE_ORDER)
    y = is_favela(gdf["label"])
    tr, te = main_split(y, cfg)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X.iloc[tr])
    X_test = X.iloc[te].reset_index(drop=True)
    return dict(X_test_raw=X_test, imputer=imputer,
                X_test=pd.DataFrame(imputer.transform(X_test), columns=MODEL_FEATURE_ORDER),
                y_test=y[te])


def load_model(cfg, site: str) -> XGBClassifier:
    """Main per-site model written by main/02_classification (class 1 = favela)."""
    model = XGBClassifier()
    model.load_model(cfg.require_product("model_json", site=site))
    return model


def target_matrix(split_target: dict, split_source: dict, rule: str) -> pd.DataFrame:
    """Imputed test features of the target site under the configured imputation rule."""
    if rule == "target_training_medians":          # medians of the target site
        return split_target["X_test"]
    if rule == "source_training_medians":          # alternative: medians of the applied model's site
        return pd.DataFrame(split_source["imputer"].transform(split_target["X_test_raw"]),
                            columns=MODEL_FEATURE_ORDER)
    raise ValueError(f"unknown transfer.target_imputation '{rule}'")


def table_s3_accuracy(cfg, sites: list[str]) -> pd.Series | None:
    """Within-site test accuracy of SI Table S3, if main/02_classification has written it."""
    path = cfg.product("site_metrics")
    if not path.exists():
        return None
    t = pd.read_csv(path)
    if not {"site", "accuracy"} <= set(t.columns):
        return None
    t = t.set_index("site")["accuracy"]
    return t.reindex(sites) if set(sites) <= set(t.index) else None


# -----------------------------------------------------------------------------
# Site bootstrap of the cross-site mean
# -----------------------------------------------------------------------------
def site_bootstrap_offdiag_mean(A: np.ndarray, resamples: int, seed: int, percentiles) -> tuple[np.ndarray, int]:
    """Percentile interval of the mean over off-diagonal cells under site resampling.

    Each resample draws n site indices with replacement and averages the cells
    (a, b) of the drawn sites with a != b (cells of the same drawn index dropped).
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        k = rng.integers(0, n, n)
        mask = k[:, None] != k[None, :]
        if mask.sum():
            means.append(A[np.ix_(k, k)][mask].mean())
    return np.percentile(means, percentiles), len(means)


# -----------------------------------------------------------------------------
# Figure (SI Fig. S11)
# -----------------------------------------------------------------------------
MM = 1 / 25.4
FIG_WIDTH_MM = 183.0            # Nature double-column width
TEXT_SWITCH = 0.6               # cell labels in black at or above this value, white below


def draw_matrix(BAL: np.ndarray, labels: list[str], L: float, B: float, W: float, H: float):
    """Matrix of W x W mm with lower-left corner at (L, B) mm on a 183 x H mm canvas."""
    n = len(labels)
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(0, 1)
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, H * MM))
    ax = fig.add_axes([L / FIG_WIDTH_MM, B / H, W / FIG_WIDTH_MM, W / H])
    cax = fig.add_axes([(L + W + 3.0) / FIG_WIDTH_MM, B / H, 3.2 / FIG_WIDTH_MM, W / H])
    ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), BAL, cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=0.6, rasterized=False)
    for i in range(n):
        for j in range(n):
            v = BAL[i, j]
            ax.text(j + .5, i + .5, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v >= TEXT_SWITCH else "white")
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + .5)
    ax.set_yticks(np.arange(n) + .5)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Target site", labelpad=4)
    ax.set_ylabel("Training site", labelpad=4)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_ticks(np.linspace(0, 1, 6))
    cb.outline.set_linewidth(0.5)
    cb.set_label("Balanced accuracy")
    return fig


def make_figure(BAL: np.ndarray, labels: list[str], path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    # Fit the drawing to the 183 mm width with 1 mm margins, then trim the height
    # to the labels (a few passes of measuring the tight bounding box).
    L, B, W, H, PAD = 40.0, 30.0, 120.0, 160.0, 1.0
    for _ in range(4):
        fig = draw_matrix(BAL, labels, L, B, W, H)
        bb = fig.get_tightbbox(fig.canvas.get_renderer())          # inches
        x0, x1, y0, y1 = bb.x0 / MM, bb.x1 / MM, bb.y0 / MM, bb.y1 / MM
        plt.close(fig)
        top, bottom = y1 - (B + W), B - y0                           # label overhang above / below the matrix
        spare = (x0 - PAD) + (FIG_WIDTH_MM - PAD - x1)
        L -= (x0 - PAD)
        W += spare                                                   # use the full width
        B = PAD + bottom
        H = B + W + top + PAD                                        # trim the height
    fig = draw_matrix(BAL, labels, L, B, W, H)
    fig.savefig(path)
    plt.close(fig)


# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    t = cfg.section("transfer")
    s8 = cfg.section("appendix_S08_transfer")
    if t["classifier"] != "model_main" or t["target"] != "test_split":
        raise ValueError("SI S8 applies the main per-site classifiers to the held-out test splits")
    sites = cfg.sites
    n = len(sites)
    if n < 3:
        sys.exit("Cross-site transfer needs the ten sites; it is not meaningful in demo mode.")
    names = [cfg.site_name(s) for s in sites]
    off = ~np.eye(n, dtype=bool)
    thr = float(s8["prediction_threshold"])

    splits = {s: load_test_split(cfg, s) for s in sites}
    models = {s: load_model(cfg, s) for s in sites}

    # ------------------------------------------------------------------ matrix
    BAL = np.full((n, n), np.nan)
    ACC, AUC, NT, NF = BAL.copy(), BAL.copy(), BAL.copy(), BAL.copy()
    for i, s in enumerate(sites):
        for j, tgt in enumerate(sites):
            X = target_matrix(splits[tgt], splits[s], t["target_imputation"])
            y = splits[tgt]["y_test"]
            p = models[s].predict_proba(X[MODEL_FEATURE_ORDER])[:, 1]      # P(favela)
            yhat = (p >= thr).astype(int)
            BAL[i, j] = balanced_accuracy_score(y, yhat)
            ACC[i, j] = accuracy_score(y, yhat)
            AUC[i, j] = roc_auc_score(y, p)
            NT[i, j], NF[i, j] = len(y), int(y.sum())

    # ------------------------------------------------------------------ summary
    log_lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        log_lines.append(str(text))

    s3 = table_s3_accuracy(cfg, sites)
    if s3 is not None:
        log(f"check: max |diagonal accuracy - SI Table S3 accuracy| = {np.nanmax(np.abs(np.diag(ACC) - s3.to_numpy())):.2e}")

    B_res, seed = int(t["bootstrap_resamples"]), int(t["bootstrap_seed"])
    pct = [float(v) for v in s8["bootstrap_percentiles"]]
    for label, A in [("balanced accuracy", BAL), ("accuracy", ACC), ("AUC", AUC)]:
        ci, nb = site_bootstrap_offdiag_mean(A, B_res, seed, pct)
        lo = np.unravel_index(np.where(off, A, np.inf).argmin(), A.shape)
        hi = np.unravel_index(np.where(off, A, -np.inf).argmax(), A.shape)
        log(f"{label:>17}: within-site mean {np.diag(A).mean():.3f} (range {np.diag(A).min():.3f}-{np.diag(A).max():.3f}); "
            f"cross-site mean {A[off].mean():.3f} [site-bootstrap {pct[1] - pct[0]:.0f}% CI {ci[0]:.3f}, {ci[1]:.3f}; B={nb}], "
            f"median {np.median(A[off]):.3f}, range {A[off].min():.3f} ({names[lo[0]]} -> {names[lo[1]]}) to "
            f"{A[off].max():.3f} ({names[hi[0]]} -> {names[hi[1]]})")
    chance = float(s8["chance_level"])
    log(f"cross-site pairs with balanced accuracy < {chance}: {(BAL[off] < chance).sum()} of {off.sum()}")
    hi = np.unravel_index(np.where(off, BAL, -np.inf).argmax(), BAL.shape)
    log(f"best cross-site pair: {names[hi[0]]} -> {names[hi[1]]} {BAL[hi]:.3f}; "
        f"within {names[hi[0]]} {BAL[hi[0], hi[0]]:.3f}")
    share = NF[0] / NT[0]
    log(f"favela share of the target test splits: {share.min():.2f}-{share.max():.2f} ("
        + ", ".join(f"{names[j]} {share[j]:.3f}" for j in range(n)) + ")")
    donors = pd.Series([BAL[i][off[i]].mean() for i in range(n)], index=names).sort_values(ascending=False)
    targets = pd.Series([BAL[:, j][off[:, j]].mean() for j in range(n)], index=names).sort_values(ascending=False)
    log("mean cross-site balanced accuracy by training site: " + ", ".join(f"{k} {v:.2f}" for k, v in donors.items()))
    log("mean cross-site balanced accuracy by target site:   " + ", ".join(f"{k} {v:.2f}" for k, v in targets.items()))

    # ------------------------------------------------------------------ tables
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append(dict(training_site=names[i], target_site=names[j],
                             pair="within-site" if i == j else "cross-site",
                             n_target_test_buildings=int(NT[i, j]), n_target_test_favela=int(NF[i, j]),
                             target_favela_share=round(NF[i, j] / NT[i, j], 4),
                             accuracy=round(ACC[i, j], 4), balanced_accuracy=round(BAL[i, j], 4),
                             auc=round(AUC[i, j], 4)))
    pairs = pd.DataFrame(rows)
    write_table(pairs, out / "transfer_pairwise_metrics.csv")
    pairs.rename(columns=SUPPLEMENTARY_HEADERS).to_csv(out / "Supplementary_Data_transfer.csv",
                                                      index=False, encoding="utf-8-sig")
    matrix = pd.DataFrame(BAL, index=pd.Index(sites, name="training_site"), columns=sites)
    write_table(matrix, out / "transfer_balanced_accuracy_matrix.csv", index=True)
    write_text("\n".join(log_lines) + "\n", out / "transfer_summary.log")

    make_figure(BAL, names, out / "Figure_R1_transfer_balanced_accuracy.pdf")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
