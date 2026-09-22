"""Step 8: SI Fig. S20, distance between test and training buildings.

Manuscript item: SI Figure S20.
  a  site-weighted cumulative distribution of the distance from each test building to
     the nearest training building for the cross-validation designs (lines; scored folds
     of the first grid placement) and three prediction tasks (broad orange lines): T1 the
     test set of a random 80/20 split, T2 a held-out favela component, T3 a held-out
     site; vertical line at 200 m, the indicator neighbourhood.
  b  error rate of the out-of-fold predictions by distance to the nearest training
     building, buildings pooled over sites; grey band = site-bootstrap 95% interval for
     the unbuffered block designs pooled; points only for bins with at least 200 test
     buildings from at least five sites (``spatial_cv.distances``).
Values are those of step 07. Figure size 183 x 85.9 mm.

Inputs  outputs/appendix/S12_spatial_cv/distances/ecdf_distances.parquet, error_by_distance.csv (step 07)
Outputs outputs/appendix/S12_spatial_cv/figures/Figure_R1_spatialcv_distance.pdf (+ _preview.png)
Run     python appendix/S12_spatial_cv/08_figure_S20.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import s12_common as C  # noqa: E402
import s12_figstyle as FS  # noqa: E402

cfg, s, P = C.setup(__file__)
FS.use()
DIST = cfg.section("spatial_cv")["distances"]
MINN, MINS = int(DIST["min_test_buildings_per_bin"]), int(DIST["min_sites_per_bin"])   # SI Fig. S20 caption
NB = float(cfg.get("indicators.neighbourhood_band_m"))   # SI Fig. S20a vertical line: the 200 m indicator neighbourhood
PRIM_L, PRIM_B = int(cfg.get("spatial_cv.primary.block_m")), int(cfg.get("spatial_cv.primary.buffer_m"))
E = pd.read_parquet(C.require(P.out / "distances" / "ecdf_distances.parquet", "07_distances.py"))
DE = pd.read_csv(C.require(P.out / "distances" / "error_by_distance.csv", "07_distances.py"))

FW, FH = 183 * FS.MM, 3.38
fig = plt.figure(figsize=(FW, FH))
Y0 = 1.13                                            # axes bottom (in); legends sit below


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])


bx = add_ax(0.495, Y0, 2.5634, 2.0279)
cx = add_ax(4.095, Y0, 2.2888, 2.0279)

# ---------------- (a) site-weighted cumulative distributions
xg = np.logspace(0, np.log10(3e4), 600)


def ecdf(curve):
    g = E[E.curve == curve]
    parts = [v["d"].to_numpy(float) for _, v in g.groupby("site") if len(v)]
    d = np.concatenate(parts)
    w = np.concatenate([np.full(len(p), 1.0 / len(p)) for p in parts])
    o = np.argsort(d)
    d, cw = d[o], np.cumsum(w[o]) / w.sum()
    i = np.searchsorted(d, xg, side="right")
    y = np.where(i > 0, cw[np.maximum(i - 1, 0)], 0.0)
    pos = np.where(y > 0)[0]
    if len(pos) and pos[0] > 1:
        y[:pos[0] - 1] = np.nan
    one = np.where(y >= 1 - 1e-9)[0]
    if len(one):
        y[one[0] + 1:] = np.nan
    return y


bx.axvline(NB, color=FS.MUTED, lw=0.7, alpha=0.6, zorder=0)
bx.text(NB * 0.925, 0.015, f"{NB:g} m", fontsize=6, color=FS.MUTED, ha="right", va="bottom")
curves = {}
for key in ("T1_random80_20", "T2_component_holdout", "T3_leave_site_out"):
    if not (E.curve == key).any():
        continue
    curves[key] = y = ecdf(key)
    bx.plot(xg, y, color=FS.TASKC, lw=4.2, alpha=0.35, solid_capstyle="butt", solid_joinstyle="round", zorder=1)
designs = [("random5_strat", "Random 5-fold", FS.NEUTRAL, "-"),
           ("grid100_b0", "Blocks 100 m", FS.BLUES[100], "-"), ("grid250_b0", "Blocks 250 m", FS.BLUES[250], "-"),
           ("grid500_b0", "Blocks 500 m", FS.BLUES[500], "-"), ("grid1000_b0", "Blocks 1,000 m", FS.BLUES[1000], "-"),
           (f"grid{PRIM_L}_b{PRIM_B}", f"Blocks {PRIM_L} m + {PRIM_B} m buffer", FS.BLUES[PRIM_L], (0, (1.2, 1.2))),
           ("kmeans5_b0", "k-means, 5 clusters", "#d55181", "-")]
handles = []
for key, lab, col, ls in designs:
    h, = bx.plot(xg, ecdf(key), color=col, lw=1.3, ls=ls, label=lab, zorder=3)
    handles.append(h)
for lab in ("T1  random 80/20 test set", "T2  held-out favela component", "T3  held-out site"):
    handles.append(Line2D([], [], color=FS.TASKC, lw=4.2, alpha=0.35, label=lab))


def at(key, q):
    y = curves[key]
    m = np.isfinite(y)
    return float(np.interp(q, y[m], xg[m]))


for key, q, fac, ha, lab in (("T1_random80_20", 0.35, 1.35, "left", "T1"), ("T2_component_holdout", 0.80, 1.3, "left", "T2"),
                             ("T3_leave_site_out", 0.50, 0.85, "right", "T3")):
    if key in curves:
        bx.text(at(key, q) * fac, q, lab, ha=ha, va="center", fontsize=6.3, color=FS.INK, fontweight="bold")
bx.set_xscale("log")
bx.set_xlim(1, 3e4)
bx.set_ylim(0, 1.02)
bx.set_xticks([1, 10, 100, 1000, 10000])
bx.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
bx.set_xlabel("Distance from test building to nearest training building (m)")
bx.set_ylabel("Cumulative share of test buildings (site-weighted)")
bx.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.16), frameon=False, handlelength=2.2, ncol=2,
          columnspacing=1.0)
bx.set_title("a  Distance to training: CV designs vs prediction tasks", loc="left", fontweight="bold")
FS.clean(bx)

# ---------------- (b) error rate by distance
bins = list(dict.fromkeys(DE.bin))
blab = [b.replace("-", "–") for b in bins]
band = DE[DE.design == "pooled_square_b0"].set_index("bin").loc[bins]
okb = ((band.n >= MINN) & (band.n_sites >= MINS)).to_numpy()
cx.fill_between(np.arange(len(bins))[okb], band.boot_lo[okb], band.boot_hi[okb], color="#8c8b87", alpha=0.22, lw=0,
                zorder=0)
lines = [("random_skf5", "Random 5-fold", FS.NEUTRAL, "-", "o"), ("sq100_b0", "Blocks 100 m", FS.BLUES[100], "-", "o"),
         ("sq250_b0", "Blocks 250 m", FS.BLUES[250], "-", "o"), ("sq500_b0", "Blocks 500 m", FS.BLUES[500], "-", "o"),
         ("sq1000_b0", "Blocks 1,000 m", FS.BLUES[1000], "-", "o"),
         (f"sq{PRIM_L}_b{PRIM_B}", f"Blocks {PRIM_L} m + {PRIM_B} m buffer", FS.BLUES[PRIM_L], (0, (1.2, 1.2)), "s")]
hh = []
for key, lab, col, ls, mk in lines:
    g = DE[DE.design == key].set_index("bin").loc[bins]
    ok = ((g.n >= MINN) & (g.n_sites >= MINS)).to_numpy()
    xs = np.arange(len(bins))[ok]
    h, = cx.plot(xs, g.error_rate[ok], color=col, ls=ls, lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.4, label=lab,
                 zorder=3)
    hh.append(h)
hh.append(Patch(color="#8c8b87", alpha=0.3, lw=0, label="95% site-bootstrap CI (unbuffered blocks pooled)"))
cx.set_xticks(range(len(bins)))
cx.set_xticklabels(blab, rotation=35, ha="right")
cx.set_xlim(-0.4, len(bins) - 0.6)
cx.set_ylim(0, 0.42)
cx.set_xlabel("Distance to nearest training building (m)")
cx.set_ylabel("Out-of-fold error rate (buildings pooled)")
cx.legend(handles=hh, loc="upper left", bbox_to_anchor=(-0.02, -0.25), frameon=False, ncol=2, handlelength=2.0,
          columnspacing=1.0, fontsize=6)
cx.set_title("b  Error rate vs distance to training", loc="left", fontweight="bold")
FS.clean(cx)

FS.save(fig, P.figures / "Figure_R1_spatialcv_distance.pdf")
print("sites per curve:", E.groupby("curve").site.nunique().to_dict())
