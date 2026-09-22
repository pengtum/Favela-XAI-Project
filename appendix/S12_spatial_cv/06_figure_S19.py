"""Step 6: SI Fig. S19, classification accuracy under spatial separation.

Manuscript item: SI Figure S19.
  a, b  balanced accuracy and AUC against the training buffer (0-400 m) for square
        blocks of 100, 250, 500 and 1,000 m (five folds, three grid placements);
        lines = mean over scored sites of the per-site means over used folds, bands =
        site-bootstrap 95% intervals; black lines = random 80/20 split (solid) and
        random five-fold cross-validation (dotted), both refitted in this pipeline;
        grey dashed = matched random-removal control (250 m blocks). 100 m blocks are
        drawn as a line only up to a 100 m buffer (faint open markers beyond); open
        markers = fewer than ten scored sites.
  c     balanced accuracy of the other block shapes, assignment rules and fold
        numbers (no buffer, 200 m buffer, matched control at 200 m), the held-out
        favela component (diamonds) and the pooled leave-site-out.
Values are those of step 05 (scoring procedure of SI S12). Figure size 183 x 137.2 mm.

Inputs  outputs/appendix/S12_spatial_cv/accuracy/scored_grid.csv, scored_designs.csv (step 05)
Outputs outputs/appendix/S12_spatial_cv/figures/Figure_R1_spatialcv_accuracy.pdf (+ _preview.png)
Run     python appendix/S12_spatial_cv/06_figure_S19.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import s12_common as C  # noqa: E402
import s12_figstyle as FS  # noqa: E402

cfg, s, P = C.setup(__file__)
FS.use()
G = pd.read_csv(C.require(P.out / "accuracy" / "scored_grid.csv", "05_accuracy_tables.py"))
D = pd.read_csv(C.require(P.out / "accuracy" / "scored_designs.csv", "05_accuracy_tables.py"))
BUFS = [int(v) for v in cfg.get("spatial_cv.buffers_m")]
BLOCKS = [int(v) for v in cfg.get("spatial_cv.block_sizes_m")]
N_SITES = len(cfg.all_sites)
PRIM_B = int(cfg.get("spatial_cv.primary.buffer_m"))       # SI S12 primary design: 200 m buffer
BUF_COLOUR = {0: FS.BUF[0], PRIM_B: FS.BUF[200]}          # no buffer / primary buffer (s12_figstyle)

FW, FH = 183 * FS.MM, 5.40                   # 183 mm x 137.2 mm
fig = plt.figure(figsize=(FW, FH))
DX = 0.218                                   # horizontal offset that centres the content on the canvas


def add_ax(x0, y0, w, h):                    # axes position in inches
    return fig.add_axes([(x0 + DX) / FW, y0 / FH, w / FW, h / FH])


AB_Y0, AB_H = 3.08, 2.05
axa = add_ax(0.95, AB_Y0, 2.4261, AB_H)
axb = add_ax(0.95 + 3.1539, AB_Y0, 2.4261, AB_H)
C_X0 = 1.60
C_W = 0.95 + 3.1539 + 2.4261 - C_X0          # panel c ends where b ends
axc = add_ax(C_X0, 0.60, C_W, 1.78)
sq = G[G.design == "square"]
r8020 = G[G.design == "random_80_20"].iloc[0]
rskf = G[G.design == "random_skf5"].iloc[0]
TRUNC100 = 100   # 100 m blocks: line and band only up to a 100 m buffer (few sites scored beyond; see caption)


def ladder(ax, metric, ylim, ref_label_y, ylabel, title, yticks):
    ax.axhline(r8020[metric], color=FS.NEUTRAL, lw=1.0, zorder=2)
    ax.axhline(rskf[metric], color=FS.NEUTRAL, lw=1.0, ls=(0, (1, 1.2)), zorder=2)
    ax.text(-8, ref_label_y, "Random split", fontsize=6, color=FS.INK, ha="left", va="bottom")
    ends = []
    # matched random-removal control: 250 m blocks only
    blk250 = sq[(sq.L == 250) & (~sq.control)].set_index("buffer")
    ctl = sq[(sq.L == 250) & (sq.control)].set_index("buffer")
    cx = [0] + list(ctl.index)
    cy = [blk250.loc[0, metric]] + list(ctl[metric])
    ax.plot(cx, cy, color=FS.CONTROL, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ends.append(("ctl", cy[-1]))
    for L in BLOCKS:
        blk = sq[(sq.L == L) & (~sq.control)].set_index("buffer").loc[BUFS]
        xs = np.array(BUFS)
        keep = xs <= (TRUNC100 if L == 100 else max(BUFS))
        ax.fill_between(xs[keep], blk[f"{metric}_ci_lo"][keep], blk[f"{metric}_ci_hi"][keep], color=FS.BLUES[L],
                        alpha=0.10, lw=0, zorder=0)
        ax.plot(xs[keep], blk[metric][keep], color=FS.BLUES[L], lw=1.5, zorder=3)
        y = blk[metric].to_numpy()
        full = blk.n_sites.to_numpy() == N_SITES
        ax.scatter(xs[keep & full], y[keep & full], s=13, color=FS.BLUES[L], edgecolor="white", lw=0.5, zorder=4)
        ax.scatter(xs[keep & ~full], y[keep & ~full], s=13, facecolor="white", edgecolor=FS.BLUES[L], lw=1.0, zorder=4)
        if L == 100:   # faint open markers only, no line or band
            ax.scatter(xs[~keep], y[~keep], s=12, facecolor="none", edgecolor=FS.BLUES[L], lw=0.8, alpha=0.45, zorder=2)
            last = y[np.isfinite(y)][-1]
        else:
            last = y[-1]
        ends.append((L, last))
    ly = FS.repel([e[1] for e in ends], 0.052 * (ylim[1] - ylim[0]))
    for (L, _), yy in zip(ends, ly):
        if L == "ctl":
            ax.text(412, yy, "250 m control", color=FS.MUTED, fontsize=6.0, va="center", ha="left")
        elif L == 100:
            ax.text(412, yy, "100 m", color="#4f8fd6", fontsize=6.3, va="center", ha="left", alpha=0.75)
        else:
            ax.text(412, yy, f"{L:,} m", color=FS.BLUES[L], fontsize=6.3, va="center", ha="left", fontweight="bold")
    # ticks first, then limits (a FixedLocator would otherwise widen the view limits)
    ax.set_xticks(BUFS)
    ax.set_yticks(yticks)
    ax.set_xlim(-12, 490)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Buffer around test blocks (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    FS.clean(ax)


ladder(axa, "balanced_accuracy", (0.38, 0.97), 0.912, "Balanced accuracy (mean of sites; band 95% CI)",
       "a  Balanced accuracy vs buffer, by block size", np.arange(0.3, 1.001, 0.1))
ladder(axb, "auc", (0.60, 1.0), 0.975, "AUC (mean of sites; band 95% CI)", "b  AUC vs buffer, by block size",
       np.arange(0.60, 1.001, 0.05))
axa.text(412, 0.965, "Block size", fontsize=6, color=FS.MUTED, ha="left", va="top")

# ---------------------------------------------------------------- (c) dot plot
rows = [("Square 250 m, K = 5 (primary design)", "main250"), ("Square 500 m, K = 5", "sq500_r0_reference"),
        ("Hexagon 250 m", "hex250"), ("Hexagon 500 m", "hex500"), ("k-means, many clusters (≈250 m)", "kmm250"),
        ("k-means, many clusters (≈500 m)", "kmm500"), ("Square 250 m, K = 10", "sq250_K10"),
        ("Square 500 m, K = 10", "sq500_K10"), ("k-means, 5 clusters (= folds)", "km5"),
        ("Held-out favela component", "component"), ("Pooled leave-site-out (9 sites → 1)", "loso")]


def get(key, b, control=False):
    if key == "loso" and (b or control):
        return None
    if key == "component" and control:
        return None
    x = D[(D.row_key == key) & (D.buffer == b) & (D.control == control)]
    if not len(x):
        return None
    x = x.iloc[0]
    return x.balanced_accuracy, x.balanced_accuracy_ci_lo, x.balanced_accuracy_ci_hi


main0 = get("main250", 0)[0]
main_prim = get("main250", PRIM_B)[0]
axc.axvline(main0, color=BUF_COLOUR[0], lw=0.8, ls=(0, (1, 1.5)), zorder=0)
axc.axvline(main_prim, color=BUF_COLOUR[PRIM_B], lw=0.8, ls=(0, (1, 1.5)), zorder=0)
axc.axvline(r8020.balanced_accuracy, color=FS.NEUTRAL, lw=1.0, zorder=0)
n = len(rows)
for i, (lab, key) in enumerate(rows):
    y = n - 1 - i
    if i % 2 == 0:
        axc.axhspan(y - 0.5, y + 0.5, color="#f1f0ec", lw=0, zorder=-1)
    if key == "loso":
        v = get(key, 0)
        if v is None:
            continue
        m, lo, hi = v
        axc.plot([lo, hi], [y, y], color=FS.NEUTRAL, lw=0.9, zorder=2)
        axc.scatter([m], [y], marker="D", s=20, color=FS.NEUTRAL, edgecolor="white", lw=0.5, zorder=3)
        continue
    for b, dy in ((0, 0.2), (PRIM_B, -0.02)):
        v = get(key, b)
        if v is None:
            continue
        m, lo, hi = v
        col = BUF_COLOUR[b]
        mk = "D" if key == "component" else "o"
        axc.plot([lo, hi], [y + dy, y + dy], color=col, lw=0.9, zorder=2)
        axc.scatter([m], [y + dy], marker=mk, s=18, color=col, edgecolor="white", lw=0.5, zorder=3)
    v = get(key, PRIM_B, control=True)
    if v is not None:
        m, lo, hi = v
        axc.plot([lo, hi], [y - 0.24, y - 0.24], color=FS.CONTROL, lw=0.8, ls=(0, (2, 1.4)), zorder=2)
        axc.scatter([m], [y - 0.24], marker="o", s=14, facecolor="white", edgecolor=FS.CONTROL, lw=0.9, zorder=3)
axc.set_yticks(range(n))
axc.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.2)
axc.tick_params(axis="y", length=0)
axc.set_xticks(np.arange(0.5, 1.001, 0.1))
axc.set_ylim(-0.6, n - 0.4)
axc.set_xlim(0.5, 0.95)
axc.set_xlabel("Balanced accuracy (mean of sites; whisker 95% CI)")
axc.text(r8020.balanced_accuracy - 0.004, n - 0.45, "random\n80/20", fontsize=5.8, color=FS.INK, ha="right", va="top",
         linespacing=0.9)
h = [Line2D([], [], marker="o", color=BUF_COLOUR[0], lw=0.9, ms=4, mec="white", label="no buffer"),
     Line2D([], [], marker="o", color=BUF_COLOUR[PRIM_B], lw=0.9, ms=4, mec="white", label=f"{PRIM_B} m buffer"),
     Line2D([], [], marker="o", color=FS.CONTROL, lw=0.8, ls=(0, (2, 1.4)), ms=3.6, mfc="white",
            label=f"control, {PRIM_B} m"),
     Line2D([], [], marker="D", color=FS.NEUTRAL, lw=0, ms=3.6, label="pooled leave-site-out"),
     Line2D([], [], color=FS.MUTED, lw=0.8, ls=(0, (1, 1.5)), label="primary design")]
axc.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12 * 2.3129 / 1.78), ncol=5, frameon=False,
           handlelength=2.0, columnspacing=1.0, fontsize=6)
axc.set_title("c  Block shape, assignment rule, K and grouped hold-outs", loc="left", fontweight="bold",
              x=-0.62 * 2.15 / C_W)
FS.clean(axc)

FS.save(fig, P.figures / "Figure_R1_spatialcv_accuracy.pdf")
print("wrote", P.figures / "Figure_R1_spatialcv_accuracy.pdf")
