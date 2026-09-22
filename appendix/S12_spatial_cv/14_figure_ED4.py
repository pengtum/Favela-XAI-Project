"""Step 14: Extended Data Fig. 4, site-level importance shares under spatial cross-validation.

Manuscript items
* Extended Data Figure 4: one panel per site,
  ordered by favela building count; each point one of the 25 indicators; x = its
  favela-only importance share in the main-text model (share of mean |SHAP| over the
  site's favela buildings; ED Fig. 2), y = its out-of-fold share averaged over the site's
  scored runs, for 250, 500 and 1,000 m blocks with the 200 m buffer (up to three grid
  placements; blue circles) and random five-fold cross-validation (four seeds; grey
  crosses); labels = the site's three leading indicators in the main-text model; sites
  with no scored 200 m design are marked 'not scored'.
* Caption numbers (ed4_caption_numbers.txt): the order of the panels (favela building
  counts), the site means of Spearman's rho for the three block sizes, and the favela
  coverage of the sites that are not scored (per grid placement and mean).
Values come from step 11 (both scoring criteria).

Inputs  outputs/appendix/S12_spatial_cv/ranking/ranking_cells.csv, ranking_long.csv (step 11)
Outputs outputs/appendix/S12_spatial_cv/figures/Figure_R1_persite_ranking.pdf (+ _preview.png)
        outputs/appendix/S12_spatial_cv/ranking/ed4_plotted_values.csv, ed4_caption_numbers.txt
Run     python appendix/S12_spatial_cv/14_figure_ED4.py
"""
import itertools
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from favela_xai.features import label_of  # noqa: E402

import s12_common as C  # noqa: E402

cfg, s, P = C.setup(__file__)
RKD = P.out / "ranking"
PRIM_B = int(cfg.get("spatial_cv.primary.buffer_m"))
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 5.5,
                     "axes.linewidth": 0.45, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "savefig.dpi": 600})
MM = 1 / 25.4
INK, MUTED = "#222222", "#5f5f5f"
REFG, LINE11, LEADER = "#8c8b87", "#8a8986", "#9a9994"
BLUE = {250: "#6baed6", 500: "#3182bd", 1000: "#08519c"}

LR = pd.read_csv(C.require(RKD / "ranking_long.csv", "11_ranking_stability.py"))
CE = pd.read_csv(RKD / "ranking_cells.csv")
DES = [(L, f"sq{L}_b{PRIM_B}_3place") for L in (250, 500, 1000)]
REF = "skf5_4seeds"
sites = (CE[["site", "site_name", "n_favela"]].drop_duplicates().sort_values("n_favela", ascending=False)
         .reset_index(drop=True))
PUB = LR[LR.design == "published"]
LIM = 45.0


def cell(site, design):
    return CE[(CE.site == site) & (CE.design == design)].iloc[0]


def coverage(site, design):
    """Favela coverage per grid placement after the first criterion (runs with a fitted fold)."""
    txt = str(cell(site, design).fav_explained_by_run)
    pairs = [x.split("=") for x in txt.split(";") if "=" in x]
    return [a for a, _ in pairs], [float(b) for _, b in pairs]


# ------------------------------------------------------------------ frame (mm, y down)
W = 183.0
L_M, GAP = 6.4, 3.9
HEAD_W = 34.6
PW = (W - L_M - 4 * GAP - HEAD_W) / 4       # panel width
R_M = HEAD_W - PW
HEAD = 8.6 - 5.08
ROW0, TICKS, ROWGAP = 0.6, 3.0, 2.2
ROW_TOPS = [ROW0, ROW0 + HEAD + PW + TICKS + ROWGAP]
XL_Y = ROW_TOPS[1] + HEAD + PW + 4.3
LEG_Y = XL_Y + 4.4
H = round(LEG_Y + 2.4, 1)
K = PW / LIM                                 # mm per percentage point (square axes)

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
REND = fig.canvas.get_renderer()


def T(x, y, txt, size=5.5, color=INK, ha="left", va="center", axx=None, **kw):
    return (axx or ax).text(x, y, txt, fontsize=size, color=color, ha=ha, va=va, **kw)


def text_wh(txt, size=5.5, **kw):
    t = fig.text(0, 0, txt, fontsize=size, **kw)
    b = t.get_window_extent(REND)
    t.remove()
    return b.width / fig.dpi / MM, b.height / fig.dpi / MM


def mm_axes(x0, ytop, w, h):
    return fig.add_axes([x0 / W, 1 - (ytop + h) / H, w / W, h / H])


PLOT, CAP = [], []
for i, srow in sites.iterrows():
    r_, c_ = divmod(i, 5)
    x0 = L_M + c_ * (PW + GAP)
    y0 = ROW_TOPS[r_]
    site, name = srow.site, srow.site_name
    t1 = T(x0, y0 + 1.6, name, 6.5, weight="bold")
    if all(cell(site, d).status != "ok" for _, d in DES):
        T(x0 + t1.get_window_extent(REND).width / fig.dpi / MM + 1.4, y0 + 1.6, "not scored", 5.5, MUTED)
    a = mm_axes(x0, y0 + HEAD, PW, PW)
    a.plot([0, LIM], [0, LIM], color=LINE11, lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    pts, scored = [], []
    pub = PUB[PUB.site == site].set_index("feature")
    for L, d in DES:
        q = LR[(LR.site == site) & (LR.design == d)]
        if cell(site, d).status != "ok" or not len(q):
            continue
        a.scatter(100 * q.share_pub, 100 * q.share_run, s=5.0, marker="o", c=BLUE[L], edgecolors="none", linewidths=0,
                  zorder=2 + L / 1000, alpha=0.95, label=d)
        scored.append(d)
        pts += [(100 * rr.share_pub, 100 * rr.share_run, d) for rr in q.itertuples()]
    q = LR[(LR.site == site) & (LR.design == REF)]
    a.scatter(100 * q.share_pub, 100 * q.share_run, s=6.5, marker="x", c=REFG, linewidths=0.5, zorder=4, label=REF)
    pts += [(100 * rr.share_pub, 100 * rr.share_run, REF) for rr in q.itertuples()]
    PLOT += [dict(site=site, design=dd, x_share_main_pct=px, y_share_oof_pct=py) for px, py, dd in pts]
    a.set_xlim(0, LIM)
    a.set_ylim(0, LIM)
    a.set_aspect("equal")
    a.set_xticks([0, 20, 40])
    a.set_yticks([0, 20, 40])
    a.tick_params(length=1.6, width=0.4, pad=1.2, labelsize=5.5)
    if c_ > 0:
        a.set_yticklabels([])
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    # ---- label the site's three leading indicators (greedy collision-free placement, leader lines when displaced)
    obst = [(K * px - 0.55, K * py - 0.55, K * px + 0.55, K * py + 0.55) for px, py, _ in pts]
    blocks = []
    if not scored:
        # keep the upper-left area of an unscored panel free of labels (sized by a coverage note that is
        # measured and then removed)
        note = ["Not scored at 200 m buffers", "(<50% of favela", "buildings covered).", "Mean favela coverage,",
                "250 / 500 / 1,000 m:", " / ".join(f"{100 * np.mean(coverage(site, d)[1] or [0]):.1f}" for _, d in DES) + "%"]
        tn = a.text(0.9, LIM - 0.6, "\n".join(note), fontsize=5.5, color=MUTED, ha="left", va="top", linespacing=1.2)
        fig.canvas.draw()
        bb = tn.get_window_extent(REND)
        (bx0, by0), (bx1, by1) = a.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        blocks.append((K * bx0, K * by0, K * bx1, K * by1))
        tn.remove()
    top = pub[pub.rank_pub <= 3].sort_values("rank_pub")
    MK = {f: [(K * px, K * py) for px, py, _ in pts if abs(px - 100 * rr.share_pub) < 1e-9] for f, rr in top.iterrows()}

    def yspan(f):
        yy_ = [K * py for px, py, dd in pts if abs(px - 100 * top.loc[f, "share_pub"]) < 1e-9 and (dd != REF or not scored)]
        return min(yy_), max(yy_)

    def close(f1, f2):                   # < 1 mm apart in x and marker y-ranges within 1.5 mm: one stacked label
        (a0, a1), (b0, b1) = yspan(f1), yspan(f2)
        return abs(K * 100 * (top.loc[f1, "share_pub"] - top.loc[f2, "share_pub"])) < 1.0 and max(a0, b0) - min(a1, b1) < 1.5

    groups = []
    for feat in top.index:
        g = next((g for g in groups if any(close(feat, f) for f in g)), None)
        (g.append(feat) if g is not None else groups.append([feat]))

    def box_pt(b, p):
        return float(np.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3])))

    placed = list(blocks)
    for g in groups:
        own = [p for f in g for p in MK[f]]
        oth = [p for f in MK if f not in g for p in MK[f]]
        xs = float(np.mean([K * 100 * top.loc[f, "share_pub"] for f in g]))
        ylo, yhi = min(p[1] for p in own), max(p[1] for p in own)
        ym = (ylo + yhi) / 2
        text = "\n".join(label_of(f) for f in g)
        w, _ = text_wh(text, linespacing=1.1)
        h = 1.7 * len(g) + 0.2 * (len(g) - 1)
        cands = [(xs + dx if side == "r" else xs - dx - w, ym + dy - h / 2, 0.0)
                 for side, dx, dy in itertools.product(("r", "l"), (0.9, 1.8, 3.0, 4.5, 6.5, 9.0, 12.0),
                                                       (0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, 6, -6, 8, -8, 10, -10, 12,
                                                        -12, 15, -15, 18, -18))]
        for dx, dy in itertools.product((-w / 2, -w + 1.0, -1.0, -w / 2 + 3, -w / 2 - 3), (0.6, 1.5, 3.0, 5.0, 8.0)):
            cands += [(xs + dx, yhi + dy, 0.5), (xs + dx, ylo - dy - h, 0.5)]
        best = None
        for bx0, by0, pref in cands:
            bx1, by1 = bx0 + w, by0 + h
            b = (bx0, by0, bx1, by1)
            if bx0 < 0.3 or by0 < 0.3 or bx1 > PW - 0.2 or by1 > PW - 0.2:
                continue
            if any(bx0 < o[2] + 0.35 and o[0] - 0.35 < bx1 and by0 < o[3] + 0.25 and o[1] - 0.25 < by1 for o in placed):
                continue
            nhit = sum(bx0 < o[2] and o[0] < bx1 and by0 < o[3] and o[1] < by1 for o in obst)
            diag = (bx0 - 0.3 <= by1) and (by0 - 0.3 <= bx1)
            own_d = min(box_pt(b, p) for p in own)
            amb = int(bool(oth) and min(box_pt(b, p) for p in oth) < own_d + 1.0)
            leader = own_d > 1.3
            stack = any(bx0 < o[2] and o[0] < bx1 and (by0 - o[3] < 1.0 and o[1] - by1 < 1.0) for o in placed[len(blocks):])
            cost = own_d + 25 * nhit + 10 * diag + 1.5 * leader + 8 * amb + pref + 2.0 * stack
            if best is None or cost < best[0]:
                best = (cost, b, leader)
        if best is None:
            warnings.warn(f"ED Fig. 4: no free place for the label of {g} in {name}")
            continue
        _, (bx0, by0, bx1, by1), leader = best
        placed.append((bx0, by0, bx1, by1))
        a.text(bx0 / K, (by0 + by1) / 2 / K, text, fontsize=5.5, color=INK, ha="left", va="center", linespacing=1.1,
               zorder=6)
        if leader:
            b = (bx0, by0, bx1, by1)
            mx, my = min(own, key=lambda p: box_pt(b, p))
            tx, ty = min(max(mx, bx0), bx1), min(max(my, by0), by1)
            vx, vy = tx - mx, ty - my
            nv = np.hypot(vx, vy)
            a.plot([(mx + 0.7 * vx / nv) / K, (tx - 0.25 * vx / nv) / K],
                   [(my + 0.7 * vy / nv) / K, (ty - 0.25 * vy / nv) / K], color=LEADER, lw=0.35, zorder=5,
                   solid_capstyle="butt")

    # caption numbers of this site
    vals = [f"{cell(site, d).rho:.2f}" if cell(site, d).status == "ok" else "n.s." for _, d in DES]
    if all(v == "n.s." for v in vals):
        cov = [coverage(site, d)[1] for _, d in DES]
        CAP.append(f"{name} is not scored under any {PRIM_B} m design (mean favela coverage "
                   + ", ".join(f"{100 * np.mean(c or [0]):.1f}" for c in cov) + "% for the three block sizes; per placement: "
                   + "; ".join(f"{L:,} m, " + ", ".join(f"{100 * v:.1f}" for v in c) + "%" for (L, _), c in zip(DES, cov)) + ")")
    else:
        CAP.append(f"{name} {'/'.join(vals)}")

for r_ in range(2):
    T(1.5, ROW_TOPS[r_] + HEAD + PW / 2, "Out-of-fold share (%)", 5.5, ha="center", rotation=90)
T(L_M + (5 * PW + 4 * GAP) / 2, XL_Y, "Main-text model favela-only share (%)", 5.5, ha="center")
items = [("o", BLUE[250], "250 m blocks"), ("o", BLUE[500], "500 m blocks"), ("o", BLUE[1000], "1,000 m blocks"),
         ("x", REFG, "Random five-fold"), ("line", LINE11, "1:1")]
head = f"With {PRIM_B} m buffer:"
LAB_TXT = "Labels: top 3 features in the main-text model"
total = (text_wh(head)[0] + 2.2 + sum(2.6 + text_wh(lb)[0] + 3.6 for _, _, lb in items) - 3.6 + 1.6 + 3.0
         + 4.0 + text_wh(LAB_TXT)[0])
kx = (W - total) / 2
T(kx, LEG_Y, head, 5.5)
kx += text_wh(head)[0] + 2.2
for j, (m_, col, lb) in enumerate(items):
    if j == 3:
        kx += 1.6
    if m_ == "line":
        ax.plot([kx - 0.4, kx + 3.0], [LEG_Y, LEG_Y], color=col, lw=0.5, ls=(0, (2.5, 1.5)))
        kx += 3.0
        T(kx + 1.2, LEG_Y, lb)
        kx += 1.2 + text_wh(lb)[0] + 3.6
        continue
    if m_ == "o":
        ax.plot([kx + 0.6], [LEG_Y], marker="o", ms=2.24, mfc=col, mec="none", ls="none")
    else:
        ax.plot([kx + 0.6], [LEG_Y], marker="x", ms=2.55, mec=col, mew=0.5, ls="none")
    T(kx + 2.0, LEG_Y, lb)
    kx += 2.0 + text_wh(lb)[0] + 3.6
T(kx + 0.4, LEG_Y, LAB_TXT, 5.5, MUTED)

fig.canvas.draw()
boxes = [(t, t.get_window_extent(REND)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
coll = [1 for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        if min(boxes[i][1].x1, boxes[j][1].x1) - max(boxes[i][1].x0, boxes[j][1].x0) > 0.5
        and min(boxes[i][1].y1, boxes[j][1].y1) - max(boxes[i][1].y0, boxes[j][1].y0) > 0.5]
if coll:
    warnings.warn(f"ED Fig. 4: {len(coll)} overlapping text boxes")
path = P.figures / "Figure_R1_persite_ranking.pdf"
fig.savefig(path)
fig.savefig(path.with_name(path.stem + "_preview.png"), dpi=1400.0 / (W * MM))
pd.DataFrame(PLOT).to_csv(RKD / "ed4_plotted_values.csv", index=False)
first, last = sites.iloc[0], sites.iloc[-1]
lines = [f"ordered by favela building count ({first.site_name} {first.n_favela:,} to {last.site_name} {last.n_favela:,})",
         "site means of Spearman's rho (250/500/1,000 m): " + "; ".join(c for c in CAP if " is not scored" not in c)]
lines += [c for c in CAP if " is not scored" in c]
(RKD / "ed4_caption_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("wrote", path)
