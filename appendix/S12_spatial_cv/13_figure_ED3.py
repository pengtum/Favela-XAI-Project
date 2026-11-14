"""Step 13: Extended Data Fig. 3, site-level feature rankings under spatial cross-validation.

Manuscript items
* Extended Data Figure 3 (183 x ~210 mm):
  a  design schematic on one site with its real folds (ed3_schematic.py);
  b  out-of-fold importance share (mean over scored runs) against the main-text model's
     share, all scored sites pooled, one point per indicator per site: random five-fold
     (four seeds; reference) and, with the 200 m buffer, 250 and 500 m blocks (three grid
     placements) and favela components; r2 = squared Pearson correlation of the two
     shares (the line is 1:1, not a fit);
  c  site mean of the per-run Spearman rho with the main-text ranking (site-bootstrap 95%
     interval), sites scored / not scored and site-runs keeping all three top features,
     for the references, the 0 m and 200 m block ladders, the favela components and the
     three alternative 250 m units; null band = split-refit null 95% interval, dotted
     line = random five-fold mean.
* Caption numbers (ed3_caption_numbers.txt): Catumbi building counts, the share of
  training buildings removed by the buffer in the fold shown and across folds, points
  per scatter, r2 values.
Values come from step 11 (both scoring criteria).

Inputs  outputs/appendix/S12_spatial_cv/ranking/ranking_summary.csv, ranking_cells.csv, ranking_long.csv (step 11);
        site tables, component groups, shap_fold_sums (steps 01, 02, 10); data/raw/Limite_Favelas_2013
Outputs outputs/appendix/S12_spatial_cv/figures/Figure_R1_ED1_ranking_stability.pdf (+ _preview.png)
        outputs/appendix/S12_spatial_cv/ranking/ed3_plotted_values.csv, ed3_caption_numbers.txt
Run     python appendix/S12_spatial_cv/13_figure_ED3.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import MultiPoint, box  # noqa: E402

import ed3_schematic as SCH  # noqa: E402
import s12_common as C  # noqa: E402
from ed3_schematic import INK, MM, MUTED, RC  # noqa: E402

cfg, s, P = C.setup(__file__)
RKD = P.out / "ranking"
PRIM_B = int(cfg.get("spatial_cv.primary.buffer_m"))
plt.rcParams.update(RC)
plt.rcParams.update({"axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})

# ------------------------------------------------------------------ data (step 11, both criteria)
T = pd.read_csv(C.require(RKD / "ranking_summary.csv", "11_ranking_stability.py"))
SU = T.rename(columns={"final_rho": "rho", "final_rho_lo": "rho_lo", "final_rho_hi": "rho_hi", "final_sites": "n_sites",
                       "final_sites_list": "sites", "final_site_runs": "n_site_runs",
                       "final_top3_all": "site_runs_top3_all_kept"}).set_index("design")
LR = pd.read_csv(RKD / "ranking_long.csv")
SITES = list(cfg.sites)
# three-letter site codes of panel c (listed in the ED Fig. 3 caption)
CODE = {"barao": "BAR", "catumbi": "CAT", "faz_quem_quer": "FQQ", "fazena_coqueiro": "FCO",
        "grotao_de_costa_barros": "GCB", "jacarezinho": "JAC", "morro_da_guaiba": "MDG", "nova_cidade": "NCI",
        "palmeras": "PAL", "rocinha": "ROC"}


def not_scored(d):
    got = set(str(SU.loc[d, "sites"]).split(";"))
    return [CODE[x] for x in SITES if x not in got]


def agreement(x, y):
    """r2 (squared Pearson correlation) and Lin's concordance correlation of two share vectors."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    return r * r, float(2 * r * sx * sy / (sx ** 2 + sy ** 2 + (x.mean() - y.mean()) ** 2))


# ------------------------------------------------------------------ style
REFG, BANDC, LINEC, TRACK, OTHER = "#8c8b87", "#e4e3de", "#3c3c3a", "#eeede9", "#b3b2ac"
BLUE = {250: "#6baed6", 500: "#3182bd", 1000: "#08519c"}
COMPC = "#262626"
FS_LETTER, FS_TITLE, FS_AXLAB, FS_TABLE, FS_KEY, FS_STAT, FS_TICK, FS_SUB = 10.0, 8.0, 8.5, 7.5, 7.5, 7.5, 7.0, 7.0
CI_LW, CI_ALPHA = 0.5, 0.45
ALT_MARK = {"hex250_b200": ("h", 3.6), "kmm250_b200": ("p", 3.5), "sq250_K10_b200": ("o", 3.2)}
PLOT = []


def style(d):
    """Marker grammar of panel c: shape = held-out unit, colour = unit size, open = 0 m buffer."""
    if d in ("skf5_4seeds", "null_3splits"):
        return REFG, "o", 3.0
    if d.startswith("component"):
        return COMPC, "D", 2.8
    if d in ALT_MARK:
        return BLUE[250], *ALT_MARK[d]
    return BLUE[int(d[2:].split("_")[0])], "o", 3.2


# ------------------------------------------------------------------ frame (mm, y down)
D = SCH.load_data(cfg, s, P)
W = 183.0
A_H = round(SCH.content_height_mm(D, W) + 0.4, 1)
B0 = A_H + 2.4
B_GRP = B0 + 6.4
B_T1, B_T2 = B0 + 10.5, B0 + 13.5
B_AX = B0 + 16.2
BX0 = 13.0
N_B = 4
BG = 8.0
SQ = round((W - 1.0 - BX0 - (N_B - 1) * BG) / N_B, 2)
B_XL = B_AX + SQ + 5.6
B_K = B_XL + 5.0
B_N = B_K + 3.4
C0 = B_N + 4.8
C_KEY1, C_KEY2 = C0 + 6.6, C0 + 10.0
C_H1, C_H2 = C0 + 14.2, C0 + 17.4
ROW_TOP = C0 + 19.8
RH, HH = 3.5, 4.0
ROWS = [("hdr", "References"), ("skf5_4seeds", "Random five-fold"), ("null_3splits", "Split-refit null"),
        ("hdr", "0 m buffer"), ("sq250_b0", "250 m blocks"), ("sq500_b0", "500 m blocks"), ("sq1000_b0", "1,000 m blocks"),
        ("component_b0", "Favela components"),
        ("hdr", f"{PRIM_B} m buffer"), (f"sq250_b{PRIM_B}_3place", "250 m blocks"), (f"sq500_b{PRIM_B}_3place", "500 m blocks"),
        (f"sq1000_b{PRIM_B}_3place", "1,000 m blocks"), (f"component_b{PRIM_B}", "Favela components"),
        ("hex250_b200", "250 m hexagons"), ("kmm250_b200", "250 m k-means units"),
        ("sq250_K10_b200", "250 m blocks, ten folds")]
yy = ROW_TOP
RY = []
for r in ROWS:
    h = HH if r[0] == "hdr" else RH
    RY.append((yy, yy + h))
    yy += h
ROW_BOT = yy
C_XL = ROW_BOT + 6.2
H = round(C_XL + 3.0, 1)

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def Tx(x, y, txt, size=FS_TABLE, color=INK, ha="left", va="center", axx=None, **kw):
    return (axx or ax).text(x, y, txt, fontsize=size, color=color, ha=ha, va=va, **kw)


def mm_axes(x0, ytop, w, h):
    return fig.add_axes([x0 / W, 1 - (ytop + h) / H, w / W, h / H])


def text_w(t):
    return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / MM


# ================================================================== a
info = SCH.draw_schematic(fig, (0, 1 - A_H / H, 1, A_H / H), D, letter="a")
LX = info["x_mm"][0]

# ================================================================== b  scatters
Tx(LX, B0 + 2.0, "b", FS_LETTER, weight="bold")
Tx(LX + 4.2, B0 + 2.0, "Importance share per feature", FS_TITLE)
PANELS = [("skf5_4seeds", "Random five-fold", REFG, "ref"),
          (f"sq250_b{PRIM_B}_3place", "250 m blocks", BLUE[250], 250),
          (f"sq500_b{PRIM_B}_3place", "500 m blocks", BLUE[500], 500),
          (f"component_b{PRIM_B}", "Favela components", COMPC, "comp")]
PX = [BX0 + i * (SQ + BG) for i in range(N_B)]
LIM = 45.0


def bracket(x0, x1, label):
    ax.plot([x0, x0, x1, x1], [B_GRP + 2.6, B_GRP + 1.7, B_GRP + 1.7, B_GRP + 2.6], color="#9a9994", lw=0.5)
    Tx((x0 + x1) / 2, B_GRP, label, FS_KEY, MUTED, ha="center")


bracket(PX[0], PX[0] + SQ, "Reference")
bracket(PX[1], PX[-1] + SQ, f"Spatial cross-validation, {PRIM_B} m buffer")

K = 3.9 / 1000.0          # pictograms at true relative scale: 1 mm = 1000/3.9 m
PICT = 5.8
comp_poly = shapely.concave_hull(MultiPoint(D["xy"][D["sc"]["te"]]), ratio=0.25).buffer(15).simplify(10)
comp_poly = affinity.translate(comp_poly, -comp_poly.centroid.x, -comp_poly.centroid.y)
lim_m = PICT / 2 / K


def pictogram(x, kind, col):
    pa = mm_axes(x, (B_T1 + B_T2) / 2 - PICT / 2, PICT, PICT)
    pa.set_xlim(-lim_m, lim_m)
    pa.set_ylim(-lim_m, lim_m)
    pa.set_aspect("equal")
    pa.axis("off")
    if kind == "ref":
        rng = np.random.default_rng(3)
        g = np.linspace(-lim_m * 0.66, lim_m * 0.66, 5)
        GX, GY = np.meshgrid(g, g)
        f = np.zeros(25, bool)
        f[rng.choice(25, 5, replace=False)] = True
        pa.plot(GX.ravel()[~f], GY.ravel()[~f], "o", ms=1.3, mfc="none", mec=OTHER, mew=0.4)
        pa.plot(GX.ravel()[f], GY.ravel()[f], "o", ms=1.5, mfc=REFG, mec=REFG, mew=0)
        return
    shape = box(-kind / 2, -kind / 2, kind / 2, kind / 2) if kind != "comp" else comp_poly
    ring = shape.buffer(PRIM_B, quad_segs=16)
    pa.add_patch(MPoly(np.asarray(shape.exterior.coords), closed=True, fc=col, ec="none"))
    pa.plot(*ring.exterior.xy, color=INK, lw=0.45, ls=(0, (1.6, 1.0)))


scatter_axes, B_COUNTS = [], {}
for i, (d, name, col, kind) in enumerate(PANELS):
    x = PX[i]
    pictogram(x, kind, col)
    Tx(x + PICT + 1.2, (B_T1 + B_T2) / 2, name, FS_TITLE, weight="bold")
    a_ = mm_axes(x, B_AX, SQ, SQ)
    scatter_axes.append(a_)
    q = LR[LR.design == d]
    a_.plot([0, LIM], [0, LIM], color="#8a8986", lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    rest, top = q[q.rank_pub > 3], q[q.rank_pub <= 3]
    a_.scatter(100 * rest.share_pub, 100 * rest.share_run, s=3.0, facecolors="none", edgecolors=OTHER, linewidths=0.35,
               zorder=2)
    mk = "D" if kind == "comp" else "o"
    a_.scatter(100 * top.share_pub, 100 * top.share_run, s=7.5 if mk == "o" else 6.0, marker=mk, c=col,
               edgecolors="white", linewidths=0.3, zorder=3)
    a_.set_xlim(0, LIM)
    a_.set_ylim(0, LIM)
    a_.set_aspect("equal")
    a_.set_xticks([0, 20, 40])
    a_.set_yticks([0, 20, 40])
    a_.tick_params(length=1.6, width=0.4, pad=1.2, labelsize=FS_TICK)
    if i > 0:
        a_.set_yticklabels([])
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a_.spines[sp].set_linewidth(0.45)
    r2_, ccc_ = agreement(100 * q.share_pub, 100 * q.share_run)
    a_.text(0.035, 0.975, f"r² = {r2_:.3f}", transform=a_.transAxes, ha="left", va="top", fontsize=FS_STAT, color=INK)
    B_COUNTS[d] = (q.site.nunique(), len(q))
    PLOT += [dict(panel="b", item=f"{d} r2", value=r2_), dict(panel="b", item=f"{d} ccc", value=ccc_),
             dict(panel="b", item=f"{d} points", value=len(q))]
    PLOT += [dict(panel="b", item=f"{d} {rr.site} {rr.feature} (main, oof %)",
                  value=f"{100 * rr.share_pub:.6f},{100 * rr.share_run:.6f}") for rr in q.itertuples()]

Tx(PX[0] + (PX[-1] + SQ - PX[0]) / 2, B_XL, "Main-text model share (%)", FS_AXLAB, ha="center")
Tx(3.6, B_AX + SQ / 2, "Out-of-fold share (%)", FS_AXLAB, ha="center", rotation=90)
kx = PX[0]
for j, (c_, m_) in enumerate([(BLUE[250], "o"), (BLUE[500], "o"), (COMPC, "D")]):
    ax.plot([kx + 0.9 + j * 2.4], [B_K], marker=m_, ms=2.9 if m_ == "o" else 2.5, mfc=c_, mec="white", mew=0.3, ls="none")
t = Tx(kx + 7.8, B_K, "each site's top 3 in the main-text model", FS_KEY)
kx += 7.8 + text_w(t) + 4.0
ax.plot([kx + 0.9], [B_K], marker="o", ms=2.1, mfc="none", mec=OTHER, mew=0.45, ls="none")
t = Tx(kx + 2.4, B_K, "other 22 features", FS_KEY)
kx += 2.4 + text_w(t) + 4.0
ax.plot([kx, kx + 4.0], [B_K, B_K], color="#8a8986", lw=0.5, ls=(0, (2.5, 1.5)))
t = Tx(kx + 4.8, B_K, "1:1", FS_KEY)
kx += 4.8 + text_w(t) + 4.0
pk = mm_axes(kx, B_K - 2.2, 4.4, 4.4)
pk.set_xlim(-560, 560)
pk.set_ylim(-560, 560)
pk.set_aspect("equal")
pk.axis("off")
sq_ = box(-125, -125, 125, 125)
pk.add_patch(MPoly(np.asarray(sq_.exterior.coords), closed=True, fc=BLUE[250], ec="none"))
pk.plot(*sq_.buffer(PRIM_B, quad_segs=16).exterior.xy, color=INK, lw=0.45, ls=(0, (1.6, 1.0)))
Tx(kx + 5.0, B_K, f"held-out unit and {PRIM_B} m buffer", FS_KEY)

# ================================================================== c  forest
Tx(LX, C0 + 2.0, "c", FS_LETTER, weight="bold")
Tx(LX + 4.2, C0 + 2.0, "Agreement with the main-text model ranking", FS_TITLE)
X_LAB = LX
X_SITES, X_NS = 41.0, 47.0
RHO_X, RHO_W = 70.0, 74.0
TOP_X, TOP_W = 149.0, 19.0


def kmark(x, y, c_, m_="o", ms=3.0, open_=False):
    ax.plot([x], [y], marker=m_, ms=ms, mfc="white" if open_ else c_, mec=c_, mew=0.75 if open_ else 0.3, ls="none")


kx, ky = X_LAB, C_KEY1
kmark(kx + 0.9, ky, INK, open_=True)
t = Tx(kx + 2.3, ky, "0 m buffer", FS_KEY)
kx += 2.3 + text_w(t) + 3.4
kmark(kx + 0.9, ky, INK)
t = Tx(kx + 2.3, ky, f"{PRIM_B} m buffer", FS_KEY)
kx += 2.3 + text_w(t) + 4.6
for j, L in enumerate((250, 500, 1000)):
    kmark(kx + 0.9 + j * 2.4, ky, BLUE[L])
t = Tx(kx + 7.6, ky, "250, 500, 1,000 m blocks", FS_KEY)
kx += 7.6 + text_w(t) + 4.0
for m_, ms_, c_, lab in (("h", 3.4, BLUE[250], "hexagons"), ("p", 3.3, BLUE[250], "k-means units"),
                         ("D", 2.6, COMPC, "favela components"), ("o", 3.0, REFG, "references")):
    kmark(kx + 0.9, ky, c_, m_, ms_)
    t = Tx(kx + 2.3, ky, lab, FS_KEY)
    kx += 2.3 + text_w(t) + 4.0
kx, ky = X_LAB, C_KEY2
ax.plot([kx + 1.0, kx + 1.0], [ky - 1.1, ky + 1.1], color=LINEC, lw=0.7, ls=(0, (2.0, 1.2)))
t = Tx(kx + 2.3, ky, "random five-fold mean", FS_KEY)
kx += 2.3 + text_w(t) + 4.0
ax.add_patch(Rectangle((kx, ky - 1.0), 3.0, 2.0, facecolor=BANDC, edgecolor="none"))
t = Tx(kx + 3.8, ky, "split-refit null 95% CI", FS_KEY)
kx += 3.8 + text_w(t) + 4.0
ax.plot([kx, kx + 3.2], [ky, ky], color=MUTED, lw=CI_LW, alpha=CI_ALPHA)
Tx(kx + 4.0, ky, "site-bootstrap 95% CI", FS_KEY)

Tx(X_LAB, C_H1, "Design", FS_TABLE, weight="bold")
Tx(X_SITES, C_H1, "Sites", FS_TABLE, ha="center", weight="bold")
Tx(X_SITES, C_H2, "scored", FS_SUB, MUTED, ha="center")
Tx(X_NS, C_H1, "Not scored", FS_TABLE, weight="bold")
Tx(RHO_X, C_H1, "Rank ρ with main-text model", FS_TABLE, weight="bold")
Tx(TOP_X, C_H1, "All top 3 kept", FS_TABLE, weight="bold")
ax.plot([X_LAB, W - 1.0], [ROW_TOP - 0.7] * 2, color="#9a9994", lw=0.5)
ax_r = mm_axes(RHO_X, ROW_TOP, RHO_W, ROW_BOT - ROW_TOP)
ax_t = mm_axes(TOP_X, ROW_TOP, TOP_W, ROW_BOT - ROW_TOP)
for a_, lim in ((ax_r, (0.7, 1.0)), (ax_t, (0, 1))):
    a_.set_ylim(ROW_BOT, ROW_TOP)
    a_.set_xlim(*lim)
    a_.patch.set_alpha(0)
    a_.set_yticks([])
    for sp in ("top", "right", "left"):
        a_.spines[sp].set_visible(False)
    a_.spines["bottom"].set_linewidth(0.5)
    a_.tick_params(axis="x", length=1.8, width=0.5, pad=1.2, labelsize=FS_TICK)
ax_r.set_xticks([0.7, 0.8, 0.9, 1.0])
ax_r.set_xticklabels(["0.7", "0.8", "0.9", "1.0"])
ax_t.spines["bottom"].set_visible(False)
ax_t.set_xticks([])
for tv in (0.75, 0.8, 0.85, 0.9, 0.95):
    ax_r.axvline(tv, color="#efeeea", lw=0.5, zorder=0)
Tx(RHO_X + RHO_W / 2, C_XL, "Spearman ρ", FS_AXLAB, ha="center")
sk, nl = SU.loc["skf5_4seeds"], SU.loc["null_3splits"]
ax_r.axvspan(nl.rho_lo, nl.rho_hi, color=BANDC, lw=0, zorder=0.5)
ax_r.axvline(sk.rho, color=LINEC, lw=0.7, ls=(0, (2.0, 1.2)), zorder=1)

for k, ((d, lab), (y0, y1)) in enumerate(zip(ROWS, RY)):
    yc = (y0 + y1) / 2
    if d == "hdr":
        if k > 0:
            ax.plot([X_LAB, W - 1.0], [y0 + 0.15] * 2, color="#d6d5d0", lw=0.45)
        Tx(X_LAB, yc + 0.35, lab, FS_TABLE, weight="bold")
        continue
    r = SU.loc[d]
    col, mk, ms = style(d)
    zero = d.endswith("_b0")
    Tx(X_LAB + 1.2, yc, lab, FS_TABLE)
    Tx(X_SITES, yc, f"{int(r.n_sites)}", FS_TABLE, ha="center")
    ns = not_scored(d)
    Tx(X_NS, yc, ", ".join(ns) if ns else "–", FS_SUB, MUTED if ns else "#b0afa9")
    if np.isfinite(r.rho):
        ax_r.plot([r.rho_lo, r.rho_hi], [yc, yc], color=col, lw=CI_LW, alpha=CI_ALPHA, solid_capstyle="butt", zorder=3)
        ax_r.plot(r.rho, yc, marker=mk, ms=ms, mfc="white" if zero else col, mec=col if zero else "white",
                  mew=0.8 if zero else 0.35, ls="none", zorder=5)
    kk, nn = int(r.site_runs_top3_all_kept), int(r.n_site_runs)
    ax_t.add_patch(Rectangle((0, yc - 0.95), 1, 1.9, facecolor=TRACK, edgecolor="none", zorder=1))
    if nn:
        face = matplotlib.colors.to_rgba(col, 0.28) if zero else col
        ax_t.add_patch(Rectangle((0, yc - 0.95), kk / nn, 1.9, facecolor=face, edgecolor=col if zero else "none",
                                 lw=0.5, zorder=2))
    Tx(TOP_X + TOP_W + 1.2, yc, f"{kk}/{nn}", FS_KEY)
    PLOT += [dict(panel="c", item=f"{d} {k_}", value=v_) for k_, v_ in
             (("n_sites", int(r.n_sites)), ("not scored", ";".join(ns)), ("rho", r.rho), ("rho_lo", r.rho_lo),
              ("rho_hi", r.rho_hi), ("top3 all kept", kk), ("site-runs", nn))]

# ================================================================== layout check and save
fig.canvas.draw()
rend = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
boxes = [(t, t.get_window_extent(rend)) for t in texts]
fb = fig.bbox
outside = [t.get_text() for t, b in boxes if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5
           or b.y1 > fb.y1 + 0.5]
coll = [(boxes[i][0].get_text()[:40], boxes[j][0].get_text()[:40]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        if min(boxes[i][1].x1, boxes[j][1].x1) - max(boxes[i][1].x0, boxes[j][1].x0) > 0.8
        and min(boxes[i][1].y1, boxes[j][1].y1) - max(boxes[i][1].y0, boxes[j][1].y0) > 0.8]
if outside or coll or H > 210.0:
    warnings.warn(f"ED Fig. 3 layout: page {W} x {H} mm; texts outside the frame {outside}; overlaps {coll}")
path = P.figures / "Figure_R1_ED1_ranking_stability.pdf"
fig.savefig(path)
fig.savefig(path.with_name(path.stem + "_preview.png"), dpi=1400.0 / (W * MM))
pd.DataFrame(PLOT).to_csv(RKD / "ed3_plotted_values.csv", index=False)

# ================================================================== caption numbers
cnt = D["counts"].set_index("panel")
iii = cnt.loc["iii"]
pct = D["removed_pct_by_fold"]
lines = [f"a: {D['site_name']} ({D['info']['n']:,} buildings, {D['info']['n_fav']:,} favela); extent "
         f"{D['info']['extent_km'][0]:.2f} x {D['info']['extent_km'][1]:.2f} km (manuscript: Catumbi 22,804 buildings, 9,248 favela)",
         f"a(iii): {100 * iii.n_removed / (iii.n_train_kept + iii.n_removed):.0f}% of the training buildings removed in the fold "
         f"shown; {min(pct):.0f}-{max(pct):.0f}% across the {len(pct)} folds (manuscript: 54% here; 18-79% across folds)",
         "b: points per scatter " + "; ".join(f"{d} {n} ({ns_} sites)" for d, (ns_, n) in B_COUNTS.items())
         + " (manuscript: 250 five-fold; 225 each block size; 175 components)"]
lines += [f"b: {r_['item']} = {r_['value']:.3f}" for r_ in PLOT if r_["panel"] == "b" and r_["item"].endswith(" r2")]
(RKD / "ed3_caption_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
D["counts"].to_csv(RKD / "ed3_schematic_counts.csv", index=False)
print("\n".join(lines))
print("wrote", path)
