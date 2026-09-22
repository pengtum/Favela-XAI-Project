"""Panel a of Extended Data Fig. 3: the spatial cross-validation design drawn on one site with its real folds.

Imported by 13_figure_ED3.py; not a step of its own. Manuscript item: ED Fig. 3a,
the design on one site (Catumbi): (i) 250 m blocks assigned whole to five folds;
(ii) one fold held out; (iii) training buildings within 200 m of the test buildings
(the indicator neighbourhood) removed; (iv) other buffered held-out units (1,000 m
block; favela component with its nearest formal buildings).

The folds are those of the analysis (s12_common.make_design: 250 m and 1,000 m blocks,
first grid placement; the held-out favela component groups of step 02) and the buffer
is favela_xai.spatial_cv.buffer_train. The site, the folds and the component group
shown are set in ``appendix_S12_spatial_cv.ed3_schematic``. Every drawn split is
checked to partition the site (test + training kept + removed = all buildings) and its
counts are compared with the fold rows of the attribution runs (step 10). The counts
quoted in the ED Fig. 3 caption are returned in ``counts`` (written by step 13).
"""
import warnings

import contourpy
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint

from favela_xai import spatial_cv as scv
from favela_xai.config import load_config
from favela_xai.io import read_favela_polygons

import s12_common as C

# Buffer drawn in panels iii-iv: the primary buffer of SI S12 (spatial_cv.primary.buffer_m,
# 200 m, the indicator neighbourhood of Methods 4.5)
BUF = float(load_config().get("spatial_cv.primary.buffer_m"))
HULL_PAD = 12.0    # m; the site-extent outline sits just outside the outermost building centroids
SITE_EDGE = "#1a1a1a"

# ------------------------------------------------------------------ style (shared with 13_figure_ED3.py)
MM = 1 / 25.4
INK, MUTED = "#222222", "#5f5f5f"
ACC = "#D55E00"                        # test buildings
TRAIN = "#4a4a4a"                      # training buildings kept
REMOVED = "#bdbdbd"                    # removed by the buffer
REMOVED_EDGE = "#8a8a8a"
BLUE250, BLUE1000 = "#6baed6", "#08519c"
FOLDC = ["#7fb7ae", "#a99bcb", "#d9c38a", "#c99aaf", ACC]
FS_TITLE, FS_TEXT, FS_SMALL, FS_LETTER = 8.0, 7.5, 7.0, 10.0
RC = {"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS_TEXT,
      "axes.linewidth": 0.4, "savefig.dpi": 600}
LEFT, G_ARROW, G_SMALL = 0.6, 8.6, 4.0      # gaps (mm): i|ii and ii|iii carry arrows; iii|iv small
MAP_CAP = 32.5                               # map height cap (mm)
HEAD_Y = 2.4
TITLE_Y, SUB_Y, MAP_TOP = 6.2, 8.9, 9.0
UNDER, LEG = 4.6, 3.1

_DATA = None


def check(ok, message):
    """Layout checks are reported as warnings, not enforced (fonts may differ between systems)."""
    if not ok:
        warnings.warn(f"ED Fig. 3a layout: {message}")


def load_data(cfg, s, p):
    """Folds, splits and polygons of the schematic site (cached)."""
    global _DATA
    if _DATA is not None:
        return _DATA
    E = C.section(cfg)["ed3_schematic"]
    site, f250, f1000, group = E["site"], int(E["fold_250"]), int(E["fold_1000"]), int(E["component_group"])
    d = C.load_site(p, site)
    y, xy = d.y, d.xy
    info = dict(n=d.n, n_fav=d.n_favela, n_formal=int(d.n - d.n_favela),
                extent_km=tuple(round(float(v) / 1000, 2) for v in np.ptp(xy, axis=0)))
    hull = MultiPoint(xy).convex_hull.buffer(HULL_PAD, quad_segs=4)
    sp250, meta250 = C.make_design("sq250", d, s, cfg, p, placement=0)
    sp1000, meta1000 = C.make_design("sq1000", d, s, cfg, p, placement=0)
    fold_of = np.empty(d.n, int)
    for f, _, te in sp250:
        fold_of[te] = f

    def split(sp, f, b):
        _, tr, te = sp[f]
        kept, nd, _ = scv.buffer_train(xy, tr, te, b)
        return dict(tr=tr, te=te, kept=kept, removed=np.setdiff1d(tr, kept), nd=nd)

    s250_0, s250, s1k = split(sp250, f250, 0), split(sp250, f250, BUF), split(sp1000, f1000, BUF)
    splits_c, codes, _ = C.component_splits(C.load_groups(p, site))
    gi = codes.index(group)
    sc = split(splits_c, gi, BUF)
    # share of training buildings removed by the buffer in every 250 m fold (ED Fig. 3 caption)
    removed_pct = [100 * split(sp250, f, BUF)["nd"] / len(sp250[f][1]) for f in range(len(sp250))]

    # counts of the drawn splits, checked against the fold rows of the attribution runs (step 10)
    fs = {}
    b = int(BUF)
    for run in ("sq250_b0_p0", f"sq250_b{b}_p0", f"sq1000_b{b}_p0", f"component_b{b}"):
        f_ = p.shap_folds / f"{site}__{run}.parquet"
        if f_.exists():
            fs[run] = pd.read_parquet(f_).set_index("fold")
    rows = []
    for panel, run, f, sp_ in (("ii", "sq250_b0_p0", f250, s250_0), ("iii", f"sq250_b{b}_p0", f250, s250),
                               ("iv (top)", f"sq1000_b{b}_p0", f1000, s1k), ("iv (bottom)", f"component_b{b}", gi, sc)):
        parts = np.concatenate([sp_["te"], sp_["kept"], sp_["removed"]])
        assert len(parts) == d.n and len(np.unique(parts)) == d.n, (panel, "split does not partition the site")
        got = (len(sp_["te"]), len(sp_["kept"]), sp_["nd"])
        if run in fs and f in fs[run].index:
            ref = fs[run].loc[f]
            check((int(ref.n_test), int(ref.n_train), int(ref.n_dropped)) == got, f"{panel} counts differ from {run}")
        rows.append(dict(panel=panel, site=site, design=run, fold_index=f, n_test=got[0],
                         n_test_favela=int(y[sp_["te"]].sum()), n_train_kept=got[1], n_removed=got[2], n_site=d.n))
    counts = pd.DataFrame(rows)

    P_ = read_favela_polygons(cfg, 2013)
    pad = 25
    x0, y0 = xy.min(0) - pad
    x1, y1 = xy.max(0) + pad
    Pc = P_.cx[x0:x1, y0:y1]
    _DATA = dict(site=site, site_name=cfg.site_name(site), y=y, xy=xy, fold_of=fold_of, s250_0=s250_0, s250=s250,
                 s1k=s1k, sc=sc, counts=counts, removed_pct_by_fold=removed_pct, Pc=Pc, ext=(x0, y0, x1, y1),
                 info=info, hull=hull, n_blocks250=int(meta250["n_units"]), n_blocks1000=int(meta1000["n_units"]),
                 n_groups=len(codes), drawn=[], drawn_outline=[])
    return _DATA


class _MM:
    """Placement helper in millimetres inside a figure region."""

    def __init__(self, fig, region):
        self.fig = fig
        Wf, Hf = fig.get_size_inches() / MM
        self.Wf, self.Hf = Wf, Hf
        bb = region if isinstance(region, Bbox) else Bbox.from_bounds(*region)
        self.l, self.b = bb.x0 * Wf, bb.y0 * Hf
        self.w, self.h = bb.width * Wf, bb.height * Hf

    def fx(self, x):
        return (self.l + x) / self.Wf

    def fy(self, y):
        return (self.b + y) / self.Hf

    def text(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("va", "center")
        kw.setdefault("fontsize", FS_TEXT)
        return self.fig.text(self.fx(x), self.fy(y), s, **kw)

    def ax(self, x, y, w, h):
        return self.fig.add_axes([self.fx(x), self.fy(y), w / self.Wf, h / self.Hf])

    def line(self, xs, ys, **kw):
        a = Line2D([self.fx(v) for v in xs], [self.fy(v) for v in ys], transform=self.fig.transFigure, **kw)
        self.fig.add_artist(a)
        return a

    def dot(self, x, y, color, ec=None, ms=2.6):
        a = Line2D([self.fx(x)], [self.fy(y)], marker="o", ms=ms, mfc=color, mec=ec or color, mew=0.35,
                   ls="none", transform=self.fig.transFigure)
        self.fig.add_artist(a)
        return a

    def width_mm(self, t):
        r = self.fig.canvas.get_renderer()
        return t.get_window_extent(r).width / self.fig.dpi / MM


def _base(ax, ext):
    x0, y0, x1, y1 = ext
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():          # the site-extent outline is the map boundary
        sp.set_visible(False)
    ax.patch.set_alpha(0)


def _site(ax, D, lw=0.6):
    hx, hy = D["hull"].exterior.xy
    ax.plot(hx, hy, color=SITE_EDGE, lw=lw, solid_joinstyle="round", zorder=7)


def _clipped(ax, D, seg, **kw):
    """Draw a polyline clipped to the site extent."""
    g = LineString(seg).intersection(D["hull"])
    parts = [] if g.is_empty else (list(g.geoms) if hasattr(g, "geoms") else [g])
    for q in parts:
        if q.geom_type != "LineString" or q.length < 1.0:
            continue
        c = np.asarray(q.coords)
        ax.plot(c[:, 0], c[:, 1], **kw)
        D["drawn"].append(c)


def _dots(ax, xy, idx, c, z=2, s=0.22):
    ax.scatter(xy[idx, 0], xy[idx, 1], s=s, c=c, linewidths=0, rasterized=True, zorder=z)


def _halo(lw):
    return [pe.Stroke(linewidth=lw + 0.7, foreground="white"), pe.Normal()]


def _outlines(ax, D, lw=0.4):
    """2013 favela outlines clipped to the site extent (boundary lines only)."""
    for g in D["Pc"].boundary:
        c = g.intersection(D["hull"])
        parts = [] if c.is_empty else (list(c.geoms) if hasattr(c, "geoms") else [c])
        for q in parts:
            if q.geom_type != "LineString" or q.length < 0.5:
                continue
            v = np.asarray(q.coords)
            ax.plot(v[:, 0], v[:, 1], color="#111111", lw=lw, solid_joinstyle="round", solid_capstyle="butt",
                    path_effects=_halo(lw), zorder=5)
            D["drawn_outline"].append(v)


def _grid(ax, D, L, c, lw):
    xy = D["xy"]
    x0, y0, x1, y1 = D["ext"]
    ox, oy = xy[:, 0].min(), xy[:, 1].min()          # first grid placement: origin at the site's (min x, min y)
    e = max(90.0, 0.12 * L)
    for gx in np.arange(ox, x1 + 1, L):
        if x0 + e < gx < x1 - e:
            _clipped(ax, D, [(gx, y0), (gx, y1)], color=c, lw=lw, zorder=4)
    for gy in np.arange(oy, y1 + 1, L):
        if y0 + e < gy < y1 - e:
            _clipped(ax, D, [(x0, gy), (x1, gy)], color=c, lw=lw, zorder=4)


def _band(ax, D, te, lw=0.55):
    """Dashed line at the buffer distance (BUF) from the test buildings."""
    xy = D["xy"]
    x0, y0, x1, y1 = D["ext"]
    gx = np.arange(x0 - 300, x1 + 300, 8.0)
    gy = np.arange(y0 - 300, y1 + 300, 8.0)
    GX, GY = np.meshgrid(gx, gy)
    dist, _ = cKDTree(xy[te]).query(np.c_[GX.ravel(), GY.ravel()], k=1)
    for seg in contourpy.contour_generator(GX, GY, dist.reshape(GX.shape)).lines(BUF):
        if len(seg) >= 2:
            _clipped(ax, D, seg, color=INK, lw=lw, ls=(0, (2.2, 1.4)), zorder=6)


def geometry(width_mm, height_mm=None, map_mm=None, ar=1.0):
    """Map height and width (mm) for a region width."""
    s_w = (width_mm - 2 * LEFT - 2 * G_ARROW - G_SMALL) / 4
    s = map_mm or min(s_w / ar, MAP_CAP)
    if height_mm is not None:
        s = min(s, height_mm - MAP_TOP - UNDER - LEG)
    return s, s * ar


def content_height_mm(D, width_mm=183.0):
    x0, y0, x1, y1 = D["ext"]
    s, _ = geometry(width_mm, ar=(x1 - x0) / (y1 - y0))
    return MAP_TOP + s + UNDER + LEG


def draw_schematic(fig, region, D, letter="a"):
    M = _MM(fig, region)
    with plt.rc_context(RC):
        return _draw(fig, M, D, letter)


def _draw(fig, M, D, letter):
    xy, ext = D["xy"], D["ext"]
    D["drawn"], D["drawn_outline"] = [], []
    x0, y0, x1, y1 = ext
    ar = (x1 - x0) / (y1 - y0)
    s, sw = geometry(M.w, M.h, None, ar)
    # the leftover width is shared between the three gaps in their design ratio
    slack = M.w - 2 * LEFT - 4 * sw
    g_arrow, g_small = G_ARROW, G_SMALL
    if slack > 2 * G_ARROW + G_SMALL:
        f_ = slack / (2 * G_ARROW + G_SMALL)
        g_arrow, g_small = G_ARROW * f_, G_SMALL * f_
    content_w = 4 * sw + 2 * g_arrow + g_small
    content_h = MAP_TOP + s + UNDER + LEG
    xo = (M.w - content_w) / 2
    yt = M.h - (M.h - content_h) / 2
    X = [xo, xo + sw + g_arrow, xo + 2 * sw + 2 * g_arrow, xo + 3 * sw + 2 * g_arrow + g_small]
    MB = yt - MAP_TOP - s
    L1 = MB - 2.2
    LY = yt - content_h + 2.0
    TY = yt - TITLE_Y
    m_per_mm = (x1 - x0) / sw

    def title(i, roman, txt, dx=0.0):
        r = M.text(X[i] + dx, TY, roman, fontsize=FS_TITLE, color=MUTED, fontstyle="italic")
        off = dx + M.width_mm(r) + 1.2
        t = M.text(X[i] + off, TY, txt, fontsize=FS_TITLE)
        room = sw + (g_small if i == 2 else g_arrow if i < 2 else 0) - 0.8
        check(off + M.width_mm(t) <= room, f"title too wide: {txt}")

    # ---- panel header: letter and site name
    HY = yt - HEAD_Y
    hx = X[0]
    if letter:
        M.text(X[0], HY, letter, fontsize=FS_LETTER, fontweight="bold")
        hx = X[0] + 4.2
    M.text(hx, HY, D["site_name"], fontsize=FS_TITLE)

    # ---- i: 250 m blocks coloured by fold, scale bar inside
    ax = M.ax(X[0], MB, sw, s)
    _base(ax, ext)
    for f in range(len(FOLDC)):
        _dots(ax, xy, np.where(D["fold_of"] == f)[0], FOLDC[f])
    _grid(ax, D, 250, BLUE250, 0.35)
    _outlines(ax, D)
    _site(ax, D)
    sbx, sby = x0 + 2.0 * m_per_mm, y0 + 2.2 * m_per_mm
    ax.plot([sbx, sbx + 500], [sby, sby], color=INK, lw=1.1, solid_capstyle="butt", zorder=9,
            path_effects=[pe.Stroke(linewidth=2.3, foreground="white"), pe.Normal()])
    ax.text(sbx + 250, sby + 1.0 * m_per_mm, "500 m", fontsize=FS_SMALL, ha="center", va="bottom", color=INK,
            zorder=9, path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    title(0, "i", "250 m blocks → 5 folds")
    t = M.text(X[0], L1, "fold", fontsize=FS_SMALL, color=MUTED)
    xx = X[0] + M.width_mm(t) + 1.1
    for k in range(len(FOLDC)):
        M.dot(xx + 0.9, L1, FOLDC[k], ms=2.5)
        tt = M.text(xx + 2.0, L1, str(k + 1), fontsize=FS_SMALL, color=MUTED)
        xx += 2.0 + M.width_mm(tt) + 1.7

    # ---- ii: the fold as test, no buffer
    S0 = D["s250_0"]
    ax = M.ax(X[1], MB, sw, s)
    _base(ax, ext)
    _dots(ax, xy, S0["tr"], TRAIN)
    _dots(ax, xy, S0["te"], ACC, z=3)
    _grid(ax, D, 250, BLUE250, 0.3)
    _outlines(ax, D)
    _site(ax, D)
    E = int(D["counts"].set_index("panel").loc["ii", "fold_index"]) + 1       # fold shown, 1-based
    others = [k for k in range(1, len(FOLDC) + 1) if k != E]
    contiguous = others == list(range(others[0], others[-1] + 1))
    title(1, "ii", f"Fold {E} = test, folds {others[0]}–{others[-1]} train" if contiguous
          else f"Fold {E} = test, other folds train")

    # ---- iii: same fold, 200 m buffer
    S1 = D["s250"]
    ax = M.ax(X[2], MB, sw, s)
    _base(ax, ext)
    _dots(ax, xy, S1["removed"], REMOVED)
    _dots(ax, xy, S1["kept"], TRAIN)
    _dots(ax, xy, S1["te"], ACC, z=3)
    _band(ax, D, S1["te"])
    _outlines(ax, D)
    _site(ax, D)
    title(2, "iii", f"+ {BUF:g} m buffer")
    pct = 100 * S1["nd"] / len(S1["tr"])

    # ---- arrows i -> ii ("fold") and ii -> iii ("buffer")
    ay = MB + s * 0.5
    for i, lab in ((0, f"fold {E}"), (1, "buffer")):
        arr = FancyArrowPatch((M.fx(X[i] + sw + 1.4), M.fy(ay)), (M.fx(X[i + 1] - 1.4), M.fy(ay)),
                              transform=fig.transFigure, arrowstyle="-|>,head_length=1.6,head_width=0.9",
                              mutation_scale=1, color=INK, lw=0.6)
        fig.add_artist(arr)
        tl = M.text(X[i] + sw + g_arrow / 2, ay + 2.4, lab, fontsize=FS_SMALL, ha="center", color=MUTED)
        check(M.width_mm(tl) < g_arrow - 0.4, f"arrow label too wide: {lab}")

    # ---- iv: other held-out units, two stacked mini-maps
    title(3, "iv", "Other held-out units")
    gap = 1.6
    sm = (s - gap) / 2
    minis = [(MB + sm + gap, D["s1k"], "1,000 m\nblocks", 1000), (MB, D["sc"], "Favela\ncomponent", None)]
    for bottom, S2, name, L in minis:
        ax = M.ax(X[3], bottom, sm * ar, sm)
        _base(ax, ext)
        _dots(ax, xy, S2["removed"], REMOVED, s=0.1)
        _dots(ax, xy, S2["kept"], TRAIN, s=0.1)
        _dots(ax, xy, S2["te"], ACC, z=3, s=0.1)
        if L:
            _grid(ax, D, L, BLUE1000, 0.55)
        _band(ax, D, S2["te"], lw=0.45)
        _outlines(ax, D, lw=0.25)
        _site(ax, D, lw=0.45)
        tx = X[3] + sm * ar + 1.4
        tn = M.text(tx, bottom + sm - 0.4, name, fontsize=FS_TEXT, va="top", linespacing=1.15)
        check(tx + M.width_mm(tn) <= X[3] + sw + LEFT, f"mini-map label too wide: {name}")
    sb = 500 / (x1 - x0) * sm * ar
    M.line([X[3], X[3] + sb], [L1, L1], color=INK, lw=1.0, solid_capstyle="butt")
    M.text(X[3] + sb + 0.9, L1, "500 m", fontsize=FS_SMALL, color=MUTED)

    # ---- single shared legend strip (centred)
    items = [("dot", "test", dict(c=ACC)), ("dot", "training kept", dict(c=TRAIN)),
             ("dot", "removed by buffer", dict(c=REMOVED, ec=REMOVED_EDGE)),
             ("outline", "2013 favela outline", {}), ("band", f"{BUF:g} m from test buildings", {}),
             ("grid", "block grid", {}), ("site", "site extent", {})]
    widths = []
    for kind, label, _ in items:
        t = M.text(0, LY, label, fontsize=FS_TEXT)
        w = M.width_mm(t)
        t.remove()
        widths.append(({"dot": 1.3, "outline": 4.0, "band": 4.0, "grid": 4.0, "site": 3.2}[kind], w))
    GAP_ITEM = 3.6
    total = sum(k + 0.8 + w for k, w in widths) + GAP_ITEM * (len(widths) - 1)
    check(total <= content_w, "legend strip too wide")
    lx = xo + (content_w - total) / 2
    for (kind, label, kw), (key, w) in zip(items, widths):
        if kind == "dot":
            M.dot(lx + key / 2, LY, kw["c"], kw.get("ec"))
        elif kind == "outline":
            M.line([lx, lx + key], [LY, LY], color="#111111", lw=0.5, path_effects=_halo(0.5))
        elif kind == "band":
            M.line([lx, lx + key], [LY, LY], color=INK, lw=0.55, ls=(0, (2.2, 1.4)))
        elif kind == "grid":
            M.line([lx, lx + key / 2 - 0.3], [LY, LY], color=BLUE250, lw=0.6)
            M.line([lx + key / 2 + 0.3, lx + key], [LY, LY], color=BLUE1000, lw=0.6)
        elif kind == "site":
            M.line([lx, lx + key, lx + key, lx, lx], [LY - 1.0, LY - 1.0, LY + 1.0, LY + 1.0, LY - 1.0],
                   color=SITE_EDGE, lw=0.6, solid_joinstyle="round")
        M.text(lx + key + 0.8, LY, label, fontsize=FS_TEXT)
        lx += key + 0.8 + w + GAP_ITEM

    # ---- every drawn grid, band and outline vertex lies inside the site extent
    tol = D["hull"].buffer(0.5)
    for c in D["drawn"] + D["drawn_outline"]:
        assert shapely.contains_xy(tol, c[:, 0], c[:, 1]).all(), "line drawn outside the site extent"
    return dict(map_mm=s, x_mm=[M.l + v for v in X], removed_pct=pct)
