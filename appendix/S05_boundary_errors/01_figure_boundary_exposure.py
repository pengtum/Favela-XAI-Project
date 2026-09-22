"""Step 1: Extended Data Fig. 1, misclassification rate by signed distance to the 2013 favela boundary.

Manuscript items
* Extended Data Fig. 1 (cited in SI S5 and Fig. 2):
  a, misclassified buildings as a percentage of all buildings in each 6 m bin,
     end bins beyond 90 m; pooled rate with Wilson 95% interval and
     site-bootstrap 95% interval (shaded);
  b, all buildings and misclassified buildings per bin;
  c, rate by distance band and side with Wilson and site-bootstrap intervals,
     building and error counts and the near/far ratio.

Inputs (main/02_classification step 04, from the stratified five-fold
out-of-fold predictions of step 03):
  boundary_error_bins.csv   pooled 6 m bins with Wilson and site-bootstrap intervals
  boundary_error_bands.csv  distance bands by side with the near/far ratio
Output:  outputs/appendix/S05_boundary_errors/Figure_R1_boundary_exposure.pdf (+ PNG preview).

Run from the repository root:
    python appendix/S05_boundary_errors/01_figure_boundary_exposure.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory as blend  # noqa: E402

from favela_xai.config import load_config  # noqa: E402

import s05_common as C  # noqa: E402

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "pdf.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6, "axes.spines.right": False,
                            "axes.spines.top": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                            "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelsize": 6,
                            "ytick.labelsize": 6, "axes.labelsize": 6.5, "legend.frameon": False})
# Colours of the figure: favela-labelled side blue, formal side green
FAV, NFV, INK, MUT = "#3F7FB0", "#5AA469", "#1F2933", "#5A6875"
FAV_D, NFV_D = "#1F4E75", "#2F6B3B"          # darker shades (misclassified part of bars, labels)
FAV_L, NFV_L = "#C9DCEB", "#CFE6D3"          # lighter shades (bootstrap band, exposure bars)
COL = {"inside": FAV, "outside": NFV, "both": INK}
LIGHT = {"inside": FAV_L, "outside": NFV_L, "both": "#D5DBE0"}
DARK = {"inside": FAV_D, "outside": NFV_D, "both": INK}
MM = 1 / 25.4
MINUS, GE, TIMES, NDASH = "−", "≥", "×", "–"


def xpos(df, tailx):
    """x position of each bin: its centre, the two end bins at +-tailx."""
    x = df.bin_centre_m.to_numpy().copy()
    x[np.isneginf(df.lower_m.to_numpy())] = -tailx
    x[np.isposinf(df.upper_m.to_numpy())] = tailx
    return x


def break_marks(ax, xs, dx=1.6, hpts=2.2, slant=1.2):
    """Axis-break marks between the last regular bin and the end bin."""
    h = hpts / (ax.get_position().height * ax.figure.get_figheight() * 72)
    tr = blend(ax.transData, ax.transAxes)
    for x in xs:
        ax.add_patch(Rectangle((x - dx, -h), 2 * dx, 2 * h, transform=tr, fc="white", ec="none", clip_on=False, zorder=5))
        for o in (-dx, dx):
            ax.plot([x + o - slant, x + o + slant], [-1.8 * h, 1.8 * h], transform=tr, color=INK, lw=0.6,
                    clip_on=False, zorder=6)


def set_xaxis(ax, tailx, breakx, xl, ticks, tail_lab, labels=True, half_range=90):
    ax.set_xlim(-xl, xl)
    ax.set_xticks([-tailx] + list(ticks) + [tailx])
    if labels:
        ax.set_xticklabels([tail_lab[0]] + [(f"+{v}" if v > 0 else f"{MINUS}{-v}") if v else "0" for v in ticks]
                           + [tail_lab[1]])
    else:
        ax.set_xticklabels([])
    ax.set_xticks([-half_range, half_range], minor=True)
    ax.tick_params(axis="x", which="minor", length=1.6, width=0.5)
    break_marks(ax, (-breakx, breakx))


def pctfmt(v, d=1):
    return f"{100 * v:.{d}f}"


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    half = int(cfg.get("boundary_errors.half_range_m"))
    BP = pd.read_csv(C.main02_table(cfg, "bins"))
    BS = pd.read_csv(C.main02_table(cfg, "bands"))
    BANDS = C.band_labels(cfg)
    near, far = BANDS[0], BANDS[-1]

    TAILX, BREAKX, XL = 102.0, 94.5, 109
    BP["x"] = xpos(BP, TAILX)
    fig = plt.figure(figsize=(183 * MM, 115 * MM))
    Lm, Bm, Tm = 0.085, 0.095, 0.955
    gl = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.10, left=Lm, right=0.585, bottom=Bm, top=Tm)
    axa = fig.add_subplot(gl[0])
    axb = fig.add_subplot(gl[1], sharex=axa)

    # ---- a: conditional error rate
    for side in ("inside", "outside"):
        d = BP[BP.side == side]
        core, tail = d[np.isfinite(d.bin_centre_m)], d[~np.isfinite(d.bin_centre_m)]
        axa.fill_between(core.x, 100 * core.boot_lo, 100 * core.boot_hi, color=LIGHT[side], lw=0, zorder=1)
        for _, r in tail.iterrows():
            axa.add_patch(Rectangle((r.x - 2.6, 100 * r.boot_lo), 5.2, 100 * (r.boot_hi - r.boot_lo), fc=LIGHT[side],
                                    ec="none", zorder=1))
        axa.plot(core.x, 100 * core.error_rate, color=COL[side], lw=0.7, zorder=2)
        axa.vlines(d.x, 100 * d.wilson_lo, 100 * d.wilson_hi, color=DARK[side], lw=0.7, zorder=3)
        axa.scatter(d.x, 100 * d.error_rate, s=9, color=COL[side], ec="white", lw=0.4, zorder=4)
    YA = max(58, float(np.ceil(100 * BP.boot_hi.max() / 2) * 2 + 2))       # y-axis top (%), at least 58
    axa.set_ylim(0, YA)
    axa.set_yticks([v for v in range(0, int(YA), 10)])
    axa.axvline(0, color=INK, ls=(0, (3, 2)), lw=0.7, zorder=2)
    axa.text(0, YA * 0.985, "2013 favela boundary", ha="center", va="top", fontsize=6, color=INK,
             bbox=dict(fc="white", ec="none", pad=0.6), zorder=7)
    axa.text(-86, 33.5, "inside (favela-labelled):\nfalse negatives", color=FAV_D, fontsize=6.5, ha="left",
             va="center", fontweight="bold", linespacing=1.15)
    axa.text(TAILX + 2, 33.5, "outside (formal):\nfalse positives", color=NFV_D, fontsize=6.5, ha="right",
             va="center", fontweight="bold", linespacing=1.15)
    nb = BS[BS.side == "both"].set_index("band")
    axa.text(-TAILX - 4, 50.5, f"both sides pooled\nwithin 6 m: {pctfmt(nb.loc[near, 'error_rate'])}%  vs  "
             f"beyond 90 m: {pctfmt(nb.loc[far, 'error_rate'])}%", fontsize=6.2, color=INK, ha="left",
             va="center", linespacing=1.35)
    axa.set_ylabel("buildings misclassified (%)")
    kx, ky = 58, 23                                   # key for the two intervals
    axa.add_patch(Rectangle((kx, ky - 1.2), 4, 2.4, fc="#DADFE3", ec="none"))
    axa.text(kx + 6, ky, "site-bootstrap 95% CI", fontsize=5.6, color=MUT, va="center")
    axa.vlines(kx + 2, ky - 5.3, ky - 2.9, color=INK, lw=0.7)
    axa.scatter([kx + 2], [ky - 4.1], s=7, color=MUT, ec="white", lw=0.4, zorder=4)
    axa.text(kx + 6, ky - 4.1, "rate, Wilson 95% CI", fontsize=5.6, color=MUT, va="center")
    set_xaxis(axa, TAILX, BREAKX, XL, (-60, -30, 0, 30, 60), ("", ""), labels=False, half_range=half)
    plt.setp(axa.get_xticklabels(), visible=False)
    axa.text(-0.135, 1.0, "a", transform=axa.transAxes, fontsize=8, fontweight="bold", va="top")

    # ---- b: exposure (all buildings and misclassified buildings per bin)
    for side in ("inside", "outside"):
        d = BP[(BP.side == side) & np.isfinite(BP.bin_centre_m)]
        axb.bar(d.x, d.n_buildings, width=5.0, color=LIGHT[side], lw=0, zorder=2)
        axb.bar(d.x, d.n_errors, width=5.0, color=DARK[side], lw=0, zorder=3)
    ymax_b = max(11000, float(np.ceil(BP[np.isfinite(BP.bin_centre_m)].n_buildings.max() / 1000) * 1000 + 1000))
    axb.set_ylim(0, ymax_b)
    ticks_b = list(range(0, int(ymax_b), 2500))
    axb.set_yticks(ticks_b)
    axb.set_yticklabels([f"{v:,}" for v in ticks_b])
    axb.axvline(0, color=INK, ls=(0, (3, 2)), lw=0.7, zorder=4)
    for side in ("inside", "outside"):
        r = BP[(BP.side == side) & ~np.isfinite(BP.bin_centre_m)].iloc[0]
        tx, ha = (-XL + 1.5, "left") if side == "inside" else (XL, "right")
        axb.text(tx, 5200, f"{int(r.n_buildings):,}\nbuildings", ha=ha, va="bottom", fontsize=5.6, color=DARK[side],
                 linespacing=1.1, multialignment="center")
        axb.text(tx, 4800, f"{int(r.n_errors):,}\nerrors", ha=ha, va="top", fontsize=5.6, color=DARK[side],
                 linespacing=1.1, multialignment="center")
    axb.annotate("all test buildings in bin", xy=(57, 2150), xytext=(57, 6200), ha="center", va="bottom",
                 fontsize=5.8, color=MUT, arrowprops=dict(arrowstyle="-", color=MUT, lw=0.5, shrinkA=1, shrinkB=0))
    axb.annotate("misclassified", xy=(33, 330), xytext=(33, 3700), ha="center", va="bottom", fontsize=5.8, color=MUT,
                 arrowprops=dict(arrowstyle="-", color=MUT, lw=0.5, shrinkA=1, shrinkB=0))
    axb.set_ylabel("buildings per 6 m bin")
    axb.set_xlabel("signed distance to the 2013 favela boundary (m)")
    set_xaxis(axb, TAILX, BREAKX, XL, (-60, -30, 0, 30, 60), (f"< {MINUS}{half} m", f"{GE} {half} m"), half_range=half)
    axb.text(-0.135, 1.02, "b", transform=axb.transAxes, fontsize=8, fontweight="bold", va="top")
    fig.align_ylabels([axa, axb])

    # ---- c: bands by side
    axc = fig.add_axes([0.705, Bm, 0.175, Tm - Bm])
    BLAB = {BANDS[0]: "< 6 m", BANDS[1]: f"6{NDASH}30 m", BANDS[2]: f"30{NDASH}90 m", BANDS[3]: f"{GE} 90 m"}
    GROUPS = [("inside", "inside: false negatives"), ("outside", "outside: false positives"), ("both", "both sides")]
    y, ypos, heads = 0, [], []
    for side, head in GROUPS:
        heads.append((y, side, head))
        y += 1.05
        for b in BANDS:
            ypos.append((y, side, b))
            y += 1
        y += 0.55
    ytop = y
    XH, C1, C2 = 0.628, 0.943, 0.993
    trc = blend(fig.transFigure, axc.transData)
    for yy, side, b in ypos:
        r = BS[(BS.side == side) & (BS.band == b)].iloc[0]
        axc.plot([100 * r.boot_lo, 100 * r.boot_hi], [yy, yy], color=LIGHT[side], lw=4.2, solid_capstyle="butt", zorder=1)
        axc.plot([100 * r.wilson_lo, 100 * r.wilson_hi], [yy, yy], color=INK, lw=0.8, zorder=2)
        axc.scatter([100 * r.error_rate], [yy], s=14, color=COL[side], ec="white", lw=0.5, zorder=3)
        axc.text(C1, yy, f"{int(r.n_buildings):,}", transform=trc, ha="right", va="center", fontsize=5.8, color=INK)
        axc.text(C2, yy, f"{int(r.n_errors):,}", transform=trc, ha="right", va="center", fontsize=5.8, color=INK)
    for yy, side, head in heads:
        rr = BS[BS.side == side].iloc[0]
        axc.text(XH, yy, head, transform=trc, ha="left", va="center", fontsize=6.3, fontweight="bold", color=DARK[side])
        axc.text(C2, yy, f"< 6 m vs {GE} 90 m: {rr.ratio_near_far:.1f}{TIMES}", transform=trc, ha="right",
                 va="center", fontsize=5.8, color=MUT)
    axc.text(C1, -1.05, "buildings", transform=trc, ha="right", va="center", fontsize=5.8, color=MUT)
    axc.text(C2, -1.05, "errors", transform=trc, ha="right", va="center", fontsize=5.8, color=MUT)
    axc.set_yticks([p[0] for p in ypos])
    axc.set_yticklabels([BLAB[p[2]] for p in ypos])
    axc.tick_params(axis="y", length=0)
    axc.spines["left"].set_visible(False)
    axc.set_ylim(ytop - 0.2, -1.6)
    xmax_c = max(52, float(np.ceil(100 * BS.boot_hi.max() / 2) * 2 + 2))      # x-axis end (%), at least 52
    axc.set_xlim(0, xmax_c)
    axc.set_xticks(list(range(0, int(xmax_c), 10)))
    axc.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    axc.set_axisbelow(True)
    axc.set_xlabel("buildings misclassified (%)")
    axc.text(XH - 0.015, Tm, "c", transform=fig.transFigure, fontsize=8, fontweight="bold", va="top")

    fig.savefig(out / C.FIGURE_EXPOSURE)
    fig.savefig((out / C.FIGURE_EXPOSURE).with_suffix(".png"), dpi=190)
    plt.close(fig)
    print(f"written {out / C.FIGURE_EXPOSURE}")


if __name__ == "__main__":
    main()
