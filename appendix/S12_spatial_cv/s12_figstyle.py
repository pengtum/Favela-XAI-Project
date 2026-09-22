"""Figure style of SI Figs S19-S20 (imported by 06 and 08; not a step of its own).

Arial, TrueType fonts in the PDF, 7 pt text, neutral ink, one sequential blue
ramp for block size and buffer, grey dashed matched control, orange prediction
tasks. Figures are 183 mm wide (double column); the PDF is saved at its own
size (no tight bounding box) with a PNG preview at most 2,000 px wide.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MM = 1 / 25.4
RC = {"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7,
      "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
      "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
      "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
      "mathtext.bf": "Arial:bold", "mathtext.default": "regular"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUES = {100: "#86b6ef", 250: "#3987e5", 500: "#1c5cab", 1000: "#0d366b"}   # block size, light -> dark
BUF = {0: "#86b6ef", 200: "#1c5cab"}                                           # buffer, same ramp
CONTROL = "#8c8b87"        # matched random-removal control: grey dashed
NEUTRAL = "#262626"        # random-split reference
TASKC = "#eb6834"          # prediction tasks (SI Fig. S20a)


def use():
    plt.rcParams.update(RC)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def repel(ys, gap):
    """Label y positions (same order) at least ``gap`` apart, staying close to the originals."""
    ys = np.asarray(ys, float)
    o = np.argsort(ys)
    out = ys[o].copy()
    for _ in range(50):
        moved = False
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                d = (gap - (out[i] - out[i - 1])) / 2
                out[i] += d
                out[i - 1] -= d
                moved = True
        if not moved:
            break
    res = np.empty_like(out)
    res[o] = out
    return res


def save(fig, path: Path, max_px: int = 2000):
    """Vector PDF at the figure's own size and a PNG preview no wider than ``max_px``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_name(path.stem + "_preview.png"), dpi=min(300, int(max_px / fig.get_figwidth())))
    plt.close(fig)
