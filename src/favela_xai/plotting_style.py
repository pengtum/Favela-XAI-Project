"""Shared matplotlib style for all figure scripts.

One place for fonts, sizes and colours so that every figure of the manuscript
(main, Extended Data and Supplementary) is drawn consistently:
* Arial (Helvetica / DejaVu Sans fallbacks), TrueType fonts embedded in PDF
  (``pdf.fonttype = 42``), editable text in SVG;
* Nature widths: 89 mm (single column) and 183 mm (double column);
* class colours: favela blue / formal-settlement green on maps (Main Figs 1-2,
  SI Fig. S2), favela orange for SHAP evidence (Main Figs 3-5);
* the five morphometric dimensions (frame and bar colours of Main Figs 3-4 and
  ED Fig. 2);
* the diverging blue-white-orange map of Tree-SHAP values (orange = toward favela).

Figure scripts call ``use_style()`` once, then ``save_figure(fig, path)``.
Figure text uses the manuscript terms ("favela", "formal settlement") and the
indicator labels of SI Table S2 (``features.label_of``); site names come from
config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4                      # inches per millimetre
WIDTH_SINGLE_MM = 89               # Nature single column
WIDTH_DOUBLE_MM = 183              # Nature double column

# Ink
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

# Classes on maps (Main Fig. 1a/1c, Main Fig. 2a; SI Fig. S2)
FAVELA_MAP = "#3F83A2"             # favela (blue)
FORMAL_MAP = "#42A969"             # formal settlement (green)
MISCLASSIFIED = "#F39C12"          # misclassified (orange; Main Fig. 2)
MISCLASSIFIED_FORMAL = "#E74C3C"   # second error colour of SI Fig. S2 (formal buildings called favela)
NEUTRAL_BUILDING = "#e8e8e8"       # buildings not shown in a category (e.g. training buildings)
BOUNDARY = "#1a1a1a"               # official 2013 favela boundary

# Classes in attribution plots (Main Figs 3-5)
FAVELA_SHAP = "#E28E2C"            # favela / toward favela (orange)
FORMAL_SHAP = "#6BAED6"            # formal settlement (light blue)

# Morphometric dimensions (SI Table S2; Main Figs 3-4; ED Fig. 2)
DIMENSION_COLOURS = {
    "Dimension": "#0F4D92",
    "Shape": "#E2700E",
    "Spatial distribution": "#2A9D8F",
    "Configuration": "#9A4D8E",
    "Connectivity": "#4C9F70",
}

# Diverging map of SHAP values: blue (toward formal) - white - orange (toward favela)
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_blue_orange", ["#0F4D92", "#F4F4F4", "#E28E2C"])

# Sequential blues for ordered design parameters (block size, buffer; SI S12 figures)
BLUES = {100: "#86b6ef", 250: "#3987e5", 500: "#1c5cab", 1000: "#0d366b"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
    "savefig.dpi": 600,
}


def use_style(**overrides) -> None:
    """Apply the shared style (optionally overriding single rcParams)."""
    plt.rcParams.update(RC)
    if overrides:
        plt.rcParams.update(overrides)


def figure_size(width_mm: float = WIDTH_DOUBLE_MM, height_mm: float = 100) -> tuple[float, float]:
    """Figure size in inches from millimetres."""
    return width_mm * MM, height_mm * MM


def dimension_colour(feature_key: str) -> str:
    """Colour of the morphometric dimension of an indicator."""
    from .features import dimension_of

    return DIMENSION_COLOURS[dimension_of(feature_key)]


def clean_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(fig, path: str | Path, preview_png: bool = False, max_preview_px: int = 2000) -> Path:
    """Save a vector PDF (and optionally a PNG preview no wider than max_preview_px)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    if preview_png:
        dpi = min(300, int(max_preview_px / fig.get_figwidth()))
        fig.savefig(path.with_suffix(".png"), dpi=dpi)
    return path
