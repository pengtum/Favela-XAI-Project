"""Helpers shared by the Fig. 1 element scripts of this folder (not a step of its own).

Used by 07_fig1_basemap.py, 08_figure1_study_area.py and
09_figure1_example_maps.py. Fig. 1 is assembled by hand from the element
maps written by these scripts (panel b, the pipeline schematic, is artwork);
see README.md.

Contents
* ``settings``: the ``main_01_features.figure1`` section of config.yaml.
* ``municipality_outline``: the municipal boundary of Rio de Janeiro
  (paths.raw.municipality, Data.Rio) in Web Mercator (EPSG:3857), the CRS of
  the basemap tiles.
* ``map_extent``: the extent of the study-area map (outline extent widened by
  the configured margins) and of the tile download.
* ``basemap_files`` / ``load_basemap``: the cached basemap tiles written by
  07_fig1_basemap.py.
* ``apply_style`` / ``save_element``: fonts of the element maps and export as
  PDF and SVG, with all text converted to vector outlines when
  ``figure1.text_as_outlines`` is true (so that the maps look identical in
  every program).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from favela_xai.io import read_vector
from favela_xai.plotting_style import use_style

WEB_MERCATOR = 3857


def settings(cfg) -> dict:
    """Section main_01_features.figure1 of config.yaml."""
    return cfg.get("main_01_features.figure1")


def municipality_outline(cfg):
    """Municipal boundary of Rio de Janeiro (Data.Rio), EPSG:3857."""
    return read_vector(cfg.raw_path("municipality"), cfg).to_crs(WEB_MERCATOR)


def map_extent(outline, f1: dict) -> dict:
    """Map extent (x and y ranges, EPSG:3857) and the padded extent of the tile download.

    The map shows the whole municipality widened by ``map_margins`` (shares of the
    outline width / height: left, right, bottom, top); the tiles cover the map
    extent widened by ``basemap_pad`` on every side.
    """
    xmin, ymin, xmax, ymax = outline.total_bounds
    w, h = xmax - xmin, ymax - ymin
    left, right, bottom, top = f1["map_margins"]
    map_x = (xmin - left * w, xmax + right * w)
    map_y = (ymin - bottom * h, ymax + top * h)
    pad = float(f1["basemap_pad"])
    pw, ph = map_x[1] - map_x[0], map_y[1] - map_y[0]
    download = (map_x[0] - pad * pw, map_y[0] - pad * ph, map_x[1] + pad * pw, map_y[1] + pad * ph)
    return dict(map_x=map_x, map_y=map_y, aspect=pw / ph, download=download)


def basemap_files(cfg) -> tuple[Path, Path]:
    """Cached tile image (.npy) and its metadata (.json)."""
    folder = cfg.resolve(cfg.get("paths.raw.basemap_cache_dir"))
    return folder / "fig1_basemap.npy", folder / "fig1_basemap.json"


def load_basemap(cfg):
    """(image, extent [xmin, xmax, ymin, ymax] in EPSG:3857, metadata) or None if not downloaded."""
    img_path, meta_path = basemap_files(cfg)
    if not (img_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return np.load(img_path), tuple(meta["image_extent"]), meta


def apply_style(f1: dict) -> None:
    """Shared style with the font size of the element maps; SVG text as paths when outlined."""
    use_style(**{"font.size": float(f1["font_size_pt"]),
                 "svg.fonttype": "path" if f1["text_as_outlines"] else "none"})


def save_element(fig, out_dir: Path, name: str, f1: dict) -> list[Path]:
    """Save an element map as PDF and SVG (tight bounding box, no title).

    With ``text_as_outlines`` every text object gets a no-op path effect, which
    makes matplotlib draw the glyphs as vector paths (no font in the file).
    """
    import matplotlib.patheffects as pe
    import matplotlib.text as mtext

    if f1["text_as_outlines"]:
        for t in fig.findobj(mtext.Text):
            t.set_path_effects([pe.Normal()])
    paths = []
    for ext in ("pdf", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths
