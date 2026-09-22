"""Step 7 (optional, needs internet): basemap tiles for the Fig. 1a study-area map.

Manuscript items
  * Fig. 1a, background of the study-area map. Layout only; no result
    depends on it.

What it does
  Downloads CARTO Positron tiles without labels (contextily, zoom 13) covering
  the municipality of Rio de Janeiro plus the map margins and caches them, so
  that 08_figure1_study_area.py draws the map offline (without the cache it
  draws the municipal outline only). Tiles: (c) OpenStreetMap contributors
  (c) CARTO; the attribution must accompany any published use of the map.

Inputs
  paths.raw.municipality   municipal boundary of Rio de Janeiro (Data.Rio; see data/README.md,
                           or python appendix/S02_site_selection/01_context_layers.py --download)
  config.yaml: main_01_features.figure1 (basemap_provider, basemap_zoom, map_margins, basemap_pad)

Outputs (paths.raw.basemap_cache_dir, default data/raw/context/basemap_cache/)
  fig1_basemap.npy    RGBA tile image
  fig1_basemap.json   image extent and map extent (EPSG:3857), provider, zoom, attribution,
                      download date

Run from the repository root:
  python main/01_features/07_fig1_basemap.py
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from favela_xai.config import load_config

import figure1_common as F1


def tile_provider(name: str):
    """xyzservices provider from a dotted name such as 'CartoDB.PositronNoLabels'."""
    import contextily as cx

    provider = cx.providers
    for part in name.split("."):
        provider = provider[part]
    return provider


def main() -> None:
    import contextily as cx

    cfg = load_config()
    f1 = F1.settings(cfg)
    outline = F1.municipality_outline(cfg)
    ext = F1.map_extent(outline, f1)

    xmin, ymin, xmax, ymax = ext["download"]
    img, image_extent = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=int(f1["basemap_zoom"]),
                                      source=tile_provider(f1["basemap_provider"]))
    img_path, meta_path = F1.basemap_files(cfg)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(img_path, img)
    meta = dict(image_extent=[float(v) for v in image_extent],      # xmin, xmax, ymin, ymax (EPSG:3857)
                map_x=list(ext["map_x"]), map_y=list(ext["map_y"]), crs="EPSG:3857",
                provider=f1["basemap_provider"], zoom=int(f1["basemap_zoom"]),
                attribution=f1["basemap_attribution"], downloaded=dt.date.today().isoformat())
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"basemap {img.shape[1]} x {img.shape[0]} px, zoom {meta['zoom']}, {meta['provider']}; "
          f"map aspect {ext['aspect']:.3f}")
    print(f"written {img_path} and {meta_path}")


if __name__ == "__main__":
    main()
