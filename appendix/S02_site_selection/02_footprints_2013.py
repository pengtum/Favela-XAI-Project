"""Step 2: centroid, area and footprint type of every 2013 building footprint of the city.

Manuscript items
* Supplementary Fig. S1e and SI S2.1: footprints per hectare of each favela
  polygon, counting all footprint types of the 2013 layer (density computed in
  04_compare_study_vs_city.py).

Inputs
* ``paths.raw.buildings_2013``: Edificacoes 2013 (Data.Rio), the city-wide
  footprint layer with all ``Tipo`` classes.

Outputs (outputs/appendix/S02_site_selection/)
* ``footprints_2013_centroids.parquet``: one row per footprint with fid,
  tipo, area_geom (m2), x, y (centroid, EPSG:31983) and valid (geometry
  validity flag).

Settings
* ``appendix_S02_site_selection.footprint_tipo_field``: attribute holding the footprint type.
* ``appendix_S02_site_selection.footprint_read_chunk``: features read per chunk (memory only).

Run from the repository root:
    python appendix/S02_site_selection/02_footprints_2013.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import CRS

from favela_xai.config import load_config
from favela_xai.io import write_table

import s02_lib as L


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    s = cfg.section("appendix_S02_site_selection")
    path = cfg.raw_path("buildings_2013")
    tipo = s["footprint_tipo_field"]
    chunk = int(s["footprint_read_chunk"])

    info = pyogrio.read_info(path)
    n = int(info["features"])
    crs = CRS.from_user_input(info["crs"]) if info.get("crs") else None
    target = CRS.from_user_input(cfg.get("crs"))
    print(f"{path.name}: {n:,} features, CRS {crs}")

    parts, t0 = [], time.time()
    for start in range(0, n, chunk):
        df = pyogrio.read_dataframe(path, columns=[tipo], skip_features=start, max_features=chunk)
        if crs is not None and not crs.equals(target):
            df = df.to_crs(target)
        geom = shapely.force_2d(np.asarray(df.geometry.values, dtype=object))
        cen = shapely.centroid(geom)
        parts.append(pd.DataFrame({
            "fid": np.arange(start, start + len(df)),
            "tipo": df[tipo].astype(str).to_numpy(),
            "area_geom": shapely.area(geom),
            "x": shapely.get_x(cen),
            "y": shapely.get_y(cen),
            "valid": shapely.is_valid(geom),
        }))
        print(f"  {start + len(df):>9,} / {n:,} features  ({time.time() - t0:.0f} s)")
    fp = pd.concat(parts, ignore_index=True)
    write_table(fp, out / L.FOOTPRINTS_PARQUET)
    print(fp["tipo"].value_counts().to_string())
    print(f"written {out / L.FOOTPRINTS_PARQUET} ({len(fp):,} rows)")


if __name__ == "__main__":
    main()
