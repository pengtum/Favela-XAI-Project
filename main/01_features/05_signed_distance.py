"""Step 5: signed distance of every building to the official 2013 favela boundary.

Manuscript items
  * Methods 4.7 and 4.8: Euclidean distance from each building centroid to the
    nearest edge of the official 2013 favela polygon, in metres (EPSG:31983),
    negative for favela buildings and positive for non-favela buildings.
  * Used by the misclassification rates (Fig. 2c, Extended Data Fig. 1;
    main/02_classification), the boundary profiles (Fig. 4;
    main/04_boundary_profiles) and the boundary analyses of SI Sections S5,
    S6 and S12-S14.

Method (favela_xai.boundary.signed_distance)
  Per site, the 2013 polygons that intersect the extent of the site's buildings
  padded by signed_distance.polygon_search_pad_m are used; the distance is
  measured from each centroid to their boundary and made negative for
  favela-labelled buildings (signed_distance.sign_by).
  signed_distance.dissolve_polygons: false = edges of the individual polygons;
  true = boundary of their dissolved union (favela/non-favela interface).

Inputs
  outputs/main/01_features/<site>_buildings_features.gpkg   (03_indicators.py)
  paths.raw.favelas_2013                                    Limite Favelas 2013 (Data.Rio)

Outputs (outputs/main/01_features/)
  signed_distance.parquet   products.signed_distance: site, uID, is_favela, signed_dist (m)

Run from the repository root (after 03_indicators.py for all sites):
  python main/01_features/05_signed_distance.py
"""
from __future__ import annotations

import numpy as np

from favela_xai.boundary import load_signed_distance
from favela_xai.config import load_config


def main() -> None:
    cfg = load_config()
    s = cfg.section("signed_distance")
    print(f"signed distance: sign by {s['sign_by']}, polygons within {s['polygon_search_pad_m']} m of the "
          f"site extent, dissolve_polygons = {s['dissolve_polygons']}")

    # Computes the distance for every site and writes products.signed_distance.
    sd = load_signed_distance(cfg, cfg.sites, recompute=True)

    for site, t in sd.groupby("site", sort=False):
        d = t["signed_dist"].to_numpy()
        fav = t["is_favela"].to_numpy(bool)
        print(f"{cfg.site_name(site):24s} buildings {len(t):6,d}  favela side {fav.sum():6,d}  "
              f"within 6 m {int((np.abs(d) < 6).sum()):5,d}  at 90 m or more {int((np.abs(d) >= 90).sum()):6,d}  "
              f"range {d.min():8.1f} to {d.max():7.1f} m")
    print(f"written {cfg.product('signed_distance')}")


if __name__ == "__main__":
    main()
