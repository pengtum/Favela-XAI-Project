"""Step 4: building counts per site and the pre-processing report.

Manuscript items
  * Methods 4.1 (also abstract and introduction): building footprints of the ten
    site windows and those within official favela boundaries, before the
    tessellation stage (columns in_window, favela_in_window; printed next to the
    totals stated in Methods 4.1, main_01_features.manuscript_totals), and
    buildings analysed (column analysed).
  * Methods 4.2 and SI Section S1: geometries discarded at the tessellation
    stage (columns tessellation_excluded_total, excluded_share_pct; printed next to
    the figure stated in Methods 4.2, 443 of the 188,735 footprints, 0.23%).
  * Fig. 1a: per-class building counts and position of each site
    (site_counts.csv, read by 08_figure1_study_area.py).

Inputs (outputs/main/01_features/)
  steps/<site>_*.csv and <site>_buildings_features.gpkg written by steps 01-03;
  config.yaml: sites[].extract_records, main_01_features.manuscript_totals

Outputs (outputs/main/01_features/)
  preprocessing_report.csv       products.preprocessing_report: per site and in total,
                                 footprints read, repaired and removed per rule, in the
                                 window (all / favela), excluded per tessellation rule,
                                 analysed (favela / formal), share excluded at tessellation,
                                 record count of the site extract
  site_counts.csv                products.site_counts: site, name, n, n_fav, n_non, lon, lat
                                 (centroid of the analysed footprints, EPSG:4326)
  indicator_missing_values.csv   missing indicator values, site x indicator

Run from the repository root (after steps 01-03 for all sites):
  python main/01_features/04_site_summary.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from favela_xai.config import load_config
from favela_xai.io import read_site_features, write_table
from favela_xai.labels import FAVELA

TESSELLATION_REASONS = ["null_geometry", "invalid_geometry", "empty_geometry", "not_polygon", "too_small",
                        "tessellation_failure"]


def read_step_table(out_dir, site: str, step: str, script: str) -> pd.DataFrame:
    path = out_dir / "steps" / f"{site}_{step}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run main/01_features/{script} for {site} first")
    return pd.read_csv(path)


def site_position(cfg, site: str) -> dict:
    """Building counts and the lon/lat of the centroid of the union of the analysed footprints."""
    g = read_site_features(cfg, site, columns=["label"])
    fav = (g["label"] == FAVELA).to_numpy()
    union = shapely.union_all(np.asarray(g.geometry.values, dtype=object))
    centre = gpd.GeoSeries([union.centroid], crs=g.crs).to_crs(4326).iloc[0]
    return dict(site=site, name=cfg.site_name(site), n=int(len(g)), n_fav=int(fav.sum()),
                n_non=int((~fav).sum()), lon=float(centre.x), lat=float(centre.y))


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    sites = cfg.sites

    # ---- pre-processing report: steps 1 and 2 side by side
    rows = []
    for site in sites:
        pre = read_step_table(out_dir, site, "preprocessing", "01_preprocess_footprints.py").iloc[0]
        tes = read_step_table(out_dir, site, "tessellation", "02_tessellation.py").iloc[0]
        row = {"site": site, "name": cfg.site_name(site),
               "extract_records": int(cfg.site_info(site)["extract_records"])}
        for col in ["read_padded_window", "removed_null_or_empty", "repaired_invalid", "removed_unrepairable",
                    "removed_duplicates", "added_by_singlepart", "removed_final_check", "cleaned_padded_window",
                    "in_window", "favela_in_window"]:
            row[col] = int(pre[col])
        for reason in TESSELLATION_REASONS:
            row[f"tessellation_excluded_{reason}"] = int(tes.get(f"excluded_{reason}", 0))
        row["tessellation_excluded_total"] = int(tes["excluded_total"])
        row["excluded_share_pct"] = 100.0 * row["tessellation_excluded_total"] / max(row["in_window"], 1)
        row["analysed"] = int(tes["tessellation_kept"])
        row["favela_analysed"] = int(tes["favela_kept"])
        row["formal_analysed"] = row["analysed"] - row["favela_analysed"]
        rows.append(row)
    report = pd.DataFrame(rows)
    total = report.drop(columns=["site", "name", "excluded_share_pct"]).sum(numeric_only=True)
    total = dict(total.astype(int), site="total", name="all sites",
                 excluded_share_pct=100.0 * total["tessellation_excluded_total"] / max(total["in_window"], 1))
    report = pd.concat([report, pd.DataFrame([total])], ignore_index=True)[report.columns]
    write_table(report, cfg.product("preprocessing_report"))

    # ---- per-site counts and positions (Fig. 1a)
    counts = pd.DataFrame([site_position(cfg, s) for s in sites])
    write_table(counts, cfg.product("site_counts"))

    # ---- missing indicator values
    missing = pd.concat([read_step_table(out_dir, s, "indicator_missing", "03_indicators.py") for s in sites],
                        ignore_index=True)
    write_table(missing.pivot(index="site", columns="indicator", values="n_missing").reindex(sites).reset_index(),
                out_dir / "indicator_missing_values.csv")

    # ---- totals (Methods 4.1, 4.2)
    stated = cfg.get("main_01_features.manuscript_totals")
    t = report.iloc[-1]
    print(f"footprints in the site windows: {int(t['in_window']):,} ({int(t['favela_in_window']):,} favela); "
          f"Methods 4.1: {stated['buildings']:,} ({stated['favela']:,} favela)")
    print(f"record count of the site extracts: "
          f"{int(t['extract_records']):,}")
    print(f"analysed after the tessellation stage: {int(t['analysed']):,} ({int(t['favela_analysed']):,} favela)")
    print("share of footprints excluded at the tessellation stage (Methods 4.2: 443 of the 188,735 footprints, 0.23%):")
    for r in report.itertuples():
        print(f"  {r.name:24s} {r.tessellation_excluded_total:5d} of {r.in_window:7,d}  ({r.excluded_share_pct:.2f}%)")


if __name__ == "__main__":
    main()
