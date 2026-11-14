"""Step 1: clean the 2013 building footprints, cut the site windows and label every building.

Manuscript items
  * Methods 4.2 and SI Section S1 (footprint pre-processing): validity check,
    geometry repair (structure method), removal of duplicates, multipart to
    singlepart, Douglas-Peucker simplification (tolerance 0.1 m), second
    validity check (favela_xai.features.preprocess_footprints).
  * Methods 4.1 (study area): the ten site windows and the favela /
    formal-settlement labels. The totals of
    Methods 4.1 are reported by 04_site_summary.py.

Inputs (config.yaml)
  paths.raw.buildings_2013   building footprints 2013 (Data.Rio), all Tipo classes
  paths.raw.favelas_2013     official favela polygons 2013 (Data.Rio)
  sites[].window             site windows, EPSG:31983
  preprocessing.*, labels.*  cleaning steps; label rule (footprint intersects the dissolved 2013 polygons)

Outputs (outputs/main/01_features/)
  <site>_footprints_clean.gpkg     cleaned footprints of the window with label ('favela' / 'not-favela')
  steps/<site>_preprocessing.csv   counts per cleaning step (collected by 04_site_summary.py)

Run from the repository root (all sites, or a subset):
  python main/01_features/01_preprocess_footprints.py [--sites palmeras rocinha]
"""
from __future__ import annotations

import argparse

import pandas as pd

from favela_xai.config import load_config
from favela_xai.features import preprocess_footprints, select_window
from favela_xai.io import read_favela_polygons, read_site_footprints, write_gpkg, write_table
from favela_xai.labels import FAVELA, assign_labels


def parse_sites(cfg, description: str) -> list[str]:
    """Sites given with --sites (default: every site in config.yaml)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sites", nargs="+", help="site keys to process (default: all)")
    args = parser.parse_args()
    sites = args.sites or cfg.sites
    unknown = sorted(set(sites) - set(cfg.all_sites))
    if unknown:
        raise SystemExit(f"unknown site key(s): {unknown}; valid keys: {cfg.all_sites}")
    return sites


def main() -> None:
    cfg = load_config()
    sites = parse_sites(cfg, "Pre-process the 2013 footprints of each site window and label them.")
    out_dir = cfg.output_dir(__file__)
    keep_cols = list(cfg.get("main_01_features.keep_source_columns", []))
    predicate = cfg.get("preprocessing.window_predicate")

    polygons_2013 = read_favela_polygons(cfg, 2013)
    for site in sites:
        # Footprints around the window (padded read), cleaned as in Methods 4.2,
        # then the window itself is applied.
        raw = read_site_footprints(cfg, site)
        clean, report = preprocess_footprints(raw, cfg)
        window = select_window(clean, cfg.site_window(site), predicate)

        # Favela label: footprint intersects the dissolved 2013 favela polygons.
        window["label"] = assign_labels(window, polygons_2013, cfg).to_numpy()
        columns = [c for c in keep_cols if c in window.columns] + ["label", window.geometry.name]
        window = window[columns]
        write_gpkg(window, cfg.product("footprints_clean", site=site))

        n_favela = int((window["label"] == FAVELA).sum())
        row = {
            "site": site,
            "read_padded_window": report["input"],
            "removed_null_or_empty": report.get("removed_null_or_empty", 0),
            "repaired_invalid": report.get("repaired_invalid", 0),
            "removed_unrepairable": report.get("removed_unrepairable", 0),
            "removed_duplicates": report.get("removed_duplicates", 0),
            "added_by_singlepart": report.get("added_by_singlepart", 0),
            "removed_final_check": report.get("removed_final_check", 0),
            "cleaned_padded_window": report["output"],
            "in_window": int(len(window)),
            "favela_in_window": n_favela,
        }
        write_table(pd.DataFrame([row]), out_dir / "steps" / f"{site}_preprocessing.csv")
        print(f"{site:24s} read {row['read_padded_window']:7,d}  in window {row['in_window']:6,d}  "
              f"favela {n_favela:6,d}  (site extract records: "
              f"{cfg.site_info(site)['extract_records']:,d})")


if __name__ == "__main__":
    main()
