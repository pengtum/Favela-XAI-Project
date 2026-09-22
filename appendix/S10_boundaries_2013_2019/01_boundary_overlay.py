"""Overlay of the official favela boundaries of 2013 and 2019 on the 2013 footprints.

Manuscript items
* SI Section S10 'Change in the official favela boundaries, 2013-2019' and SI
  Table S8.
* In-text numbers of SI S10: buildings added to the favela boundaries by 2019,
  buildings removed, buildings with the same status in both years (favela in
  both + formal in both), and the sites without change.
* Methods 4.12 'Overlay of the 2013 and 2019 favela boundaries' and the
  sentence on the 2019 overlay in Methods 4.1.

Rule (Methods 4.12; ``boundary_overlay`` in config.yaml): a building counts as
inside in a given year when its footprint intersects the dissolved favela
polygons of that year. Each building then has one of four statuses: favela in
both years, formal in both years, added by 2019 (outside 2013, inside 2019) or
removed by 2019 (inside 2013, outside 2019). Only buildings are counted; no
statistical test is run. The polygons of each year are dissolved once (invalid
polygons repaired first, favela_xai.labels.polygon_union); intersecting the
dissolved union is the same test as intersecting any single polygon of that
year.

Inputs
* Per-site building tables of main/01_features (products.features_gpkg): the
  2013 footprints of the analysed building set with their favela label.
* Official favela polygons 2013 and 2019 (paths.raw.favelas_2013 and
  paths.raw.favelas_2019; Data.Rio, the 2019 polygons accessed in September
  2024).

Outputs (outputs/appendix/S10_boundaries_2013_2019/)
* overlay_counts.csv            per site: the four status counts, the number of
                                buildings, the buildings whose 2013 status differs
                                from their label (zero by construction, since the
                                labels use the same rule), and the centroid rule
                                for comparison only
* table_S8_boundary_status_counts.csv   SI Table S8 (rows in the printed
                                order, sum row)
* table_R1_cadastre_counts.tex  SI Table S8 in LaTeX, with caption and column
                                headers
* overlay_log.txt               printed log with the in-text numbers

Run (from the repository root, after main/01_features):
    python appendix/S10_boundaries_2013_2019/01_boundary_overlay.py
"""
from __future__ import annotations

import pandas as pd

from favela_xai.config import load_config
from favela_xai.io import read_favela_polygons, read_site_features, write_table, write_text
from favela_xai.labels import FAVELA, inside_polygons, polygon_union

STATUS = ["favela_in_both", "formal_in_both", "added_by_2019", "removed_by_2019"]

# Caption and column headers of SI Table S8.
CAPTION = (
    r"\caption{Per-site building counts by favela status in the official favela boundaries of 2013 and of 2019," "\n"
    r"  on the 2013 building footprints. A building is counted as inside the boundaries of a given year if its footprint" "\n"
    r"  intersects a favela polygon of that year. Favela in both: inside in both years. Formal in both: outside in" "\n"
    r"  both years. Added by 2019: outside in 2013, inside in 2019. Removed by 2019: inside in 2013, outside in" "\n"
    r"  2019.}"
)
HEADER = [r" & Favela & Formal & Added & Removed \\",
          r"Site & in both & in both & by 2019 & by 2019 \\"]
CSV_COLUMNS = ["Site", "Favela in both", "Formal in both", "Added by 2019", "Removed by 2019"]


def thousands(x) -> str:
    """Integer with the SI thousands separator, e.g. 15{,}317."""
    return f"{int(x):,}".replace(",", "{,}")


def sum_label(n_sites: int) -> str:
    return "Sum (ten sites)" if n_sites == 10 else f"Sum ({n_sites} site{'s' if n_sites != 1 else ''})"


def is_favela(values: pd.Series):
    """True for favela (label strings or 1/0 codes)."""
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=float) == 1
    return (values.astype(str) == FAVELA).to_numpy()


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    ov = cfg.section("boundary_overlay")
    rule = ov["rule"]
    if rule != "footprint_intersects" or not ov["dissolve_polygons"]:
        raise ValueError("Methods 'Overlay of the 2013 and 2019 favela boundaries': the footprint "
                         "intersects the dissolved favela polygons of that year")
    log_lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        log_lines.append(str(text))

    p13 = read_favela_polygons(cfg, 2013)
    p19 = read_favela_polygons(cfg, 2019)
    u13 = polygon_union(p13)
    u19 = polygon_union(p19)
    log(f"favela polygons: 2013 {len(p13):,}, 2019 {len(p19):,} ({cfg.get('crs')}, dissolved)")

    rows = []
    for site in cfg.sites:
        b = read_site_features(cfg, site, columns=["label"])
        i13 = inside_polygons(b, u13, rule)
        i19 = inside_polygons(b, u19, rule)
        # Centroid rule, for comparison only (it does not define the table)
        c13 = inside_polygons(b, u13, "centroid_within")
        c19 = inside_polygons(b, u19, "centroid_within")
        fav = is_favela(b["label"])
        rows.append(dict(
            site=site,
            n_buildings=len(b),
            favela_in_both=int((i13 & i19).sum()),
            formal_in_both=int((~i13 & ~i19).sum()),
            added_by_2019=int((~i13 & i19).sum()),
            removed_by_2019=int((i13 & ~i19).sum()),
            label_differs_from_2013_overlay=int((fav != i13).sum()),
            label_differs_from_2013_centroid_rule=int((fav != c13).sum()),
            centroid_rule_added=int((~c13 & c19).sum()),
            centroid_rule_removed=int((c13 & ~c19).sum()),
            centroid_vs_footprint_status_differs=int(((c13 != i13) | (c19 != i19)).sum()),
        ))
    counts = pd.DataFrame(rows)
    assert (counts[STATUS].sum(axis=1) == counts["n_buildings"]).all()
    write_table(counts, out / "overlay_counts.csv")

    # ------------------------------------------------------------------ log
    with pd.option_context("display.width", 250):
        log(counts.to_string(index=False))
    tot = counts[STATUS].sum()
    log(f"sums: favela in both {tot.favela_in_both:,}; formal in both {tot.formal_in_both:,}; "
        f"added by 2019 {tot.added_by_2019:,}; removed by 2019 {tot.removed_by_2019:,}")
    log(f"same status in both years (favela in both + formal in both): "
        f"{tot.favela_in_both + tot.formal_in_both:,} of {counts.n_buildings.sum():,} buildings")
    unchanged = [cfg.site_name(r.site) for r in counts.itertuples()
                 if r.added_by_2019 == 0 and r.removed_by_2019 == 0]
    log(f"sites without any change: {', '.join(unchanged) if unchanged else 'none'}")
    log(f"buildings whose 2013 overlay status differs from their label: {counts.label_differs_from_2013_overlay.sum()}")
    log(f"centroid rule (comparison only): added {counts.centroid_rule_added.sum()}, "
        f"removed {counts.centroid_rule_removed.sum()}; status differs from the footprint rule for "
        f"{counts.centroid_vs_footprint_status_differs.sum():,} buildings")

    # ------------------------------------------------------------------ SI Table S8
    order = cfg.display_order("table_S8")
    t = counts.set_index("site").loc[order, STATUS].reset_index()
    printed = pd.DataFrame({CSV_COLUMNS[0]: [cfg.site_name(s) for s in t.site],
                            **{c: t[s].to_numpy() for c, s in zip(CSV_COLUMNS[1:], STATUS)}})
    printed.loc[len(printed)] = [sum_label(len(order))] + [int(tot[s]) for s in STATUS]
    write_table(printed, out / "table_S8_boundary_status_counts.csv")

    lines = ["% Supplementary Table S8 (counts only). Generated by appendix/S10_boundaries_2013_2019/01_boundary_overlay.py",
             r"\begin{table}[H]"]
    lines += [r"\centering", CAPTION, r"\label{tab:site_counts}", r"\small",
              r"\begin{tabular}{lrrrr}", r"\toprule", *HEADER, r"\midrule"]
    for r in t.itertuples():
        lines.append(f"{cfg.site_latex(r.site)} & " + " & ".join(thousands(getattr(r, s)) for s in STATUS) + r" \\")
    lines += [r"\midrule",
              f"{sum_label(len(order))} & " + " & ".join(thousands(tot[s]) for s in STATUS) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_text("\n".join(lines) + "\n", out / "table_R1_cadastre_counts.tex")
    write_text("\n".join(log_lines) + "\n", out / "overlay_log.txt")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
