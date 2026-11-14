"""SI Section S6: recorded heights of machine-room and ordinary building polygons.

In SI Section S6 the numerator of the adjacency ratio (SI Section S3, item 19)
rises with the number of roof-level structures nearby, which stand on taller
buildings. The heights quoted there are computed here: among the analysed
polygons with a recorded height, the median height of the polygons that the
cadastre classes as machine rooms (casa de máquina) against that of the
ordinary building polygons, over all sites and in each site.

Method
* cadastre class: attribute ``Tipo`` of the 2013 building layer. Machine
  rooms: class name containing CASA_DE_MAQUINA (A122 - CASA_DE_MAQUINA);
  ordinary building polygons: class name containing EDIFICACAO
  (A101 - EDIFICACAO, A102 - EDIFICACAO_FAVELA). The other classes are not
  compared;
* recorded height: attribute ``ALTURA`` of the same layer (m); only polygons
  with ALTURA > 0 (a recorded height) enter;
* median height of each group over the polygons of all sites (pooled) and
  within each site; per site, whether the machine-room median exceeds the
  ordinary-building median (a site without a machine room with a recorded
  height is reported and not counted).

Inputs
* products.features_gpkg (main/01_features): ``Tipo`` and ``ALTURA`` of every
  analysed building, kept from the 2013 building layer
  (``main_01_features.keep_source_columns``).
Outputs (outputs/appendix/S06_attribution_analyses/)
* recorded_heights_by_site.csv: per site, and over all sites (final row
  'all_sites'), the number of polygons of each group, the number with a
  recorded height, their median height (m) and whether the machine-room median
  exceeds the ordinary-building median.

Run from the repository root (after main/01_features):
    python appendix/S06_attribution_analyses/07_recorded_heights.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_table

TABLE = "recorded_heights_by_site.csv"
CLASS_FIELD = "Tipo"       # cadastre class of the 2013 building layer
HEIGHT_FIELD = "ALTURA"    # recorded height of the 2013 building layer (m)
# Groups compared: (key, text contained in the class name). A polygon belongs
# to the first group whose text its class name contains.
GROUPS = [("machine_room", "CASA_DE_MAQUINA"), ("ordinary", "EDIFICACAO")]


def site_polygons(cfg, site: str) -> pd.DataFrame:
    """Cadastre class and recorded height of the analysed polygons of one site."""
    path = cfg.require_product("features_gpkg", site=site)
    fields = set(pyogrio.read_info(path)["fields"])
    missing = [c for c in (CLASS_FIELD, HEIGHT_FIELD) if c not in fields]
    if missing:
        raise ConfigError(f"{path} lacks the columns {missing}; keep them with "
                          "main_01_features.keep_source_columns and re-run main/01_features.")
    raw = pyogrio.read_dataframe(path, columns=[CLASS_FIELD, HEIGHT_FIELD], read_geometry=False)
    return pd.DataFrame({"site": site, "tipo": raw[CLASS_FIELD].to_numpy(),
                         "height_m": pd.to_numeric(raw[HEIGHT_FIELD], errors="coerce").to_numpy()})


def assign_groups(tipo: pd.Series) -> pd.Series:
    """Group key of every polygon from its cadastre class ('' = not compared)."""
    names = tipo.fillna("").astype(str)
    group = pd.Series("", index=tipo.index, dtype=object)
    for key, text in GROUPS:
        group[(group == "") & names.str.contains(text, regex=False)] = key
    return group


def summarise(polygons: pd.DataFrame) -> dict:
    """Polygons, polygons with a recorded height and their median height, per group."""
    row = {}
    for key, _ in GROUPS:
        h = polygons.loc[polygons["group"] == key, "height_m"]
        recorded = h[h > 0]
        row[f"n_{key}"] = int(len(h))
        row[f"n_{key}_with_height"] = int(len(recorded))
        row[f"median_height_{key}_m"] = float(recorded.median()) if len(recorded) else np.nan
    mr, ordinary = row["median_height_machine_room_m"], row["median_height_ordinary_m"]
    row["machine_room_median_higher"] = bool(mr > ordinary) if np.isfinite(mr) and np.isfinite(ordinary) else None
    return row


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    sites = cfg.sites

    polygons = pd.concat([site_polygons(cfg, s) for s in sites], ignore_index=True)
    polygons["group"] = assign_groups(polygons["tipo"])
    rows = [dict(site=s, site_name=cfg.site_name(s), **summarise(polygons[polygons["site"] == s])) for s in sites]
    rows.append(dict(site="all_sites", site_name="All sites (pooled)", **summarise(polygons)))
    table = pd.DataFrame(rows)
    write_table(table, out / TABLE)

    # ---- values quoted in SI Section S6 --------------------------------------------
    per_site, pooled = table.iloc[:-1], table.iloc[-1]
    defined = per_site["machine_room_median_higher"].notna()
    n_higher = int(per_site.loc[defined, "machine_room_median_higher"].astype(bool).sum())
    without = per_site.loc[~defined, "site_name"].tolist()
    lines = [
        "Median height of the analysed polygons with a recorded height (ALTURA > 0):",
        "  machine rooms (casa de máquina)  {:5.1f} m  ({:,} of {:,} polygons)".format(
            pooled["median_height_machine_room_m"], pooled["n_machine_room_with_height"], pooled["n_machine_room"]),
        "  ordinary building polygons       {:5.1f} m  ({:,} of {:,} polygons)".format(
            pooled["median_height_ordinary_m"], pooled["n_ordinary_with_height"], pooled["n_ordinary"]),
        "  (SI Section S6: 14.8 m against 4.0 m)",
        "{:32s}  {:>13s}  {:>9s}".format("Median per site (m):", "machine rooms", "ordinary"),
        *["  {:30s}  {:13.1f}  {:9.1f}".format(r.site_name, r.median_height_machine_room_m, r.median_height_ordinary_m)
          for r in per_site.itertuples()],
        f"Machine-room median above the ordinary-building median in {n_higher} of {int(defined.sum())} sites"
        + (" (no machine room with a recorded height: " + ", ".join(without) + ")" if without else ""),
        "  (SI Section S6: the difference holds in each of the ten sites)",
    ]
    print("\n".join(lines))
    print(f"[saved] {out / TABLE}")


if __name__ == "__main__":
    main()
