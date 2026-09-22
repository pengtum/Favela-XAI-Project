"""Step 1: administrative context layers of Rio de Janeiro (municipality, planning areas, bairros).

Manuscript items
* Supplementary Fig. S1a: municipal boundary and planning areas AP1-AP5.
* Supplementary Table S1: bairro and planning area of each site (used by
  06_site_context_table.py).

Inputs
* ``paths.raw.municipality``, ``paths.raw.planning_areas``, ``paths.raw.bairros``:
  GeoJSON exports of layers 0, 1 and 4 of the Data.Rio administrative-boundaries
  service (Prefeitura da Cidade do Rio de Janeiro, CC BY 4.0). With
  ``--download`` the layers are fetched from
  ``appendix_S02_site_selection.context_service_url`` and written to those paths.

Outputs (outputs/appendix/S02_site_selection/context_layers/)
* ``rio_planning_areas.gpkg``: layers ``planning_areas`` (full resolution) and
  ``planning_areas_simplified`` (map only), columns ap (AP1-AP5), code, zone, area_km2.
* ``rio_municipality.gpkg``: layers ``municipality`` and ``municipality_simplified``.
* ``rio_bairros.gpkg``: layer ``bairros`` with the bairro name in column ``bairro``.
* ``context_layers_check.txt``: check that the five planning areas do not
  overlap and together equal the municipal boundary.

Settings
* ``site_context.context_simplify_m``: simplification tolerance of the map layers.
* ``appendix_S02_site_selection``: service address and layer ids, attribute names
  of the planning-area code and the bairro name, planning-area names.

Run from the repository root:
    python appendix/S02_site_selection/01_context_layers.py [--download]
"""
from __future__ import annotations

import argparse
import urllib.request

import geopandas as gpd
import shapely

from favela_xai.config import load_config
from favela_xai.io import read_vector, write_gpkg, write_text

import s02_lib as L


def download(cfg) -> None:
    """Fetch the three layers as GeoJSON from the Data.Rio service to their raw-data paths."""
    base = cfg.get("appendix_S02_site_selection.context_service_url").rstrip("/")
    ids = cfg.get("appendix_S02_site_selection.context_layer_ids")
    for name in ("municipality", "planning_areas", "bairros"):
        rel = cfg.get(f"paths.raw.{name}")
        path = cfg.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{int(ids[name])}/query?where=1%3D1&outFields=*&returnGeometry=true&f=geojson"
        print(f"  GET {url}")
        with urllib.request.urlopen(url, timeout=180) as r:
            path.write_bytes(r.read())
        print(f"  wrote {path}")


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--download", action="store_true",
                     help="fetch the three layers from Data.Rio before processing")
    args = ap_.parse_args()

    cfg = load_config()
    s = cfg.section("appendix_S02_site_selection")
    out = cfg.output_dir(__file__)
    (out / L.CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    if args.download:
        download(cfg)
    log = []

    def P(msg: str) -> None:
        print(msg)
        log.append(msg)

    # --- read (GeoJSON from the service is WGS84; read_vector reprojects to EPSG:31983)
    ap = read_vector(cfg.raw_path("planning_areas"), cfg)
    mun = read_vector(cfg.raw_path("municipality"), cfg)
    bai = read_vector(cfg.raw_path("bairros"), cfg)

    # --- planning areas: five features with codes 1-5 that do not overlap
    code_field = s["planning_area_code_field"]
    codes = sorted(str(v) for v in ap[code_field])
    P(f"[planning areas] {len(ap)} features, codes {codes}")
    assert len(ap) == 5 and codes == ["1", "2", "3", "4", "5"], "expected the five planning areas AP1-AP5"
    names = {str(k): v for k, v in s["planning_area_names"].items()}
    ap = ap.assign(code=ap[code_field].astype(str))
    ap["ap"] = "AP" + ap["code"]
    ap["zone"] = ap["code"].map(lambda c: names[c].split(" ", 1)[1])
    ap["area_km2"] = ap.geometry.area / 1e6
    ap = ap[["ap", "code", "zone", "area_km2", "geometry"]].sort_values("code").reset_index(drop=True)
    dissolved = shapely.union_all(ap.geometry.values)
    P("[planning areas] area km2: " + ", ".join(f"{r.ap} {r.area_km2:.2f}" for r in ap.itertuples())
      + f"; sum {ap.area_km2.sum():.2f}; dissolved {dissolved.area / 1e6:.2f}")
    P(f"[planning areas] overlap (sum - dissolved) {ap.area_km2.sum() - dissolved.area / 1e6:.6f} km2")

    # --- municipality: one feature equal to the dissolved planning areas
    assert len(mun) == 1, "expected one municipal boundary feature"
    mu = shapely.union_all(mun.geometry.values)
    P(f"[municipality] area {mu.area / 1e6:.2f} km2; symmetric difference with the dissolved planning areas "
      f"{dissolved.symmetric_difference(mu).area / 1e6:.6f} km2")
    mun = gpd.GeoDataFrame({"area_km2": [mu.area / 1e6]}, geometry=[mu], crs=cfg.get("crs"))

    # --- bairros: keep the name only (first configured attribute name present in the layer)
    name_field = next((f for f in s["bairro_name_fields"] if f in bai.columns), None)
    if name_field is None:
        raise KeyError(f"bairro layer has none of the attributes {s['bairro_name_fields']} "
                       f"(appendix_S02_site_selection.bairro_name_fields); attributes: {list(bai.columns)}")
    bai = gpd.GeoDataFrame({"bairro": bai[name_field].astype(str).str.strip()}, geometry=bai.geometry.values,
                           crs=cfg.get("crs"))
    P(f"[bairros] {len(bai)} bairros (name attribute '{name_field}')")

    # --- write (full resolution + simplified map layers)
    tol = float(cfg.get("site_context.context_simplify_m"))
    sim = ap.copy()
    sim["geometry"] = sim.geometry.simplify(tol, preserve_topology=True)
    ms = mun.copy()
    ms["geometry"] = ms.geometry.simplify(tol, preserve_topology=True)
    pa_path = out / L.PLANNING_AREAS_GPKG
    mu_path = out / L.MUNICIPALITY_GPKG
    write_gpkg(ap, pa_path, layer="planning_areas")
    sim.to_file(pa_path, layer="planning_areas_simplified", driver="GPKG")
    write_gpkg(mun, mu_path, layer="municipality")
    ms.to_file(mu_path, layer="municipality_simplified", driver="GPKG")
    write_gpkg(bai, out / L.BAIRROS_GPKG, layer="bairros")
    P(f"[simplified {tol:g} m] planning-area area change {abs(sim.geometry.area.sum() - ap.geometry.area.sum()) / 1e4:.1f} ha")
    write_text("\n".join(log) + "\n", out / L.CONTEXT_DIR / "context_layers_check.txt")
    print(f"written to {out / L.CONTEXT_DIR}")


if __name__ == "__main__":
    main()
