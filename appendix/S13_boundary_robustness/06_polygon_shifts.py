"""Step 6 of SI Section S13: small movements of the 2013 polygons (position of the line).

Manuscript items
* SI Table S17, column 'Polygons shifted 2.5 m': range of x0 over rigid 2.5 m
  translations in eight compass directions.
* SI Table S17 legend and SI S13 last paragraph: largest change of |dS| and
  range of the transition width w under rigid shifts of 2.5, 5 or 10 m and
  under per-polygon random shifts (20 runs per distance); label side, five
  leading indicators; summarised by 08_table_S17_x0_sensitivity.py from
  jitter.csv.

Design (SI S13; SI Table S17; ``boundary_robustness``)
* Rigid: all 2013 polygons translated together by 2.5, 5 or 10 m
  (shift_distances_m) in each of eight compass directions
  (``appendix_S13_boundary_robustness.shift_directions_deg``).
* Per polygon: every polygon translated by the same distance in its own
  uniformly random direction, 20 runs per distance (per_polygon_runs). Run r at
  distance index i and site index j (position in the config site list) draws
  the directions from numpy default_rng(per_polygon_seed_base + 100 i + 11 r +
  1000 j) (``appendix_S13_boundary_robustness.per_polygon_seed_steps``).
* Distances are recomputed to the moved polygons; the side of each building is
  set by polygon membership ('geom') or by its label ('label', the convention of
  main-text Figure 4 used in SI Table S17). Buildings, labels and attributions
  are fixed.
* Profiles and metrics as in main-text Figure 4 (robustness_common).
  Quantities: SHAP of the five leading indicators and the label step.

Inputs: cache/<site>_buildings.parquet, cache/<site>_polygons.parquet
(01_prepare_inputs.py).
Output (outputs/appendix/S13_boundary_robustness/)
* jitter.csv: one 'run' row per movement x side x quantity (x0, w, dS, max
  slope and the change from the unmoved line) and one 'range' row per
  movement type x side x quantity x distance (number of runs, min / max / mean
  and largest absolute change of x0, dS and w).

Run from the repository root:
    python appendix/S13_boundary_robustness/06_polygon_shifts.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import robustness_common as rc
from favela_xai.config import ConfigError
from favela_xai.io import write_table

QUANTITIES = ["shap_" + f for f in rc.KEY5] + ["label"]


def movement_rows(li, sd_geom: dict, model: str, size: float, run: int, direction: str) -> list[dict]:
    """Pooled metrics of the moved line, with the side set by polygon membership and by label."""
    sd_label = {s: rc.label_side(sd_geom[s], li.fav[s]) for s in li.sites}
    rows = []
    for sign, sd in (("geom", sd_geom), ("label", sd_label)):
        _, pooled = rc.line_profiles(sd, li, QUANTITIES)
        for q in QUANTITIES:
            kind, ind = rc.quantity_parts(q)
            m = rc.profile_metrics(pooled[q])
            rows.append(dict(jitter_model=model, jitter_m=size, run=run, direction=direction, sign=sign,
                             quantity=kind, indicator=ind, x0=m["x0"], w=m["w"], dS=m["dS"], maxslope=m["maxslope"]))
    return rows


def main() -> None:
    cfg = rc.cfg
    sizes = [float(v) for v in rc.BR["shift_distances_m"]]
    directions = dict(rc.FS["shift_directions_deg"])
    if len(directions) != int(rc.BR["x0_directions"]):
        raise ConfigError("shift_directions_deg must list boundary_robustness.x0_directions directions")
    n_runs = int(rc.BR["per_polygon_runs"])
    seed0 = int(rc.BR["per_polygon_seed_base"])
    steps = rc.FS["per_polygon_seed_steps"]

    li = rc.line_inputs(cfg.sites)

    # unmoved line: reference values of every quantity, for both sides
    sd_geom, sd_label = rc.real_line(li)
    ref = {}
    for sign, sd in (("geom", sd_geom), ("label", sd_label)):
        _, pooled = rc.line_profiles(sd, li, QUANTITIES)
        for q in QUANTITIES:
            ref[(sign,) + rc.quantity_parts(q)] = rc.profile_metrics(pooled[q])

    rows = []
    # rigid translation of all polygons
    for size in sizes:
        for run, (name, deg) in enumerate(directions.items()):
            dx, dy = size * np.cos(np.radians(deg)), size * np.sin(np.radians(deg))
            sd = {s: rc.polygon_side_distance(rc.translate(li.polygons[s], dx, dy), li.x[s], li.y[s])
                  for s in li.sites}
            rows += movement_rows(li, sd, "rigid_common", size, run, name)
        print(f"rigid shifts of {size:g} m done", flush=True)
    # every polygon in its own random direction
    for i, size in enumerate(sizes):
        for run in range(n_runs):
            sd = {}
            for s in li.sites:
                j = cfg.all_sites.index(s)
                rng = np.random.default_rng(seed0 + int(steps["distance"]) * i + int(steps["run"]) * run
                                            + int(steps["site"]) * j)
                theta = rng.uniform(0, 2 * np.pi, len(li.polygons[s]))
                moved = rc.translate_each(li.polygons[s], size * np.cos(theta), size * np.sin(theta))
                sd[s] = rc.polygon_side_distance(moved, li.x[s], li.y[s])
            rows += movement_rows(li, sd, "per_polygon_random", size, run, "random")
        print(f"per-polygon shifts of {size:g} m done", flush=True)

    runs = pd.DataFrame(rows)
    for key in ("x0", "w", "dS"):
        runs[f"{key}_unjittered"] = [ref[(sg, qk, ind)][key]
                                     for sg, qk, ind in zip(runs["sign"], runs["quantity"], runs["indicator"])]
        runs[f"{key}_dev"] = runs[key] - runs[f"{key}_unjittered"]

    def max_abs(v):
        v = np.abs(np.asarray(v, dtype=float))
        return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan

    ranges = runs.groupby(["jitter_model", "sign", "quantity", "indicator", "jitter_m"], sort=False).agg(
        n_runs=("x0", "size"), n_x0_nan=("x0", lambda v: int(v.isna().sum())),
        x0_unjittered=("x0_unjittered", "first"), x0_min=("x0", "min"), x0_max=("x0", "max"), x0_mean=("x0", "mean"),
        x0_maxabsdev=("x0_dev", max_abs), x0_meandev=("x0_dev", "mean"),
        dS_unjittered=("dS_unjittered", "first"), dS_min=("dS", "min"), dS_max=("dS", "max"),
        dS_maxabsdev=("dS_dev", max_abs),
        w_unjittered=("w_unjittered", "first"), w_min=("w", "min"), w_max=("w", "max"),
        w_maxabsdev=("w_dev", max_abs)).reset_index()
    ranges["x0_range"] = ranges["x0_max"] - ranges["x0_min"]
    runs["row_type"], ranges["row_type"] = "run", "range"
    write_table(pd.concat([runs, ranges], ignore_index=True), rc.OUT / "jitter.csv")

    pd.set_option("display.width", 220)
    show = ranges[(ranges["sign"] == "label") & (ranges["quantity"] == "shap")]
    print(show[["jitter_model", "indicator", "jitter_m", "x0_unjittered", "x0_min", "x0_max", "dS_maxabsdev",
                "w_min", "w_max"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
