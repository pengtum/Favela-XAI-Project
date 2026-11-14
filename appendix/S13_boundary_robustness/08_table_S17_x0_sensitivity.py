"""Step 8 of SI Section S13: Supplementary Table S17 (sensitivity of the sign-change distance x0).

Manuscript items
* SI Table S17: x0 (m) of the SHAP boundary profiles of the five leading
  indicators for all buildings, without boundary-straddling buildings, without
  straddling and near-line buildings, and the range over rigid 2.5 m polygon
  shifts in eight compass directions; |dS| of all buildings.
* The values in the SI Table S17 legend (and SI S13 last paragraph): largest
  |dS| change and w range over rigid shifts of 2.5, 5 or 10 m and over
  per-polygon random shifts (20 runs per distance), label side, five leading
  indicators.
* The x0 ranges quoted in SI S13 for each group of indicators (printed at the
  end of the run).

The sign convention is that of main-text Figure 4 (negative on the
favela-labelled side). Nothing is fitted: the script reads the outputs of steps
1, 2, 5 and 6 and checks that the 'All' column equals the x0 of SI Table S5
(main/04_boundary_profiles) and of the unmoved line of step 6.

Inputs (outputs/appendix/S13_boundary_robustness/): straddler_metrics_key5.csv
(02), straddler_counts.csv (01), jitter.csv (06), sanity.csv (05);
outputs/main/04_boundary_profiles/transition_metrics.csv (``products.transition_metrics``).
Outputs (outputs/appendix/S13_boundary_robustness/)
* x0_sensitivity.csv: one row per indicator (numbers and formatted strings,
  w for the building sets and the shifted polygons, polygon-side x0).
* x0_sensitivity_note_values.csv: the caption values by movement type and distance.
* table_R1_x0_sensitivity.tex: SI Table S17 (booktabs; $-$ minus).

Run from the repository root:
    python appendix/S13_boundary_robustness/08_table_S17_x0_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import robustness_common as rc
from favela_xai.config import ConfigError
from favela_xai.io import write_table, write_text

# Row names as printed in SI Table S17
NAME = {"mean_interbuilding_distance": "Mean interbuilding distance", "adjacency": "Adjacency",
        "perimeter_wall": "Perimeter wall", "squareness": "Squareness",
        "avg_weighted_dist": "Average weighted distance"}
MINUS = "−"


def fmt1(v: float) -> str:
    """One decimal with a true minus sign (CSV)."""
    return f"+{v:.1f}" if v >= 0.05 else (f"{MINUS}{abs(v):.1f}" if v <= -0.05 else "0.0")


def tex(v: float, digits: int = 1, sign: bool = True) -> str:
    """Number in the SI style, with $+$ / $-$ signs."""
    s = f"{abs(v):.{digits}f}"
    if not sign:
        return s
    return ("$-$" if v <= -0.5 * 10 ** -digits else "$+$") + s


def tex_list(values) -> str:
    """'2.5, 5 or 10' from [2.5, 5, 10]."""
    items = [tex(v, 1, False).removesuffix(".0") if float(v).is_integer() else tex(v, 1, False)
             for v in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def main() -> None:
    cfg = rc.cfg
    rigid_m = float(rc.BR["x0_rigid_shift_m"])
    n_dirs = int(rc.BR["x0_directions"])
    n_runs = int(rc.BR["per_polygon_runs"])
    sizes = [float(v) for v in rc.BR["shift_distances_m"]]

    K = pd.read_csv(rc.OUT / "straddler_metrics_key5.csv").set_index("feature").loc[rc.KEY5]
    CNT = pd.read_csv(rc.OUT / "straddler_counts.csv").set_index("site").loc["all"]
    J = pd.read_csv(rc.OUT / "jitter.csv")
    SAN = pd.read_csv(rc.OUT / "sanity.csv").set_index("indicator").loc[rc.KEY5]
    JR = J[(J["row_type"] == "range") & (J["quantity"] == "shap") & J["indicator"].isin(rc.KEY5)]
    lab = JR[JR["sign"] == "label"]
    r25 = lab[(lab["jitter_model"] == "rigid_common") & (lab["jitter_m"] == rigid_m)].set_index("indicator").loc[rc.KEY5]

    # ------------------------------------------------------------ consistency checks
    if not ((r25["n_runs"] == n_dirs).all() and (r25["n_x0_nan"] == 0).all()):
        raise ConfigError(f"jitter.csv: expected {n_dirs} rigid {rigid_m:g} m runs with a sign change for every indicator")
    if not np.allclose(r25["x0_unjittered"].to_numpy(), K["x0_full"].to_numpy(), atol=1e-3):
        raise ConfigError("x0 of the unmoved line (06) differs from x0 of all buildings (02)")
    if not np.allclose(SAN["x0_label_sign"].to_numpy(), K["x0_full"].to_numpy(), atol=1e-3):
        raise ConfigError("x0 of the real line in sanity.csv (05) differs from x0 of all buildings (02)")
    t5_path = cfg.product("transition_metrics")
    if t5_path.exists():
        t5 = pd.read_csv(t5_path).set_index("feature").loc[rc.KEY5]
        if not np.allclose(t5["x0"].to_numpy(), K["x0_full"].to_numpy(), atol=1e-3):
            raise ConfigError("the 'All' column differs from x0 in SI Table S5 (main/04_boundary_profiles)")
        print("check: 'All' column equals x0 of SI Table S5 (main/04_boundary_profiles): OK")
    else:
        print(f"note: {t5_path} not found; the 'All' column was not compared with SI Table S5")
    n_str, n_near_only = int(CNT["n_straddlers"]), int(CNT["n_near_not_straddler"])
    print(f"boundary-straddling buildings {n_str:,}; near-line (not straddling) {n_near_only:,}; "
          f"excluded together {n_str + n_near_only:,}")

    # ------------------------------------------------------------ table values
    rows = []
    for f in rc.KEY5:
        k, r, s = K.loc[f], r25.loc[f], SAN.loc[f]
        rows.append(dict(
            indicator=f, indicator_name=NAME[f], abs_dS_all=abs(k.dS_full),
            x0_all=k.x0_full, x0_excl_straddling=k.x0_excl_straddlers,
            x0_excl_straddling_nearline=k.x0_excl_straddlers_and_near,
            x0_shift_min=r.x0_min, x0_shift_max=r.x0_max, x0_shift_max_abs_change=r.x0_maxabsdev,
            w_all=k.w_full, w_excl_straddling=k.w_excl_straddlers, w_excl_straddling_nearline=k.w_excl_straddlers_and_near,
            w_shift_min=r.w_min, w_shift_max=r.w_max,
            abs_dS_excl_straddling=abs(k.dS_excl_straddlers), abs_dS_excl_straddling_nearline=abs(k.dS_excl_straddlers_and_near),
            x0_polygon_side=s.x0_geom_sign,
            x0_all_fmt=fmt1(k.x0_full), x0_excl_straddling_fmt=fmt1(k.x0_excl_straddlers),
            x0_excl_straddling_nearline_fmt=fmt1(k.x0_excl_straddlers_and_near),
            x0_shift_fmt=f"{fmt1(r.x0_min)} to {fmt1(r.x0_max)}"))
    T = pd.DataFrame(rows)
    write_table(T, rc.OUT / "x0_sensitivity.csv")

    # ------------------------------------------------------------ caption values (label side, five indicators)
    note = []
    for model, desc, n_each in (("rigid_common", f"rigid shift, {n_dirs} directions", n_dirs),
                                ("per_polygon_random", f"random per-polygon shift, {n_runs} runs", n_runs)):
        q = lab[lab["jitter_model"] == model]
        for m in sizes + ["all"]:
            qq = q if m == "all" else q[q["jitter_m"] == m]
            note.append(dict(shift=desc, distance_m=m, n_runs_each=n_each, max_abs_dS_change=qq["dS_maxabsdev"].max(),
                             w_min=qq["w_min"].min(), w_max=qq["w_max"].max(), max_abs_x0_change=qq["x0_maxabsdev"].max()))
    NT = pd.DataFrame(note)
    write_table(NT, rc.OUT / "x0_sensitivity_note_values.csv")
    rig = NT[NT["shift"].str.startswith("rigid") & (NT["distance_m"] == "all")].iloc[0]
    per = NT[NT["shift"].str.startswith("random") & (NT["distance_m"] == "all")].iloc[0]

    # ------------------------------------------------------------ LaTeX table and caption
    body = [f"{NAME[f]} & {tex(t.abs_dS_all, 2, False)} & {tex(t.x0_all)} & {tex(t.x0_excl_straddling)} & "
            f"{tex(t.x0_excl_straddling_nearline)} & {tex(t.x0_shift_min)} to {tex(t.x0_shift_max)} \\\\"
            for f, t in T.set_index("indicator").iterrows()]
    caption = (
        r"Sign-change distance $x_0$ (m) of the SHAP boundary profiles of the five leading indicators under "
        r"alternative treatments of buildings at the 2013 favela boundary. Attributions are those of the main-text "
        r"model; binning, the minimum of 20 buildings per bin and site, smoothing and the distance sign (negative on "
        r"the favela-labelled side) are as in main-text Figure~4. Boundary-straddling: footprints that intersect a "
        r"2013 polygon boundary. Near-line: centroid closer to the boundary than the square root of the footprint "
        r"area. Polygons shifted: all 2013 polygons translated rigidly, range over eight compass directions. "
        r"$|\Delta S|$: mean SHAP value of bins at $-$25\,m or less minus that of bins at $+$25\,m or more, all "
        rf"buildings. Rigid shifts of {tex_list(sizes)}\,m change $|\Delta S|$ by at most "
        rf"{tex(rig.max_abs_dS_change, 2, False)} and keep the transition width $w$ within "
        rf"{tex(rig.w_min, 1, False)}--{tex(rig.w_max, 1, False)}\,m; shifting each polygon in its own random "
        rf"direction ({n_runs} runs per distance) changes $|\Delta S|$ by at most "
        rf"{tex(per.max_abs_dS_change, 2, False)} and keeps $w$ within "
        rf"{tex(per.w_min, 1, False)}--{tex(per.w_max, 1, False)}\,m.")
    shift_head = tex(rigid_m, 1, False).removesuffix(".0") if rigid_m.is_integer() else tex(rigid_m, 1, False)
    lines = [
        "% Generated by appendix/S13_boundary_robustness/08_table_S17_x0_sensitivity.py (x0_sensitivity.csv).",
        "% Requires booktabs, float.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:x0_sensitivity}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Buildings included} & Polygons \\",
        r"\cmidrule(lr){3-5}",
        r"Indicator & $|\Delta S|$ & All & Excl.\ straddling & Excl.\ straddling and near-line & shifted "
        + shift_head + r"\,m \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}", ""]
    write_text("\n".join(lines), rc.OUT / "table_R1_x0_sensitivity.tex")

    # ------------------------------------------------------------ read-out (SI S13 text)
    pd.set_option("display.width", 200)
    print(T[["indicator_name", "abs_dS_all", "x0_all_fmt", "x0_excl_straddling_fmt",
             "x0_excl_straddling_nearline_fmt", "x0_shift_fmt"]].to_string(index=False))
    print(NT.round(3).to_string(index=False))
    # indicator groups of the x0 ranges in SI S13 (squareness: without the 'All' column)
    all_cols = ["x0_all", "x0_excl_straddling", "x0_excl_straddling_nearline", "x0_shift_min", "x0_shift_max"]
    for group, feats, cols in (
            ("mean interbuilding distance, adjacency, perimeter wall",
             ["mean_interbuilding_distance", "adjacency", "perimeter_wall"], all_cols),
            ("squareness (buildings excluded or polygons moved)", ["squareness"], all_cols[1:]),
            ("average weighted distance", ["avg_weighted_dist"], all_cols)):
        feats = [f for f in feats if f in rc.KEY5]
        if feats:
            v = T.set_index("indicator").loc[feats, cols].to_numpy().ravel()
            print(f"x0 range, {group}: {fmt1(np.nanmin(v))} to {fmt1(np.nanmax(v))} m")
    print(f"caption: rigid |dS| change <= {rig.max_abs_dS_change:.3f}, w {rig.w_min:.2f}-{rig.w_max:.2f} m; "
          f"per-polygon |dS| change <= {per.max_abs_dS_change:.3f}, w {per.w_min:.2f}-{per.w_max:.2f} m")
    print(f"saved {rc.OUT / 'table_R1_x0_sensitivity.tex'}")


if __name__ == "__main__":
    main()
