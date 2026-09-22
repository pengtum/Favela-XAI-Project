"""Step 2 of SI Section S13: boundary profiles without boundary-straddling and near-line buildings.

Manuscript items
* SI Table S17, columns 'All', 'Excl. straddling' and 'Excl. straddling and
  near-line' (sign-change distance x0 of the five leading indicators) and the
  |dS| column (all buildings).
* SI Section S13 'Buildings on the line, and the position of the line': the x0
  ranges (together with step 6).

For each of the 25 indicators the SHAP boundary profile (2.5 m bins, at least 20
buildings per bin and site, mean of the per-site profiles, as in main-text
Figure 4) and its metrics x0, w and dS are recomputed on three building sets:
all buildings, all but the boundary-straddling buildings, and all but the
straddling and near-line buildings. The attributions are those of the main
model and are not refitted.

Inputs: cache/<site>_buildings.parquet (01_prepare_inputs.py).
Outputs (outputs/appendix/S13_boundary_robustness/)
* straddler_metrics.csv: set x indicator: x0, w, dS, |dS|, finite bins,
  buildings within 40 m, and the change relative to all buildings.
* straddler_metrics_key5.csv: the five leading indicators in wide form
  (x0_<set>, w_<set>, dS_<set>), read by 08_table_S17_x0_sensitivity.py.

Run from the repository root:
    python appendix/S13_boundary_robustness/02_straddler_exclusions.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import robustness_common as rc
from favela_xai.io import write_table

# Building sets of SI Table S17
SETS = {
    "full": lambda df: np.ones(len(df), dtype=bool),
    "excl_straddlers": lambda df: ~df["straddler"].to_numpy(),
    "excl_straddlers_and_near": lambda df: ~(df["straddler"].to_numpy() | df["near_line"].to_numpy()),
}


def main() -> None:
    sites = rc.cfg.sites
    data = {s: rc.read_buildings(s, columns=["signed_dist", "straddler", "near_line"] + rc.SHAP) for s in sites}

    rows = []
    for set_name, keep_fn in SETS.items():
        keep = {s: keep_fn(data[s]) for s in sites}
        n_within = int(sum(((np.abs(data[s]["signed_dist"].to_numpy()) <= rc.HALF) & keep[s]).sum() for s in sites))
        for f in rc.INDICATORS:
            per_site = np.vstack([rc.site_profile(data[s]["signed_dist"].to_numpy()[keep[s]],
                                                  data[s]["shap_" + f].to_numpy()[keep[s]]) for s in sites])
            m = rc.profile_metrics(rc.pooled_profile(per_site))
            rows.append(dict(set=set_name, feature=f, x0=m["x0"], w=m["w"], dS=m["dS"], abs_dS=abs(m["dS"]),
                             n_finite_bins=m["n_bins"], n_buildings_within_40m=n_within))
    met = pd.DataFrame(rows)
    full = met[met["set"] == "full"].set_index("feature")
    for c in ("x0", "w", "dS"):
        met["delta_" + c] = met[c].to_numpy() - full.loc[met["feature"], c].to_numpy()
    write_table(met, rc.OUT / "straddler_metrics.csv")

    # five leading indicators, wide form (SI Table S17)
    k5 = met[met["feature"].isin(rc.KEY5)].pivot(index="feature", columns="set", values=["x0", "w", "dS"]).loc[rc.KEY5]
    k5.columns = [f"{metric}_{s}" for metric, s in k5.columns]
    k5 = k5[[f"{metric}_{s}" for metric in ("x0", "w", "dS") for s in SETS]]
    write_table(k5.reset_index(), rc.OUT / "straddler_metrics_key5.csv")

    pd.set_option("display.width", 200)
    print(k5.round(2).to_string())
    # largest changes over the indicators with a resolved transition (|dS| >= 0.1, Methods 4.8)
    strong = full.index[full["abs_dS"] >= float(rc.BP["resolved_min_abs_delta_s"])]
    changes = met[(met["set"] != "full") & met["feature"].isin(strong)].groupby("set")[["delta_x0", "delta_w", "delta_dS"]]
    print("\nlargest |change| over the indicators with |dS| >= 0.1:")
    print(changes.agg(lambda v: np.nanmax(np.abs(v))).round(3).to_string())
    # Check that the five leading indicators are the five largest plateau
    # contrasts of the full set (SI Table S5).
    top5 = list(full["abs_dS"].sort_values(ascending=False).index[:5])
    print("\nlargest |dS| (all buildings):", ", ".join(f"{f} {full.loc[f, 'abs_dS']:.2f}" for f in top5))
    if set(top5) != set(rc.KEY5):
        print("NOTE: appendix_S13_boundary_robustness.leading_indicators differ from the five largest |dS| of this run.")


if __name__ == "__main__":
    main()
