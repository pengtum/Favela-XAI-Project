"""SI Table S5: boundary-transition metrics of the 25 indicators (LaTeX).

Manuscript items
* SI Table S5 (SI Section S6): |dS|, x0 and w of every indicator, rows ordered
  by |dS|; x0 and w only where |dS| >= 0.1 (Methods 4.8).
* SI Section S6: the x0 values of the four leading indicators (mean
  interbuilding distance +4.7 m, adjacency +4.2 m, perimeter wall +4.9 m,
  squareness +3.0 m), their widths (about 16-18 m) and the sign change of
  average weighted distance about 6 m inside; printed at the end of the run.

The table rows are computed; the caption is the fixed SI Table S5 caption.

Inputs
  outputs/main/04_boundary_profiles/transition_metrics.csv (01_boundary_profiles.py)

Outputs (outputs/main/04_boundary_profiles/)
  table_S5_transition_metrics.tex

Run (from the repository root)
  python main/04_boundary_profiles/03_table_S5.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from favela_xai.config import load_config
from favela_xai.io import write_text

CAPTION = r"""\caption{\textbf{Boundary-transition metrics for the 25 morphological indicators.} For each indicator,
the cross-site-mean Tree-SHAP contribution $S(x)$ is taken in bins of signed distance $x$ to the
official 2013 favela boundary ($x<0$ inside, $x>0$ outside). $|\Delta S|=|S_{\mathrm{in}}-S_{\mathrm{out}}|$
is the boundary-discriminative \emph{strength} (deep-inside minus deep-outside SHAP level); $x_0$ is the
\emph{sign-flip distance} where $S$ crosses zero
($x_0>0$ = outside the legal line); $w$
is the $|S'|$-weighted transition \emph{width} (small = sharp, large = graded). $x_0$ and $w$ are reported only where a
transition is resolved ($|\Delta S|\ge0.1$); for the remaining indicators the profile is essentially flat. Rows are
ordered by $|\Delta S|$.}"""

# Indicators whose x0 and w are quoted in the SI Section S6 text
QUOTED = ["mean_interbuilding_distance", "adjacency", "perimeter_wall", "squareness", "avg_weighted_dist"]


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    min_ds = float(cfg.get("boundary_profiles.resolved_min_abs_delta_s"))
    m = pd.read_csv(cfg.require_product("transition_metrics"))
    # Rows by decreasing |dS|; ties keep the panel order of Fig. 4 (stable sort)
    m = m.sort_values("panel_order")
    m = m.assign(_key=-m["abs_delta_s"]).sort_values("_key", kind="stable").drop(columns="_key").reset_index(drop=True)

    rows = []
    for r in m.itertuples():
        shown = r.abs_delta_s >= min_ds          # resolved transition (Methods 4.8)
        x0 = f"${r.x0:+.1f}$" if shown and np.isfinite(r.x0) else "--"
        w = f"{r.w:.1f}" if shown and np.isfinite(r.w) else "--"
        rows.append(f"{r.label} & {r.dimension} & {r.abs_delta_s:.2f} & {x0} & {w} \\\\")

    tex = "\n".join([
        r"\begin{table}[H]\centering",
        CAPTION,
        r"\label{tab:spillover_metrics}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Indicator & Morphometric dimension & $|\Delta S|$ & $x_0$ (m) & $w$ (m) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path = write_text(tex, out / "table_S5_transition_metrics.tex")
    print(tex)
    print(f"[saved] {path}")

    # Values quoted in the SI Section S6 text
    print("SI Section S6 values:")
    q = m.set_index("feature")
    for f in QUOTED:
        if f in q.index:
            r = q.loc[f]
            print(f"  {r['label']:20s} x0 = {r['x0']:+.1f} m, w = {r['w']:.1f} m, |dS| = {r['abs_delta_s']:.2f}")
    lead = q.loc[[f for f in QUOTED[:4] if f in q.index], "w"]
    if len(lead):
        print(f"  widths of the four leading indicators: {lead.min():.1f}-{lead.max():.1f} m")


if __name__ == "__main__":
    main()
