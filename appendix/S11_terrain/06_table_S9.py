"""SI Table S9 (terrain ablation accuracies) and the in-text numbers of SI S11.

Manuscript items
* SI Table S9: per-site test-set accuracy under the 25 morphological
  indicators (Morph), the nine terrain features (Terrain) and their union
  (Combined), with the means (tabular environment only).
* In-text numbers of SI S11 'Experiment 2' (terrain-only mean accuracy, drop
  against the morphological classifier, sites with the largest drops) and
  'Experiment 3' (number of sites improved by the combined model, range of
  the gains, sites where accuracy falls), and of the Results paragraph on
  terrain.

Inputs
* outputs/appendix/S11_terrain/summary_three_model_comparison.csv (02).
* products.site_metrics (main/02_classification), optional: SI Table S3 mean
  accuracy, compared with the terrain-only mean.

Outputs (outputs/appendix/S11_terrain/)
* ``table_S9_terrain_only_tabular.tex``: rows in the order of SI Table S3,
  values with three decimals (0.979).
* ``S11_in_text_numbers.txt``: the numbers quoted in SI S11 and the Results.

Run from the repository root:
    python appendix/S11_terrain/06_table_S9.py
"""
from __future__ import annotations

import pandas as pd

from favela_xai.config import ConfigError, load_config
from favela_xai.io import write_text

import terrain_common as tc

COLUMNS = [("morph_accuracy", "Morph"), ("terrain_accuracy", "Terrain"), ("combined_accuracy", "Combined")]


def dec(x: float, digits: int = 3) -> str:
    """Number with three decimals, as in the SI (0.979)."""
    return f"{x:.{digits}f}"


def tabular(summary: pd.DataFrame, order: list[str], cfg) -> str:
    """LaTeX tabular of SI Table S9 (booktabs)."""
    names = {s: cfg.site_latex(s) for s in order}
    width = max(len(n) for n in list(names.values()) + ["Mean"])
    lines = [r"\begin{tabular}{lccc}", r"\toprule", "Site & Morph & Terrain & Combined \\\\", r"\midrule"]
    for site in order:
        r = summary.loc[site]
        lines.append(f"{names[site]:<{width}} & " + " & ".join(dec(r[c]) for c, _ in COLUMNS) + r" \\")
    lines.append(r"\midrule")
    means = summary.loc[order, [c for c, _ in COLUMNS]].mean()
    lines.append(f"{'Mean':<{width}} & " + " & ".join(dec(means[c]) for c, _ in COLUMNS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def in_text_numbers(summary: pd.DataFrame, order: list[str], cfg) -> str:
    """Numbers quoted in SI S11 and in the Results paragraph on terrain.

    Per-site drops and gains are given twice: from the unrounded accuracies and
    from the three-decimal values of SI Table S9.
    """
    s = summary.loc[order].copy()
    s["name"] = [cfg.site_name(x) for x in s.index]
    acc_cols = [c for c, _ in COLUMNS]
    r3 = s[acc_cols].round(3)                                   # three-decimal values of Table S9
    s["drop_pp"] = (s["morph_accuracy"] - s["terrain_accuracy"]) * 100
    s["gain_pp"] = (s["combined_accuracy"] - s["morph_accuracy"]) * 100
    s["drop_pp_table"] = (r3["morph_accuracy"] - r3["terrain_accuracy"]) * 100
    s["gain_pp_table"] = (r3["combined_accuracy"] - r3["morph_accuracy"]) * 100
    m = s[acc_cols].mean()

    out = ["SI Section S11: numbers quoted in the text (accuracies on the held-out test split)", ""]
    out.append(f"Mean accuracy over {len(s)} sites: Morph {m['morph_accuracy']:.3f}, "
               f"Terrain {m['terrain_accuracy']:.3f} ({m['terrain_accuracy'] * 100:.0f}%), "
               f"Combined {m['combined_accuracy']:.3f}")
    out.append(f"Mean drop Morph - Terrain: {(m['morph_accuracy'] - m['terrain_accuracy']) * 100:.1f} points")
    try:
        s3 = tc.table_s3_accuracy(cfg).reindex(order)
        out.append(f"SI Table S3 mean accuracy: {s3.mean():.3f}; drop to the terrain-only mean: "
                   f"{(s3.mean() - m['terrain_accuracy']) * 100:.1f} points; "
                   f"largest |Morph - Table S3| difference: {(s['morph_accuracy'] - s3).abs().max():.4f}")
    except ConfigError as exc:
        out.append(f"SI Table S3 not available ({exc.__class__.__name__}); run main/02_classification")
    out.append("")
    out.append("Drop Morph - Terrain by site (points), largest first [unrounded | from the Table S9 values]:")
    for name, v, vt in s.sort_values("drop_pp", ascending=False)[["name", "drop_pp", "drop_pp_table"]].itertuples(index=False):
        out.append(f"  {name:<24s} {v:5.1f} | {vt:5.1f}")
    out.append("")
    improved = s[s["gain_pp"] > 0]
    worse = s[s["gain_pp"] <= 0]
    out.append(f"Combined better than Morph in {len(improved)} of {len(s)} sites; gains "
               f"{improved['gain_pp'].min():.1f} to {improved['gain_pp'].max():.1f} points (unrounded), "
               f"{improved['gain_pp_table'].min():.1f} to {improved['gain_pp_table'].max():.1f} points "
               "(from the Table S9 values)" if len(improved)
               else "Combined better than Morph in no site")
    for name, v, vt in worse[["name", "gain_pp", "gain_pp_table"]].itertuples(index=False):
        out.append(f"  not improved: {name} ({v:+.1f} points; {vt:+.1f} from the Table S9 values)")
    out.append("Gain Combined - Morph by site (points), largest first [unrounded | from the Table S9 values]:")
    for name, v, vt in s.sort_values("gain_pp", ascending=False)[["name", "gain_pp", "gain_pp_table"]].itertuples(index=False):
        out.append(f"  {name:<24s} {v:+5.1f} | {vt:+5.1f}")
    return "\n".join(out) + "\n"


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    summary = pd.read_csv(tc.require(out / tc.THREE_MODEL_CSV, "02_terrain_ablation.py")).set_index("site")
    order = cfg.display_order("table_S3")             # SI Table S9 lists the sites in the order of Table S3

    tex = tabular(summary, order, cfg)
    write_text(tex, out / tc.TABLE_S9_TEX)
    numbers = in_text_numbers(summary, order, cfg)
    write_text(numbers, out / tc.IN_TEXT_NUMBERS)
    print(tex)
    print(numbers)
    print(f"  written {out / tc.TABLE_S9_TEX} and {out / tc.IN_TEXT_NUMBERS}")


if __name__ == "__main__":
    main()
