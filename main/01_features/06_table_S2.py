"""Step 6: SI Table S2 and the indicator registry.

Manuscript items
  * SI Table S2: full name, figure label and morphometric dimension of the 25
    indicators, in the order of the definitions of SI Section S3.
  * Fig. 1, Methods 4.3 and SI Table S2: dimension 3, shape 13, spatial
    distribution 4, configuration 3, connectivity 2 (checked against the registry).

The registry (favela_xai.features.FEATURES) defines the indicator names,
figure labels and dimensions used by every figure script. No input data needed.

Outputs (outputs/main/01_features/)
  table_S2_indicators.tex   tabular of SI Table S2 (header and 25 rows)
  indicator_registry.csv    one row per indicator: column name, full name, figure label,
                            dimension, computed on, computation (momepy call or geometry
                            operation) and SI S3 formula

Run from the repository root:
  python main/01_features/06_table_S2.py
"""
from __future__ import annotations

import pandas as pd

from favela_xai.config import load_config
from favela_xai.features import DIMENSIONS, FEATURES, check_registry, features_by_dimension, table_s2_body
from favela_xai.io import write_table, write_text


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    sizes = cfg.get("indicators.dimension_sizes")
    check_registry(sizes)

    # SI Table S2: header and rows in the order of SI S3
    lines = [r"\begin{tabular}{ll@{\hspace{1.5em}}l}", r"\toprule",
             r"Indicator (full name) & Figure label & Dimension \\", r"\midrule"]
    lines += table_s2_body().rstrip("\n").split("\n")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_text("\n".join(lines) + "\n", out_dir / "table_S2_indicators.tex")

    # Full registry of the 25 indicators
    registry = pd.DataFrame([dict(indicator=f.key, full_name=f.full_name, figure_label=f.label,
                                  dimension=f.dimension, computed_on=f.computed_on, computed_as=f.method,
                                  si_s3_formula=f.si_formula) for f in FEATURES])
    write_table(registry, out_dir / "indicator_registry.csv")

    counts = {d: len(v) for d, v in features_by_dimension().items()}
    print("indicators per dimension: " + ", ".join(f"{d.lower()} {counts[d]}" for d in DIMENSIONS)
          + " (Fig. 1 caption: dimension 3, shape 13, spatial distribution 4, configuration 3, connectivity 2)")
    print(f"written {out_dir / 'table_S2_indicators.tex'} and {out_dir / 'indicator_registry.csv'}")


if __name__ == "__main__":
    main()
