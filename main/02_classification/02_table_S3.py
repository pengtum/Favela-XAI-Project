"""Step 2: per-site test-set performance (SI Table S3; Fig. 2b; Results 2.1 numbers).

Manuscript items
  * SI Table S3: per-site classification performance on the held-out test set
    (favela as positive class): accuracy, precision, recall, F1 (favela),
    F1 (macro), with the mean over the ten sites; SI Section S4: Rocinha F1
    0.980, Nova Cidade 0.788, Grotao de Costa Barros 0.852.
  * Fig. 2b: per-site held-out test accuracy and the ten-site mean (0.933)
    (drawn by 06_figure2.py from the CSV written here).
  * Results 2.1: mean accuracy 93.3%, per-site F1 0.79-0.98,
    highest accuracy Rocinha (0.977), lowest Nova Cidade (0.881) and Grotao de
    Costa Barros (0.866).

Metrics (favela_xai.model.classification_metrics): accuracy; precision, recall
and F1 with favela as the positive class; macro-averaged F1 over the two
classes; computed on the held-out 20% test buildings of each site (is_test).

Inputs
  outputs/main/02_classification/<site>_buildings_shap.gpkg   products.shap_gpkg (01_train_models.py)
  config.yaml: display_orders.table_S3 (row order of SI Table S3)

Outputs (outputs/main/02_classification/)
  table_S3_site_metrics.csv   products.site_metrics: site, name, n_test, favela_share_test,
                              accuracy, precision, recall, f1_favela, f1_macro (one row per site)
  table_S3_site_metrics.tex   tabular of SI Table S3 (rows in the printed order, mean row)
  table_S3_summary.txt        the numbers quoted in Results 2.1 and SI Section S4

Run from the repository root:
  python main/02_classification/02_table_S3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyogrio

from favela_xai.config import load_config
from favela_xai.io import write_table, write_text
from favela_xai.labels import label_to_int
from favela_xai.model import classification_metrics

METRICS = ["accuracy", "precision", "recall", "f1_favela", "f1_macro"]
HEADER = r"Site & Accuracy & Precision & Recall & $F_{1}$ (favela) & $F_{1}$ (macro) \\"


def tex_number(v: float) -> str:
    """0.977 -> 0.977 (three decimals, as in the SI)."""
    return f"{v:.3f}"


def site_metrics(cfg, site: str) -> dict:
    path = cfg.require_product("shap_gpkg", site=site)
    df = pyogrio.read_dataframe(path, columns=["label", "predicted_label", "is_test"], read_geometry=False)
    test = df["is_test"].astype(bool).to_numpy()
    y = label_to_int(df["label"])[test]
    pred = label_to_int(df["predicted_label"])[test]
    return dict(site=site, name=cfg.site_name(site), **classification_metrics(y, pred))


def main() -> None:
    cfg = load_config()
    out_dir = cfg.output_dir(__file__)
    table = pd.DataFrame([site_metrics(cfg, s) for s in cfg.sites])
    write_table(table, cfg.product("site_metrics"))

    # SI Table S3 as printed: rows in the manuscript order, three decimals, mean row
    t = table.set_index("site")
    order = cfg.display_order("table_S3")
    width = max(len(cfg.site_latex(s)) for s in order) + 2
    lines = [r"\begin{tabular}{lccccc}", r"\toprule", HEADER, r"\midrule"]
    for s in order:
        lines.append(f"{cfg.site_latex(s):<{width}}& " + " & ".join(tex_number(t.loc[s, m]) for m in METRICS) + r" \\")
    lines.append(r"\midrule")
    lines.append(f"{'Mean':<{width}}& " + " & ".join(tex_number(t[m].mean()) for m in METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_text("\n".join(lines) + "\n", out_dir / "table_S3_site_metrics.tex")

    # Summary numbers (Results 2.1, SI Section S4)
    acc = t["accuracy"].sort_values(ascending=False)
    f1 = t["f1_favela"]
    summary = [
        f"mean test accuracy over {len(t)} sites: {t['accuracy'].mean():.3f} "
        f"(Results 2.1 '93.3%'; Main Fig. 2b 'mean 0.933')",
        f"per-site F1 (favela): {f1.min():.2f}-{f1.max():.2f} (Results 2.1 '0.79-0.98')",
        f"highest accuracy: {cfg.site_name(acc.index[0])} {acc.iloc[0]:.3f} (Results 2.1: Rocinha 0.977)",
        "lowest accuracies: " + ", ".join(f"{cfg.site_name(s)} {acc[s]:.3f}" for s in acc.index[-2:][::-1])
        + " (Results 2.1: Nova Cidade 0.881, Grotao de Costa Barros 0.866)",
        "F1 (favela) of the SI S4 text: " + ", ".join(
            f"{cfg.site_name(s)} {f1[s]:.3f}" for s in ("rocinha", "nova_cidade", "grotao_de_costa_barros") if s in f1)
        + " (SI S4: 0.980, 0.788, 0.852)",
        "means of the other columns: " + ", ".join(f"{m} {t[m].mean():.3f}" for m in METRICS[1:])
        + " (SI Table S3 mean row: 0.896, 0.910, 0.903, 0.923)",
        f"test buildings: {int(t['n_test'].sum()):,} (favela share per site "
        f"{t['favela_share_test'].min():.2f}-{t['favela_share_test'].max():.2f})",
    ]
    write_text("\n".join(summary) + "\n", out_dir / "table_S3_summary.txt")
    print("\n".join(summary))
    print(t[METRICS].round(3).to_string())
    if not np.isfinite(t[METRICS].to_numpy()).all():
        print("warning: some metrics are undefined (a site without test buildings of one class)")


if __name__ == "__main__":
    main()
