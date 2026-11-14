# appendix/S07_pca: within-site reading of the attribution PC1

Thresholds the first component (PC1) of the pooled, standardised PCA of the
Tree-SHAP attributions (Fig. 5, `main/05_attribution_plane`) within each site
at its ROC-Youden optimum, with the official 2013 favela labels as reference,
and maps the buildings above the threshold ("favela-like").

## Manuscript items

| Item | Script |
|---|---|
| SI Section S7, second paragraph: per-site ROC-Youden thresholds of PC1; the threshold ordering follows the official polygon at most sites, the weakest agreement being Nova Cidade; favela-like buildings are also found beyond the polygon | 01 |
| SI Fig. S10: favela-like buildings in seven representative sites | 02 |
| Results 2.4: buildings immediately outside the polygon often share the PC1 range of their favela neighbours (maps of SI Fig. S10) | 01 (per-site thresholds and favela-like flags), 02 (maps) |

Computed elsewhere: the PCA itself (Methods 4.6, Fig. 5) in
`main/05_attribution_plane/01_attribution_pca.py`; the raw-indicator benchmark
of the first paragraph of SI S7 (SI Table S6) in
`appendix/S09_block_complexity_k/03_persite_protocol.py`.

## Run order

From the repository root, after `main/02_classification` and
`main/05_attribution_plane/01_attribution_pca.py`:

```bash
python appendix/S07_pca/01_pc1_youden_thresholds.py   # per-site thresholds, favela-like flag per building
python appendix/S07_pca/02_figure_pca_mapping.py      # SI Fig. S10 maps
```

## Inputs

- `products.attribution_pca_scores` (`main/05_attribution_plane`): `site`, `row`, `uID`, `is_favela`, `PC1` per building.
- `products.shap_gpkg` (`main/02_classification`): building geometry, in the row order of the PCA scores.
- Official 2013 favela polygons (`paths.raw.favelas_2013`; source in `data/README.md`).

## Outputs (`outputs/appendix/S07_pca/`)

| File | Content | Script |
|---|---|---|
| `pc1_youden_thresholds.csv` | per site: threshold, Youden J, TPR, FPR, AUC of PC1, class sizes and means, share of favela-like buildings inside and outside the 2013 polygons, whether the site is mapped | 01 |
| `pc1_favela_like.parquet` | per building: `site`, `row`, `uID`, `is_favela`, `PC1`, `favela_like` | 01 |
| `Figure_R1_PCA_mapping_<site>.pdf` | map of one site (panels of SI Fig. S10) | 02 |
| `Figure_R1_PCA_mapping.pdf` | all mapped sites in one panel grid with site names and a legend | 02 |

The overview map of Rio de Janeiro in SI Fig. S10 is not drawn by these
scripts.

## Settings (`config.yaml`: `pca`, `appendix_S07_pca`)

- PC1: first component of one PCA of the 25 attribution columns of all ten sites, each column standardised over the pooled buildings; oriented so that favela buildings score higher (Methods 4.6).
- Threshold: per site, the ROC point of PC1 (favela = positive class) that maximises Youden's J = TPR − FPR (`favela_xai.attribution.youden_threshold`).
- `favela_like_rule`: `>` (PC1 above the threshold) | `>=` (at or above).
- Mapped sites (`map_sites`): Fazenda Coqueiro, Faz Quem Quer, Morro da Guaíba, Nova Cidade, Barão, Catumbi, Rocinha.
- Maps: favela-like #FF7F0E, other buildings #8B959A, 2013 outlines black; extent = building extent + 2%; buildings rasterised at 600 dpi, outlines vector.
