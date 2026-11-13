# appendix/S10_boundaries_2013_2019: change in the official favela boundaries, 2013-2019

Overlays the official favela polygons of 2013 and of 2019 on the same 2013
building footprints and counts, per site, the buildings that are favela in both
years, formal in both, added to the favela boundaries by 2019, or removed from
them. Only the boundaries differ between the two overlays; no statistical test
is run. Main result (SI S10): the changes are small (a few hundred buildings
added or removed, several sites without any change), so the favela labels of
the ten sites are stable over the six years.

## Manuscript items

| Item | Script |
|---|---|
| SI Section S10 "Change in the official favela boundaries, 2013-2019": buildings added, removed and unchanged; sites without change | 01 |
| SI Table S8: building counts by status in the 2013 and 2019 boundaries | 01 |
| Methods 4.12 "Overlay of the 2013 and 2019 favela boundaries" and the sentence on the 2019 overlay in Methods 4.1 | 01 |

## Run order

From the repository root, after `main/01_features`:

```bash
python appendix/S10_boundaries_2013_2019/01_boundary_overlay.py   # SI Table S8 and the SI S10 counts
```

## Inputs

- Per-site building tables of `main/01_features` (`products.features_gpkg`): the 2013 footprints of the analysed buildings and their label.
- Official favela polygons 2013 and 2019 (*Limite Favelas*, Data.Rio; the 2019 polygons were accessed in September 2024): `paths.raw.favelas_2013`, `paths.raw.favelas_2019`.

The paper's classification analyses retain the labels assigned when the study
data were prepared. The recomputed 2013 overlay in Table S8 differs from those
labels for 15 buildings (a net increase of five favela buildings). A fresh run
of the released pipeline assigns labels using the configured geometric rule.
With the default `footprint_intersects` rule, labels and the 2013 overlay are
expected to agree when computed from the same footprints and boundaries. The
15-building discrepancy describes the retained study labels; it is not an
expected discrepancy for a fresh rebuild.

## Outputs (`outputs/appendix/S10_boundaries_2013_2019/`)

| File | Content |
|---|---|
| `overlay_counts.csv` | per site: buildings, the four status counts, buildings whose 2013 overlay status differs from their label, and the same counts under a centroid rule for comparison |
| `table_S8_boundary_status_counts.csv` | SI Table S8: rows in the printed order, sum row |
| `table_R1_cadastre_counts.tex` | SI Table S8 in LaTeX |
| `overlay_log.txt` | sums, buildings with the same status in both years, sites without change |

## Definitions and settings (`config.yaml`: `boundary_overlay`, `display_orders.table_S8`, `crs`)

- Inside rule: a building is inside in a given year when its footprint intersects the dissolved favela polygons of that year (`rule: footprint_intersects`, `dissolve_polygons: true`); invalid polygons are repaired with `make_valid` before the union.
- Footprints: the same 2013 building set of the ten sites in both overlays (EPSG:31983).
- Status groups: favela in both; formal in both; added by 2019 (outside 2013, inside 2019); removed by 2019 (inside 2013, outside 2019).
- Centroid rule: computed for comparison only; it does not enter SI Table S8.
