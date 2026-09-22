"""Step 4: study favelas versus all other 2013 favela polygons (SI S2.1 numbers; data of SI Fig. S1).

Manuscript items (SI Section S2.1, 'Site selection, analysis windows and site context')
* Share of the 1,076 official 2013 polygons and of the mapped favela area
  covered by the study favelas -> study_vs_city.csv, section 'coverage'
  (n_polygons_study, share_polygons, share_area_full_polygons).
* Footprint density and distance to the centre, study versus other favelas,
  counted by polygon and weighted by polygon area -> section 'distribution'
  (study_median, other_median of bldg_density_per_ha and dist_centre_km) and
  section 'area_weighted' (area-weighted median and mean).
* Supplementary Fig. S1b-e: two-sample Kolmogorov-Smirnov D and p of study
  versus other favelas -> section 'distribution' (ks_D_study_vs_other,
  ks_p_study_vs_other).

Inputs (outputs of steps 02 and 03 of this folder)
* footprints_2013_centroids.parquet, polygons_base.parquet, study_membership.csv,
  site_hulls.parquet; products.features_gpkg for the building count of each site.

Outputs (outputs/appendix/S02_site_selection/)
* city_polygons.csv: every polygon with its descriptors, study flag, site and
  footprint counts (input of SI Fig. S1).
* study_polygons.csv: the study polygons with their percentile among all polygons.
* study_vs_city.csv: distributions, KS and Mann-Whitney tests, quintile
  coverage, area-weighted statistics and coverage shares.
* study_by_site.csv: per-site summary of the study polygons.
* footprint_count_check.csv: analysed buildings of each site versus the 2013
  footprints of any type (and of types A101/A102) inside the site's building hull.
* name_reconciliation_SI_S2.csv: favela names listed in SI S2 versus the
  names of the study polygons (log only).
* key_numbers_S2.txt: the SI S2.1 numbers.

Settings
* ``site_context.density_all_tipo`` (all footprint types, SI Fig. S1e);
  ``site_context.comparison_test`` (ks_2samp);
  ``appendix_S02_site_selection.area_weighted_statistic`` (median | mean): the
  area-weighted statistic listed first in key_numbers_S2.txt (both are written);
  ``appendix_S02_site_selection.footprint_building_tipos`` (footprint types of the secondary A101/A102 density).

Run from the repository root:
    python appendix/S02_site_selection/04_compare_study_vs_city.py
"""
from __future__ import annotations

import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import stats

from favela_xai.config import load_config
from favela_xai.io import write_table, write_text

import s02_lib as L

# Favela names listed for each site in SI Section S2 (cluster composition),
# used only for the name check written to name_reconciliation_SI_S2.csv.
SI_S2_NAMES = {
    "grotao_de_costa_barros": ["Associação Comunitária Vila Alvorada", "Cabo Otávio", "Caminho do Job",
                               "Caminho do Padre", "Chico Mendes (Morro do Chapadão)", "Deus é Vida",
                               "Faz Quem Quer (RA - Anchieta)", "Final Feliz", "Grotão de Costa Barros",
                               "Monte Sinai", "Nova Conquista (RA - Pavuna)", "Oito de Dezembro",
                               "Parque Boa Esperança (RA - Pavuna)", "Parque Criança Esperança",
                               "Parque Raio de Sol", "Parque Tiradentes", "Quitanda", "Terra Nostra",
                               "Vila Santa Teresa", "Village Costa Barros"],
    "faz_quem_quer": ["Faz Quem Quer (RA - Madureira)", "Morro do Sapê"],
    "morro_da_guaiba": ["Rua Castro Menezes, nº 928", "Mangueirinha", "Morro da Guaíba", "Vila Pequirí"],
    "rocinha": ["Matinha (RA - Rocinha)", "Do Horto", "Vidigal", "Rocinha", "Chácara do Céu",
                "Vila Parque da Cidade"],
    "palmeras": ["Palmeiras", "Comunidade Luz Divina"],
    "catumbi": ["Rua Coqueiros, nº 10", "Vila Santa Bárbara", "A.M. e Amigos do Vale",
                "A.M. e Amigos de Santa Tereza", "Comunidade de Clara Nunes", "Vila Paraíso", "Azevedo Lima",
                "São Carlos", "Catumbi", "Morro Santos Rodrigues", "Rato", "Ocidental Fallet",
                "Travessa Vista Alegre", "Baronesa", "Luiz Marcelino", "Vila Anchieta", "Morro da Coroa",
                "Unidos de Santa Tereza"],
    "jacarezinho": ["Antiga Company", "Engenheiro Alberto Moas, próxima ao nº 75",
                    "Linha Amarela (acesso à Av. Automóvel Clube)", "B-1", "Portelinha", "Vila União (RA - Méier)",
                    "Vila Padre Angelo", "Vila Angela", "Comunidade do Pedrosa", "Vitória de Manguinhos",
                    "Rua Lagoa Redonda", "Mandela de Pedra", "Vila da Rua Viúva Cláudio, nº 211", "Malvinas",
                    "Comunidade Nova Jerusalém", "Céu Azul", "Vila Arará", "Parque João Goulart",
                    "Parque Oswaldo Cruz", "Parque Carlos Chagas", "Parque Félix Ferreira",
                    "Chácara de Del Castilho", "Chp-2", "Vila Turismo", "Dois de Maio", "Jacarezinho", "Marlene",
                    "Tautá", "Parque União de Del Castilho", "Pica Pau Amarelo", "Praça Marimbá, 60 (Fundos)",
                    "Rua Matinoré, nº 163 (Fundos)", "Vila União (RA - São Cristóvão)", "Rua São João",
                    "Vila Maria", "Tancredo Neves (RA - Jacarezinho)", "Vila Matinoré", "Vila Jandira",
                    "Carlos Drumond de Andrade", "Vila Vitória (RA - São Cristóvão)"],
    "fazena_coqueiro": ["Comunidade da Favelinha", "Rua Anes Dias", "Jardim Clarice", "Rua Congo, nº 147",
                        "Coréia (RA - Bangu)", "Fazenda Coqueiro", "Jacaré"],
    "barao": ["Amigos da Aerobita", "Barão", "Travessa Antonina"],
    "nova_cidade": ["Estrada de Inhoaíba (próxima Lt 02 - Qd 32)", "Nova Cidade", "Parque Proletário Vila Esperança",
                    "Bairro Nova Aguiar", "Vila União (RA - Campo Grande)"],
}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of values weighted by weights (lower weighted median)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def main() -> None:
    cfg = load_config()
    out = cfg.output_dir(__file__)
    if not cfg.get("site_context.density_all_tipo"):
        raise ValueError("SI Fig. S1e counts all footprint types (site_context.density_all_tipo: true)")
    if cfg.get("site_context.comparison_test") != "ks_2samp":
        raise ValueError("SI Fig. S1 reports the two-sample Kolmogorov-Smirnov test (site_context.comparison_test)")

    pol = L.read_polygons(out, cfg)
    mem = pd.read_csv(L.require(out / L.MEMBERSHIP_CSV, "03_polygons_and_membership.py"))
    hulls = gpd.read_parquet(L.require(out / L.SITE_HULLS_PARQUET, "03_polygons_and_membership.py"))
    fp = pd.read_parquet(L.require(out / L.FOOTPRINTS_PARQUET, "02_footprints_2013.py"))

    pol["study"] = False
    pol.loc[mem.poly_index.unique(), "study"] = True
    first = mem.drop_duplicates("poly_index").set_index("poly_index")      # a polygon in two sites keeps the first
    for c in ["site", "site_name", "n_fav_buildings_in_site", "frac_polygon_area_in_site_hull"]:
        pol[c] = first[c].reindex(pol.index)

    # ---------------- footprints per polygon (centroid within polygon, all Tipo classes) ----------------
    tree = shapely.STRtree(np.asarray(pol.geometry.values, dtype=object))
    p_idx, g_idx = tree.query(shapely.points(fp.x.to_numpy(), fp.y.to_numpy()), predicate="within")
    first_hit = pd.Series(g_idx, index=p_idx)
    first_hit = first_hit[~first_hit.index.duplicated(keep="first")]
    fp["poly"] = -1
    fp.loc[first_hit.index.to_numpy(), "poly"] = pol.index.to_numpy()[first_hit.to_numpy()]
    fp["is_bld"] = fp.tipo.isin(cfg.get("appendix_S02_site_selection.footprint_building_tipos"))
    inside = fp[fp.poly >= 0]
    g_all, g_bld = inside.groupby("poly"), inside[inside.is_bld].groupby("poly")
    pol["n_buildings"] = g_all.size().reindex(pol.index).fillna(0).astype(int)
    pol["bldg_density_per_ha"] = pol.n_buildings / pol.area_ha                         # SI Fig. S1e
    pol["median_fp_area_m2"] = g_all.area_geom.median().reindex(pol.index)
    pol["n_buildings_A101_A102"] = g_bld.size().reindex(pol.index).fillna(0).astype(int)
    pol["bldg_density_A101_A102_per_ha"] = pol.n_buildings_A101_A102 / pol.area_ha    # secondary density (A101/A102 only)
    print(f"footprints inside a favela polygon: {len(inside):,} of {len(fp):,} "
          f"({int(len(p_idx) - len(first_hit))} centroids in two overlapping polygons, first kept)")

    # footprints of the 2013 layer inside each site's building hull versus the analysed buildings
    chk = []
    for _, h in hulls.iterrows():
        b = h.geometry.bounds
        sub = fp[(fp.x >= b[0]) & (fp.x <= b[2]) & (fp.y >= b[1]) & (fp.y <= b[3])]
        sub = sub[shapely.contains_xy(h.geometry, sub.x.to_numpy(), sub.y.to_numpy())]
        chk.append(dict(site=h.site, n_analysed=len(L.site_buildings(cfg, h.site)), n_2013_all_in_hull=len(sub),
                        n_2013_A101_A102_in_hull=int(sub.is_bld.sum())))
    write_table(pd.DataFrame(chk), out / "footprint_count_check.csv")
    print(pd.DataFrame(chk).to_string(index=False))

    VARS = ["area_ha", "perimeter_m", "compactness", "dist_centre_km", "n_buildings", "bldg_density_per_ha",
            "median_fp_area_m2", "bldg_density_A101_A102_per_ha"]
    for v in VARS:
        allv = pol[v].dropna().to_numpy()
        pol[f"pctl_{v}"] = [stats.percentileofscore(allv, x, kind="mean") if np.isfinite(x) else np.nan
                            for x in pol[v].to_numpy()]
        pol.loc[~pol.study, f"pctl_{v}"] = np.nan

    city_cols = ["poly_index", "objectid", "cod_favela", "nome", "study", "site", "area_ha", "perimeter_m",
                 "compactness", "n_parts", "cx", "cy", "dist_centre_km", "n_buildings", "bldg_density_per_ha",
                 "median_fp_area_m2", "n_buildings_A101_A102", "bldg_density_A101_A102_per_ha"]
    write_table(pol[[c for c in city_cols if c in pol.columns]], out / L.CITY_POLYGONS_CSV)
    st, oth = pol[pol.study].copy(), pol[~pol.study]
    st_cols = ["poly_index", "cod_favela", "nome", "site", "site_name", "n_fav_buildings_in_site",
               "frac_polygon_area_in_site_hull"] + VARS + [f"pctl_{v}" for v in VARS]
    write_table(st.sort_values(["site", "area_ha"], ascending=[True, False])[st_cols], out / "study_polygons.csv")

    # ---------------- distributions and tests ----------------
    def q(a, p):
        return float(np.nanquantile(a, p)) if len(a) else np.nan

    rows = []
    for v in VARS:
        a, b, c = st[v].dropna().to_numpy(), oth[v].dropna().to_numpy(), pol[v].dropna().to_numpy()
        ks = stats.ks_2samp(a, b)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(section="distribution", variable=v, n_study=len(a), n_other=len(b), n_city=len(c),
                         study_median=q(a, .5), study_q1=q(a, .25), study_q3=q(a, .75),
                         other_median=q(b, .5), other_q1=q(b, .25), other_q3=q(b, .75),
                         city_median=q(c, .5), city_q1=q(c, .25), city_q3=q(c, .75),
                         ks_D_study_vs_other=ks.statistic, ks_p_study_vs_other=ks.pvalue,
                         mannwhitney_p_study_vs_other=mw.pvalue,
                         study_pctl_min=np.nanmin(st[f"pctl_{v}"]), study_pctl_median=np.nanmedian(st[f"pctl_{v}"]),
                         study_pctl_max=np.nanmax(st[f"pctl_{v}"])))
    # quintile coverage (descriptive)
    qrows = []
    for v in VARS:
        edges = np.nanquantile(pol[v], [0, .2, .4, .6, .8, 1])
        qi = np.clip(np.searchsorted(edges[1:-1], pol[v].to_numpy(), side="right"), 0, 4)
        qi = np.where(np.isfinite(pol[v].to_numpy()), qi, -1)
        for k in range(5):
            m = qi == k
            qrows.append(dict(section="quintile_coverage", variable=v, quintile=k + 1, q_lo=edges[k], q_hi=edges[k + 1],
                              n_city=int(m.sum()), n_study=int((m & pol.study.to_numpy()).sum()),
                              share_of_quintile_in_study=float((m & pol.study.to_numpy()).sum() / max(m.sum(), 1)),
                              share_of_study_in_quintile=float((m & pol.study.to_numpy()).sum() / pol.study.sum())))
    # area-weighted statistics: each polygon weighted by its area (SI S2.1)
    wrows = []
    for v in VARS:
        sv, ov = st.dropna(subset=[v]), oth.dropna(subset=[v])
        wrows.append(dict(section="area_weighted", variable=v,
                          study_area_weighted_mean=np.average(sv[v], weights=sv.area_ha),
                          other_area_weighted_mean=np.average(ov[v], weights=ov.area_ha),
                          study_area_weighted_median=weighted_median(sv[v].to_numpy(), sv.area_ha.to_numpy()),
                          other_area_weighted_median=weighted_median(ov[v].to_numpy(), ov.area_ha.to_numpy())))
    # coverage of the city's favelas
    union_hull = shapely.union_all(np.asarray(hulls.geometry.values, dtype=object))
    clipped = shapely.intersection(np.asarray(pol.geometry.values, dtype=object), union_hull)
    cov = dict(section="coverage", n_polygons_city=len(pol), n_polygons_study=int(pol.study.sum()),
               share_polygons=float(pol.study.mean()), area_ha_city=pol.area_ha.sum(),
               area_ha_study_full_polygons=st.area_ha.sum(),
               share_area_full_polygons=st.area_ha.sum() / pol.area_ha.sum(),
               area_ha_favela_within_site_hulls=shapely.area(clipped).sum() / 1e4,
               share_area_within_site_hulls=shapely.area(clipped).sum() / 1e4 / pol.area_ha.sum(),
               n_fav_labelled_buildings_study=int(mem.n_fav_buildings_in_site.sum()),
               n_footprints_all_types_in_all_polygons=int(pol.n_buildings.sum()),
               n_footprints_all_types_in_study_polygons=int(st.n_buildings.sum()),
               share_footprints_in_study_polygons=st.n_buildings.sum() / pol.n_buildings.sum())
    persite = st.groupby("site").agg(
        n_polygons=("nome", "size"), area_ha_sum=("area_ha", "sum"), area_ha_median=("area_ha", "median"),
        compactness_median=("compactness", "median"), dist_centre_km_median=("dist_centre_km", "median"),
        n_fav_labelled_buildings=("n_fav_buildings_in_site", "sum"),
        n_polygons_partly_outside_hull=("frac_polygon_area_in_site_hull", lambda x: int((x < 0.999).sum())),
        n_footprints_sum=("n_buildings", "sum"), bldg_density_median=("bldg_density_per_ha", "median"))
    write_table(persite.reset_index(), out / "study_by_site.csv")
    comp = pd.concat([pd.DataFrame(rows), pd.DataFrame(qrows), pd.DataFrame(wrows), pd.DataFrame([cov])],
                     ignore_index=True)
    write_table(comp, out / "study_vs_city.csv")

    # ---------------- SI S2 name reconciliation (log only) ----------------
    rec = []
    in_layer = {norm_name(n) for n in pol.nome.astype(str)}
    for site in cfg.sites:
        got = {norm_name(n): n for n in mem[mem.site == site].nome.astype(str)}
        listed = {norm_name(n): n for n in SI_S2_NAMES.get(site, [])}
        for k in sorted(set(got) | set(listed)):
            rec.append(dict(site=site, name=got.get(k, listed.get(k)), in_SI_S2=k in listed,
                            in_polygon_derived_set=k in got, name_exists_in_2013_polygons=k in in_layer))
    rec = pd.DataFrame(rec)
    write_table(rec, out / "name_reconciliation_SI_S2.csv")

    # ---------------- key numbers of SI S2.1 ----------------
    D = pd.DataFrame(rows).set_index("variable")
    W = pd.DataFrame(wrows).set_index("variable")
    wstat = cfg.get("appendix_S02_site_selection.area_weighted_statistic")
    if wstat not in ("median", "mean"):
        raise ValueError("appendix_S02_site_selection.area_weighted_statistic must be 'median' or 'mean'")
    wother = "mean" if wstat == "median" else "median"
    lines = [
        f"study polygons {cov['n_polygons_study']} of {cov['n_polygons_city']:,} "
        f"({100 * cov['share_polygons']:.1f}%); other polygons {cov['n_polygons_city'] - cov['n_polygons_study']}",
        f"share of the city's mapped favela area (full study polygons) {100 * cov['share_area_full_polygons']:.1f}%",
        f"footprints per ha, median study {D.loc['bldg_density_per_ha', 'study_median']:.1f} versus other "
        f"{D.loc['bldg_density_per_ha', 'other_median']:.1f}",
        f"distance to Praca XV, median study {D.loc['dist_centre_km', 'study_median']:.1f} versus other "
        f"{D.loc['dist_centre_km', 'other_median']:.1f} km",
        f"distance to Praca XV weighted by polygon area ({wstat}, as reported): study "
        f"{W.loc['dist_centre_km', f'study_area_weighted_{wstat}']:.1f} versus other "
        f"{W.loc['dist_centre_km', f'other_area_weighted_{wstat}']:.1f} km "
        f"({wother}: {W.loc['dist_centre_km', f'study_area_weighted_{wother}']:.1f} versus "
        f"{W.loc['dist_centre_km', f'other_area_weighted_{wother}']:.1f} km)",
        "KS study vs other (SI Fig. S1 b-e): " + "; ".join(
            f"{v} D = {D.loc[v, 'ks_D_study_vs_other']:.2f}, p = {D.loc[v, 'ks_p_study_vs_other']:.2g}"
            for v in ["area_ha", "compactness", "dist_centre_km", "bldg_density_per_ha"]),
        f"SI S2 names matched to study polygons: {int((rec.in_SI_S2 & rec.in_polygon_derived_set).sum())} "
        f"of {int(rec.in_SI_S2.sum())}",
    ]
    write_text("\n".join(lines) + "\n", out / "key_numbers_S2.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
