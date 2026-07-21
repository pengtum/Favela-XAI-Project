"""
Per-site deep-feature importance rankings, FAVELA buildings only
(standalone replacement for main-text Figure 2c). For each site, the top-15
indicators by mean |SHAP| computed over favela buildings, as horizontal
bars coloured by feature (same canonical palette as the other figures, so
the same feature has the same colour across sites). Exports PDF + SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES = os.path.join(PROJ,"results","all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
RAW = ["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
       "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
       "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
       "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH = ["shap_"+f for f in RAW]
PRETTY = {'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','shared_walls':'shared walls','avg_weighted_dist':'avg wt. dist.','num_neighbors':'no. neighbours','area':'area','CAR':'cov-area ratio','tes_area':'tess. cell area','cwa':'compact. axis (t)','compact_weighted_axis':'compact. axis','neighbour_dist':'neighbour dist.','shape_idx':'shape index','facade_ratio':'facade ratio','corners':'corners','rect_idx':'rect. index','rectangularity':'rectangularity','square_compact':'square compact','elongation':'elongation','fractal':'fractal','convexity':'convexity','cell_alignment':'cell align.','perimeter':'perimeter','lal':'longest axis'}
# canonical per-feature palette (shades within the 5-group hues) — shared across figures
FEATCOL = {
 'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249',
 'perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418',
 'shared_walls':'#8C4A28','rectangularity':'#D99441','shape_idx':'#C77A2E','convexity':'#A0511C',
 'facade_ratio':'#E8A05A','corners':'#7A3A12','rect_idx':'#D9742E','square_compact':'#C46A1C','elongation':'#F0A868',
 'avg_weighted_dist':'#4C9F70','fractal':'#1E7A45',
 'num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566',
 'area':'#0F4D92','perimeter':'#5B8FC0','lal':'#2C6FB0',
}
# 25 -> 5 morphometric dimensions, and the canonical 5-group palette (same as group_signature)
GROUPS = {'Dimension':['area','perimeter','lal'],
          'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
          'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
          'Configuration':['tes_area','cell_alignment','num_neighbors'],
          'Connectivity':['fractal','avg_weighted_dist']}
FEAT_GROUP = {f:g for g,fs in GROUPS.items() for f in fs}
GBASE = {'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
# sites ordered by per-site TEST ACCURACY high->low (Supplementary Table S1 / tab:site_metrics_test), matching main-text Fig 2b
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP = {'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de Costa Barros','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
TOPN=15

imp_by_site={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+SH,read_geometry=False).dropna()
    g=g[g.label=='favela']
    imp=g[SH].abs().mean(); imp.index=RAW
    imp_by_site[s]=imp.sort_values(ascending=False)
    print(f"  {DISP[s]:22s} top3: "+", ".join(f"{PRETTY.get(f,f)}({v:.2f})" for f,v in imp_by_site[s].head(3).items()))

from matplotlib.patches import Patch
# ONE shared feature order, by overall (cross-site mean) importance; every panel uses it
overall = pd.concat([imp_by_site[s] for s in SITES], axis=1).mean(axis=1).sort_values(ascending=False)
order   = list(overall.index)[::-1]                  # reversed so most-important is at TOP in barh
ylabels = [PRETTY.get(f,f) for f in order]
colors  = [GBASE[FEAT_GROUP[f]] for f in order]
fig,axes=plt.subplots(2,5,figsize=(13.5,9.0))
for idx,(ax,s) in enumerate(zip(axes.ravel(),SITES)):
    ax.barh(range(len(order)), imp_by_site[s].reindex(order).values, color=colors, height=.82)
    ax.set_title(DISP[s],fontsize=8,fontweight='bold')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ylabels if idx%5==0 else [], fontsize=5.4)   # labels only on the left column
    ax.set_ylim(-0.6,len(order)-0.4)
    ax.tick_params(axis='x',labelsize=5.6); ax.set_xlabel('mean |SHAP|',fontsize=6.5)
fig.legend(handles=[Patch(fc=GBASE[g],label=g) for g in GROUPS],
           loc='lower center',ncol=5,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.005))
fig.suptitle('Per-site deep-feature importance (favela buildings only); features share one order (overall cross-site importance), bars coloured by morphometric dimension',fontsize=8.2,y=1.0)
fig.tight_layout(rect=[0,0.03,1,0.97])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_persite_importance_favela.{ext}'),bbox_inches='tight')
fig.savefig('/tmp/persite_imp_preview.png',dpi=115,bbox_inches='tight'); plt.close(fig)
print("shared order (top->bottom):", [PRETTY.get(f,f) for f in order[::-1]])
print("[saved] Figure_X_persite_importance_favela.pdf/.svg")
