"""
City-wide map of the dominant morphological feature per building (all ten
sites). Each favela building is coloured by the feature with the largest
toward-favela SHAP. Feature colours are shades WITHIN their morphometric
dimension's canonical hue, so the colour family matches the 5-group palette
used in the group-composition figure (cool teal = spatial distribution,
warm = shape, green = connectivity). Pure visualization of stored SHAP.
Exports PDF + SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "legend.frameon": False,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES = os.path.join(PROJ,"results","all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
RAW = ["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
       "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
       "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
       "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH = ["shap_"+f for f in RAW]
PRETTY = {'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','shared_walls':'shared walls','avg_weighted_dist':'avg wt. dist.','num_neighbors':'no. neighbours','area':'area','CAR':'cov-area ratio','tes_area':'tess. cell area','cwa':'compact. axis (t)','compact_weighted_axis':'compact. axis'}
GROUPS = {
 'Dimension':            ['area','perimeter','lal'],
 'Shape':                ['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
 'Spatial distribution': ['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
 'Configuration':        ['tes_area','cell_alignment','num_neighbors'],
 'Connectivity':         ['fractal','avg_weighted_dist'],
}
FEAT_GROUP = {f:g for g,fs in GROUPS.items() for f in fs}
GABBR = {'Spatial distribution':'spatial','Shape':'shape','Connectivity':'connectivity','Configuration':'config','Dimension':'dimension'}
# canonical 5-group palette (group base hues — same as group-composition figure)
GBASE = {'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
# per-feature colours = distinct shades WITHIN the group hue (high contrast, but family = group)
FEATCOL = {
 'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249',  # spatial (teal)
 'perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418',
 'shared_walls':'#8C4A28','rectangularity':'#D99441',                                                       # shape (warm)
 'avg_weighted_dist':'#4C9F70','fractal':'#1E7A45',                                                          # connectivity (green)
 'num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566',                                  # configuration (violet)
 'area':'#0F4D92','perimeter':'#5B8FC0',                                                                     # dimension (blue)
}
# sites ordered by per-site TEST ACCURACY high->low (Supplementary Table S1 / tab:site_metrics_test), matching main-text Fig 2b
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP = {'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
GREY='#e8e8e8'; OTHER='#c8c8c8'
GORDER=['Spatial distribution','Shape','Configuration','Connectivity','Dimension']

data={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty]
    g['driver']=[RAW[i] for i in np.argmax(g[SH].values,axis=1)]
    data[s]=g
allfav=pd.concat([data[s][data[s].label=='favela']['driver'] for s in SITES])
top=sorted(allfav.value_counts().head(6).index,key=lambda f:(GORDER.index(FEAT_GROUP[f]),-int((allfav==f).sum())))
print("top features:",[(PRETTY.get(f,f),GABBR[FEAT_GROUP[f]],int((allfav==f).sum())) for f in top])

fig=plt.figure(figsize=(11.5,13.0))
gs=fig.add_gridspec(6,4,height_ratios=[1,0.045,1,0.045,1,0.045],hspace=0.09,wspace=0.04)
for i,s in enumerate(SITES):
    rp=i//4; c=i%4
    axm=fig.add_subplot(gs[rp*2,c]); axb=fig.add_subplot(gs[rp*2+1,c])
    g=data[s]; fav=g[g.label=='favela']
    g[g.label=='not-favela'].plot(ax=axm,color=GREY,ec='none',rasterized=True)
    fav[~fav.driver.isin(top)].plot(ax=axm,color=OTHER,ec='none',rasterized=True)
    for f in top:
        sub=fav[fav.driver==f]
        if len(sub): sub.plot(ax=axm,color=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]),ec='none',rasterized=True)
    axm.set_title(DISP[s],fontsize=8.5); axm.set_aspect('equal'); axm.axis('off')
    # per-site composition bar (share of favela buildings by dominant feature)
    comp=fav['driver'].value_counts(normalize=True); left=0.0
    for f in top:
        w=float(comp.get(f,0.0))
        if w>0: axb.barh(0,w,left=left,height=1,color=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]))
        left+=w
    ow=float(comp[~comp.index.isin(top)].sum())
    if ow>0: axb.barh(0,ow,left=left,height=1,color=OTHER)
    axb.set_xlim(0,1); axb.set_ylim(-0.5,0.5); axb.set_xticks([]); axb.set_yticks([])
    for sp in axb.spines.values(): sp.set_linewidth(0.4); sp.set_color('#bbbbbb')
handles=[Patch(fc=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]),label=f'{PRETTY.get(f,f)} · {GABBR[FEAT_GROUP[f]]}') for f in top]
handles+=[Patch(fc=OTHER,label='other feature'),Patch(fc=GREY,label='non-favela')]
lax=fig.add_subplot(gs[4:6,2:4]); lax.axis('off')
lax.legend(handles=handles,loc='center',ncol=2,fontsize=8,title='dominant feature  ·  morphometric dimension',title_fontsize=8)
fig.suptitle('Dominant morphological feature per favela building (map) with per-site composition (bar)\ncolour family = morphometric dimension',fontsize=9,y=0.997)
fig.tight_layout(rect=[0,0.0,1,0.975])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_dominant_feature_map.{ext}'),bbox_inches='tight',dpi=400)
fig.savefig('/tmp/dom_feat_preview.png',dpi=120,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_dominant_feature_map.pdf/.svg")
