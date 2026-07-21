"""
NEW Figure 4 (boundary spillover of all 25 indicators) with the signed distance
computed the SAME way as main-text Figure 2d / Section 2.1: Euclidean distance
(EPSG:31983) from each building centroid to the nearest edge of the OFFICIAL
2013 favela polygon (Limite_Favelas_2013.shp), negative inside / positive
outside. Replaces the KD-tree "nearest opposite-class building" proxy so that
every "signed distance to favela boundary" axis in the paper is on one scale.
Output: Figure_X_spillover_all25_polygon.pdf  (compare vs the proxy version
before swapping into the manuscript).
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
import shapely.geometry as sg
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"legend.frameon":False,"axes.linewidth":0.6})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
POLY=gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp"))  # official 2013 favela polygon
RAW=["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
     "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
     "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
     "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH=["shap_"+f for f in RAW]
PRETTY={'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall',
        'squareness':'squareness','area':'area','perimeter':'perimeter','shared_walls':'shared walls',
        'num_neighbors':'no. neighbours','CAR':'cov-area ratio','tes_area':'tess. cell area','neighbour_dist':'neighbour dist.',
        'shape_idx':'shape index','convexity':'convexity','elongation':'elongation','rectangularity':'rectangularity',
        'corners':'corners','compact_weighted_axis':'compact. axis','cwa':'compact. axis (t)','fractal':'fractal',
        'cell_alignment':'cell align.','avg_weighted_dist':'avg wt. dist.','lal':'longest axis','facade_ratio':'facade ratio',
        'rect_idx':'rect. index','square_compact':'square compact'}
GROUPS={'Dimension':['area','perimeter','lal'],
 'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
 'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
 'Configuration':['tes_area','cell_alignment','num_neighbors'],'Connectivity':['fractal','avg_weighted_dist']}
FEAT_GROUP={f:g for g,fs in GROUPS.items() for f in fs}
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
SITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}

print("loading + computing polygon-edge signed distance ...")
data={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label']+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    c=g.geometry.centroid; fav=(g['label']=='favela').values
    pl=POLY.to_crs(g.crs); bb=g.total_bounds                                   # official polygon -> signed distance to nearest edge
    near=pl[pl.intersects(sg.box(bb[0]-300,bb[1]-300,bb[2]+300,bb[3]+300))]
    bnd=near.boundary.union_all(); dist=c.distance(bnd).values
    g['signed_dist']=np.where(fav,-dist,dist)
    data[s]=g; print(f"  {DISP[s]:18s} {len(g):>7d}")
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)

# ---- bin SHAP vs signed distance (2.5 m bins), >=20 buildings per bin ----
DLO,DHI=-40,40; edges=np.linspace(DLO,DHI,33); cen=0.5*(edges[:-1]+edges[1:])
def binmean(d,y):
    m=np.isfinite(d)&np.isfinite(y)&(d>=DLO)&(d<=DHI); d,y=d[m],y[m]
    idx=np.clip(np.digitize(d,edges)-1,0,len(edges)-2)
    return np.array([y[idx==b].mean() if (idx==b).sum()>=20 else np.nan for b in range(len(edges)-1)])

impall=ALL[SH].abs().mean()
gm={g:float(np.mean([impall['shap_'+f] for f in fs])) for g,fs in GROUPS.items()}
gord=sorted(GROUPS,key=lambda g:-gm[g])
order25=[f for g in gord for f in sorted(GROUPS[g],key=lambda f:-impall['shap_'+f])]
figC,axesC=plt.subplots(5,5,figsize=(12.5,13.0)); axfC=axesC.ravel()
for k,f in enumerate(order25):
    ax=axfC[k]; gc=GBASE[FEAT_GROUP[f]]
    M=np.vstack([binmean(data[s]['signed_dist'].values,data[s]['shap_'+f].values) for s in SITES])
    mu=np.nanmean(M,0); sd=np.nanstd(M,0); ok=np.isfinite(mu)
    ax.axvspan(DLO,0,color='#3F83A2',alpha=0.06); ax.axvspan(0,DHI,color='#42A969',alpha=0.06)
    ax.fill_between(cen[ok],(mu-sd)[ok],(mu+sd)[ok],color=gc,alpha=0.22,lw=0)
    ax.plot(cen[ok],mu[ok],color=gc,lw=1.9)
    ax.axvline(0,color='#B64342',lw=0.8,ls='--'); ax.axhline(0,color='#999999',lw=0.5,ls=':')
    ax.set_xlim(DLO,DHI)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(gc); sp.set_linewidth(1.6)
    ax.set_title(PRETTY.get(f,f),fontsize=9.5,color=gc,fontweight='bold',pad=2)
    ax.tick_params(labelsize=7.5); ax.locator_params(axis='both',nbins=3)
    if k%5==0: ax.set_ylabel('SHAP ($\\rightarrow$ fav.)',fontsize=8.5)
gleg=[Patch(ec=GBASE[g],fc='none',lw=2.8,label=g) for g in gord]
figC.legend(handles=gleg,loc='upper center',ncol=5,fontsize=11,bbox_to_anchor=(0.5,0.964),
            title='morphometric dimension (frame colour) — line = cross-site mean, band = $\\pm$1 SD across the ten sites; red dashed = official 2013 favela boundary',title_fontsize=10)
figC.suptitle('Boundary spillover of every morphological indicator: SHAP vs signed distance to the official favela boundary ($x<0$ inside, $x>0$ outside)',fontsize=12.5,y=1.0)
figC.tight_layout(rect=[0,0,1,0.95])
for ext in ('pdf','svg'): figC.savefig(os.path.join(OUT,f'Figure_X_spillover_all25_polygon.{ext}'),bbox_inches='tight',dpi=300)
figC.savefig('/tmp/spillover_poly_preview.png',dpi=125,bbox_inches='tight'); plt.close(figC)
print("[saved] Figure_X_spillover_all25_polygon.pdf/.svg")
