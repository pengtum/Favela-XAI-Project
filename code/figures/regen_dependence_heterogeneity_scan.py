"""
APPENDIX figure (Fig-4 style): per-site SHAP dependence for ALL 25 indicators,
showing the CROSS-SITE heterogeneity of each indicator's dependence. Same layout
and styling as main-text Fig 4 (5x5, panels GROUPED by morphometric dimension =
frame/title colour, ordered by mean|SHAP| within and across groups), but instead
of a pooled trend + band it draws the TEN per-site binned-mean dependence lines
(coloured by site) + the pooled trend in bold, so the divergence/reversal of the
curves is visible. A heterogeneity score (mean cross-site std of the per-site
binned means) is annotated per panel. Pure visualization of stored SHAP. PDF+SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":11,"axes.linewidth":0.6,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
RAW=["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
     "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
     "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
     "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH=["shap_"+f for f in RAW]
PRETTY={'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall',
 'squareness':'squareness','area':'area','perimeter':'perimeter','shared_walls':'shared walls','num_neighbors':'no. neighbours',
 'CAR':'cov-area ratio','tes_area':'tess. cell area','neighbour_dist':'neighbour dist.','shape_idx':'shape index',
 'convexity':'convexity','elongation':'elongation','rectangularity':'rectangularity','corners':'corners',
 'compact_weighted_axis':'compact. axis','cwa':'compact. axis (t)','fractal':'fractal','cell_alignment':'cell align.',
 'avg_weighted_dist':'avg wt. dist.','lal':'longest axis','facade_ratio':'facade ratio','rect_idx':'rect. index','square_compact':'square compact'}
GROUPS={'Dimension':['area','perimeter','lal'],
 'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
 'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
 'Configuration':['tes_area','cell_alignment','num_neighbors'],'Connectivity':['fractal','avg_weighted_dist']}
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
FEAT_GROUP={f:g for g,fs in GROUPS.items() for f in fs}
SITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
SITECOL={s:c for s,c in zip(SITES, plt.cm.tab10(np.linspace(0,1,10)))}

print("loading (no geometry) ...")
data={}
for s in SITES:
    try: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH,ignore_geometry=True)
    except Exception: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH).drop(columns='geometry',errors='ignore')
    data[s]=g.apply(lambda c: pd.to_numeric(c,errors='coerce') if c.name in RAW+SH else c); print(f"  {s:22s} {len(data[s]):>7d}")
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)

def curves(f):
    x=ALL[f].values; lo,hi=np.nanpercentile(x,[1,98]); edges=np.linspace(lo,hi,16); cen=0.5*(edges[:-1]+edges[1:])
    def bm(xx,yy,minn=25):
        m=np.isfinite(xx)&np.isfinite(yy)&(xx>=lo)&(xx<=hi); xx,yy=xx[m],yy[m]
        idx=np.clip(np.digitize(xx,edges)-1,0,len(edges)-2)
        return np.array([yy[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
    per={s:bm(data[s][f].values,data[s]['shap_'+f].values) for s in SITES}; pooled=bm(x,ALL['shap_'+f].values)
    het=float(np.nanmean(np.nanstd(np.vstack([per[s] for s in SITES]),axis=0)))
    return cen,per,pooled,het,(lo,hi)
res={f:curves(f) for f in RAW}
imp=ALL[SH].abs().mean(); imp.index=RAW
# group order by mean per-feature |SHAP|; within group by |SHAP| (same as Fig 4)
gmean={g:float(np.mean([imp[f] for f in fs])) for g,fs in GROUPS.items()}
gord=sorted(GROUPS,key=lambda g:-gmean[g])
order=[f for g in gord for f in sorted(GROUPS[g],key=lambda f:-imp[f])]

fig,axes=plt.subplots(5,5,figsize=(13.0,13.4)); axf=axes.ravel()
for k,f in enumerate(order):
    ax=axf[k]; cen,per,pooled,het,(lo,hi)=res[f]; gc=GBASE[FEAT_GROUP[f]]
    ax.axhline(0,color='#cccccc',lw=0.6,ls='--')
    for s in SITES:
        yv=per[s]; ok=np.isfinite(yv); ax.plot(cen[ok],yv[ok],color=SITECOL[s],lw=1.0,alpha=0.85)
    ok=np.isfinite(pooled); ax.plot(cen[ok],pooled[ok],color='black',lw=2.4,zorder=10)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(gc); sp.set_linewidth(1.7)
    ax.set_title(f"{PRETTY.get(f,f)}",fontsize=11,color=gc,fontweight='bold',pad=3)
    ax.text(0.96,0.06,f"het {het:.2f}",transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,color='#555555')
    ax.tick_params(labelsize=8.5); ax.locator_params(axis='both',nbins=3)
    if k%5==0: ax.set_ylabel('SHAP ($\\rightarrow$ fav.)',fontsize=9.5)
gleg=[Patch(ec=GBASE[g],fc='none',lw=2.6,label=g) for g in gord]
l1=fig.legend(handles=gleg,loc='upper center',ncol=5,fontsize=11,bbox_to_anchor=(0.5,0.963),
    title='morphometric dimension (panel-frame colour); ordered by mean |SHAP|',title_fontsize=10.5)
fig.add_artist(l1)
sleg=[Line2D([0],[0],color='black',lw=2.4,label='pooled, all sites')]+[Line2D([0],[0],color=SITECOL[s],lw=1.4,label=DISP[s]) for s in SITES]
fig.legend(handles=sleg,loc='lower center',ncol=6,fontsize=9.5,bbox_to_anchor=(0.5,-0.012))
fig.suptitle('Cross-site heterogeneity of the SHAP dependence for all 25 indicators (ten per-site lines + pooled; x = feature value, y = SHAP $\\rightarrow$ favela)',fontsize=12.5,y=0.997)
fig.tight_layout(rect=[0,0.03,1,0.95])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_dependence_heterogeneity_scan.{ext}'),bbox_inches='tight',dpi=220)
fig.savefig('/tmp/dep_het_scan_preview.png',dpi=120,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_dependence_heterogeneity_scan.pdf/.svg (Fig-4 style, grouped)")
