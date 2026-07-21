"""
Heterogeneity of a single indicator (mean interbuilding distance), two views:
  A) Figure_X_interbuild_shapmap.pdf -- per-building shap_mean_interbuilding_distance
     mapped across the ten favelas (diverging, + -> favela), WITH the favela
     boundary outline (favela footprints dissolved) overlaid on each site.
  B) Figure_X_interbuild_persite_dependence.pdf -- the Fig-4-style dependence
     (SHAP vs feature value) drawn PER SITE (one line each) + pooled, to show
     whether the indicator's dependence pattern differs across sites.
Pure visualization of stored SHAP. Arial, editable text, raster maps. PDF+SVG.
Site order = per-site test accuracy high->low (matches Fig 3 / Fig 4).
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "legend.frameon": False,
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES = os.path.join(PROJ,"results","all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
FEAT = "mean_interbuilding_distance"; SHCOL = "shap_"+FEAT
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP = {'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
CMAP = LinearSegmentedColormap.from_list("blue_orange", ["#0F4D92","#F4F4F4","#E28E2C"])
SITECOL = {s:c for s,c in zip(SITES, plt.cm.tab10(np.linspace(0,1,10)))}

print("loading ...")
data={}; allv=[]
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label',FEAT,SHCOL])
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].copy()
    g['v']=pd.to_numeric(g[SHCOL],errors='coerce'); g['x']=pd.to_numeric(g[FEAT],errors='coerce')
    data[s]=g; allv.append(g['v'].dropna().values); print(f"  {DISP[s]:18s} {len(g):>7d}")
vmax=float(np.percentile(np.abs(np.concatenate(allv)),99)); norm=TwoSlopeNorm(vmin=-vmax,vcenter=0.0,vmax=vmax)
print(f"vmax={vmax:.2f}")

# ===================== Figure A: map + favela boundary =====================
fig=plt.figure(figsize=(11.5,9.6)); gs=fig.add_gridspec(3,4,hspace=0.12,wspace=0.04)
for i,s in enumerate(SITES):
    ax=fig.add_subplot(gs[i//4,i%4]); g=data[s]
    g.plot(ax=ax,column='v',cmap=CMAP,norm=norm,ec='none',rasterized=True)
    fav=g[g.label=='favela']
    try:
        reg=fav.geometry.simplify(1).buffer(10).unary_union.buffer(-6).simplify(2)
        gpd.GeoSeries([reg],crs=g.crs).boundary.plot(ax=ax,color='#111111',lw=0.6,zorder=5)
    except Exception as e: print("  [bnd fail]",s,e)
    ax.set_title(DISP[s],fontsize=9,fontweight='bold'); ax.set_aspect('equal'); ax.axis('off')
lax=fig.add_subplot(gs[2,2:4]); lax.axis('off')
cax=lax.inset_axes([0.08,0.46,0.84,0.10])
sm=mpl.cm.ScalarMappable(cmap=CMAP,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
cb.set_label('deep-feature contribution of mean interbuilding distance\n($\\leftarrow$ formal | favela $\\rightarrow$)',fontsize=8.2)
lax.plot([0.1,0.2],[0.2,0.2],color='#111111',lw=0.9,transform=lax.transAxes)
lax.text(0.23,0.2,'official favela boundary',va='center',fontsize=7.5,transform=lax.transAxes)
fig.suptitle('Spatial pattern of the mean-interbuilding-distance contribution across the ten favelas',fontsize=10.5,y=0.98)
fig.tight_layout(rect=[0,0,1,0.965])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_interbuild_shapmap.{ext}'),bbox_inches='tight',dpi=400)
fig.savefig('/tmp/interbuild_shapmap_preview.png',dpi=125,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_interbuild_shapmap.pdf/.svg")

# ============ Figure B: per-site dependence (SHAP vs value), like Fig 4 ============
allx=np.concatenate([data[s]['x'].dropna().values for s in SITES])
DLO,DHI=np.percentile(allx,[1,98]); edges=np.linspace(DLO,DHI,16); cen=0.5*(edges[:-1]+edges[1:])
def binmean(x,y,minn=30):
    m=np.isfinite(x)&np.isfinite(y)&(x>=DLO)&(x<=DHI); x,y=x[m],y[m]
    idx=np.clip(np.digitize(x,edges)-1,0,len(edges)-2)
    return np.array([y[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
figB,axB=plt.subplots(figsize=(7.4,5.2))
for s in SITES:
    yv=binmean(data[s]['x'].values,data[s]['v'].values); ok=np.isfinite(yv)
    axB.plot(cen[ok],yv[ok],color=SITECOL[s],lw=1.7,marker='o',ms=2.5,label=DISP[s])
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)
yp=binmean(ALL['x'].values,ALL['v'].values); ok=np.isfinite(yp)
axB.plot(cen[ok],yp[ok],color='black',lw=3.2,label='pooled',zorder=12)
axB.axhline(0,color='#999999',lw=0.7,ls='--')
axB.set_xlim(DLO,DHI); axB.set_xlabel('mean interbuilding distance  (m)',fontsize=9)
axB.set_ylabel('deep-feature contribution (SHAP)  ($\\rightarrow$ favela)',fontsize=9)
axB.set_title('Per-site dependence of the mean-interbuilding-distance contribution\n(one line per favela; black = pooled)',fontsize=9.5,loc='left')
axB.legend(ncol=2,fontsize=7.2,loc='upper right')
figB.tight_layout()
for ext in ('pdf','svg'): figB.savefig(os.path.join(OUT,f'Figure_X_interbuild_persite_dependence.{ext}'),bbox_inches='tight',dpi=300)
figB.savefig('/tmp/interbuild_persite_dep_preview.png',dpi=130,bbox_inches='tight'); plt.close(figB)
print("[saved] Figure_X_interbuild_persite_dependence.pdf/.svg")
