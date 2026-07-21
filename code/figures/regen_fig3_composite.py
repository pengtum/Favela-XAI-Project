"""
DRAFT composite Figure-3 panel: spatial + statistical heterogeneity of feature
importance, two contrasting favelas (rows) x three LARGE map columns + a bottom
dependence row:
  col 0  dominant-feature map  (EXACT Fig-3 style: non-favela grey, favela
         coloured by dominant deep feature with the canonical FEATCOL palette)
  col 1  per-building SHAP map of indicator A (mean interbuilding distance)
  col 2  per-building SHAP map of indicator B (squareness | adjacency)  [+ favela boundary]
  bottom row: dominant-feature legend + SHAP dependence (dep_adjacency style: each
         dot = one building, two bold per-site trend lines) for indicators A and B,
         each plot aligned under its map column.
Rocinha (interbuilding-dominant) vs Barao (squareness-dominant). Loads only the
two hero sites. Pure visualization of stored SHAP. Arial, editable, raster maps.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":12,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
RAW=["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
     "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
     "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
     "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH=["shap_"+f for f in RAW]
PRETTY={'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','avg_weighted_dist':'avg wt. dist.','cwa':'compact. axis (t)','CAR':'cov-area ratio','tes_area':'tess. cell area'}
GABBR={'mean_interbuilding_distance':'spatial','adjacency':'spatial','perimeter_wall':'shape','squareness':'shape','cwa':'shape','avg_weighted_dist':'connect.'}
# canonical Fig-3 dominant-feature palette + GLOBAL top-6 drivers (hardcoded to match dominant_feature_map exactly)
FEATCOL={'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249','perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418','shared_walls':'#8C4A28','rectangularity':'#D99441','avg_weighted_dist':'#4C9F70','fractal':'#1E7A45','num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566','area':'#0F4D92','perimeter':'#5B8FC0'}
TOPDR=['mean_interbuilding_distance','adjacency','perimeter_wall','squareness','cwa','avg_weighted_dist']
GREY='#e8e8e8'; OTHER='#c8c8c8'
CMAP=LinearSegmentedColormap.from_list("blue_orange",["#0F4D92","#F4F4F4","#E28E2C"])
HERO=[("rocinha","Rocinha","#1f77b4"),("barao","Barão","#d62728")]
FEATB=os.environ.get("FEATB","squareness")              # second indicator: "squareness" or "adjacency"
SUF="" if FEATB=="squareness" else "_"+FEATB
HFEAT=["mean_interbuilding_distance",FEATB]; HFLAB={"mean_interbuilding_distance":"mean interbuilding distance  (m)","squareness":"squareness","adjacency":"building adjacency"}
# zoom each hero site's MAPS to a readable window (Rocinha is large -> focus on the separate south-east district)
ZOOM={'rocinha':([679720,680980],[7455500,7456680])}

print("loading two hero sites ...")
data={}
for s,_,_ in HERO:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].copy()
    for c in RAW+SH: g[c]=pd.to_numeric(g[c],errors='coerce')
    g['driver']=[RAW[i] for i in np.argmax(g[SH].values,axis=1)]
    data[s]=g; print(f"  {s:10s} {len(g):>7d}")
fvmax={f:float(np.nanpercentile(np.abs(np.concatenate([data[s]['shap_'+f].values for s,_,_ in HERO])),99)) for f in HFEAT}
fnorm={f:TwoSlopeNorm(vmin=-fvmax[f],vcenter=0,vmax=fvmax[f]) for f in HFEAT}

def boundary(g,ax):
    try:
        reg=g[g.label=='favela'].geometry.simplify(1).buffer(10).unary_union.buffer(-6).simplify(2)
        gpd.GeoSeries([reg],crs=g.crs).boundary.plot(ax=ax,color='#111111',lw=0.6,zorder=5)
    except Exception: pass

# --- maps-first layout: rows 0-1 = large maps (2 sites x 3 cols), row 2 = colourbars, row 3 = dominant legend + 2 dependence plots ---
fig=plt.figure(figsize=(17.5,16.6))
gs=fig.add_gridspec(4,3,height_ratios=[1,1,0.065,0.80],hspace=0.10,wspace=0.05)
COLTITLE=['dominant feature',f'SHAP: {PRETTY.get(HFEAT[0],HFEAT[0])}',f'SHAP: {PRETTY.get(HFEAT[1],HFEAT[1])}']
for i,(s,name,scol) in enumerate(HERO):
    g=data[s]; fav=g[g.label=='favela']
    # col 0: dominant feature map (exact Fig-3 style)
    ax=fig.add_subplot(gs[i,0])
    g[g.label=='not-favela'].plot(ax=ax,color=GREY,ec='none',rasterized=True)
    fav[~fav.driver.isin(TOPDR)].plot(ax=ax,color=OTHER,ec='none',rasterized=True)
    for f in TOPDR:
        sub=fav[fav.driver==f]
        if len(sub): sub.plot(ax=ax,color=FEATCOL[f],ec='none',rasterized=True)
    ax.set_aspect('equal')
    if s in ZOOM: ax.set_xlim(*ZOOM[s][0]); ax.set_ylim(*ZOOM[s][1])
    ax.axis('off')
    ax.text(-0.04,0.5,name,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=17,fontweight='bold',color=scol)
    if i==0: ax.set_title(COLTITLE[0],fontsize=15,fontweight='bold',pad=6)
    # col 1,2: SHAP maps of the two indicators
    for j,f in enumerate(HFEAT):
        ax=fig.add_subplot(gs[i,1+j]); g.plot(ax=ax,column='shap_'+f,cmap=CMAP,norm=fnorm[f],ec='none',rasterized=True)
        boundary(g,ax); ax.set_aspect('equal')
        if s in ZOOM: ax.set_xlim(*ZOOM[s][0]); ax.set_ylim(*ZOOM[s][1])
        ax.axis('off')
        if i==0: ax.set_title(COLTITLE[1+j],fontsize=15,fontweight='bold',color=FEATCOL[f],pad=6)
# row 2: colourbars under each SHAP-map column
for j,f in enumerate(HFEAT):
    axc=fig.add_subplot(gs[2,1+j]); axc.axis('off'); cax=axc.inset_axes([0.16,0.05,0.68,0.5])
    sm=mpl.cm.ScalarMappable(cmap=CMAP,norm=fnorm[f]); sm.set_array([])
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.ax.tick_params(labelsize=9); cb.outline.set_visible(False)
    cb.set_label(f'SHAP of {PRETTY.get(f,f)}  ($\\leftarrow$ formal  |  favela $\\rightarrow$)',fontsize=10.5)
# row 3 col 0: dominant-feature legend
axl=fig.add_subplot(gs[3,0]); axl.axis('off')
domleg=[Patch(fc=FEATCOL[f],label=f'{PRETTY.get(f,f)} ({GABBR.get(f,"")})') for f in TOPDR]+[Patch(fc=OTHER,label='other feature'),Patch(fc=GREY,label='non-favela')]
axl.legend(handles=domleg,loc='center',ncol=1,fontsize=12,title='dominant feature per building (col 1)',title_fontsize=12.5,handlelength=1.2,labelspacing=0.7)
# row 3 cols 1,2: SHAP dependence (each dot = a building; two per-site trend lines)
def binc(f):
    allx=np.concatenate([data[s][f].values for s,_,_ in HERO]); lo,hi=np.nanpercentile(allx,[1,98])
    edges=np.linspace(lo,hi,17); cen=0.5*(edges[:-1]+edges[1:])
    def bm(xx,yy,minn=20):
        mm=np.isfinite(xx)&np.isfinite(yy)&(xx>=lo)&(xx<=hi); xx,yy=xx[mm],yy[mm]; idx=np.clip(np.digitize(xx,edges)-1,0,len(edges)-2)
        return np.array([yy[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
    return cen,(lo,hi),bm
rng=np.random.default_rng(5)
for j,f in enumerate(HFEAT):
    ax=fig.add_subplot(gs[3,1+j]); cen,(lo,hi),bm=binc(f)
    ax.axhline(0,color='#cccccc',lw=0.8,ls='--')
    for s,name,scol in HERO:
        d=data[s]; xs=d[f].values; ys=d['shap_'+f].values; mm=np.isfinite(xs)&np.isfinite(ys)&(xs>=lo)&(xs<=hi); xs,ys=xs[mm],ys[mm]
        if len(xs)>1500: ii=rng.choice(len(xs),1500,replace=False); xs,ys=xs[ii],ys[ii]
        ax.scatter(xs,ys,s=5,c=scol,alpha=0.13,lw=0,rasterized=True,zorder=1)
        yv=bm(d[f].values,d['shap_'+f].values); ok=np.isfinite(yv); ax.plot(cen[ok],yv[ok],color=scol,lw=3.0,marker='o',ms=4,zorder=5,label=name)
    ax.set_xlim(lo,hi); ax.set_title(f'SHAP dependence: {PRETTY.get(f,f)}',fontsize=14,fontweight='bold',color=FEATCOL[f],loc='left',pad=4)
    ax.set_ylabel('SHAP ($\\rightarrow$ favela)',fontsize=12); ax.set_xlabel(HFLAB[f],fontsize=12); ax.tick_params(labelsize=10.5)
    ax.legend(fontsize=12,loc='best')
fig.suptitle(f'Figure 3 (draft): spatial \\& statistical heterogeneity of feature importance --- interbuilding distance + {PRETTY.get(FEATB,FEATB)}  (Rocinha vs Bar\\~ao)',fontsize=14,y=0.997)
fig.tight_layout(rect=[0,0.0,1,0.975])
fig.savefig(f'/tmp/fig3_composite{SUF}_preview.png',dpi=85,bbox_inches='tight')
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig3_composite{SUF}.{ext}'),bbox_inches='tight',dpi=340)
plt.close(fig); print(f"[saved] Figure_X_fig3_composite{SUF}.pdf/.svg")
