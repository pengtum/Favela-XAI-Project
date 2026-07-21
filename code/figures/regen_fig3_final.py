"""
FINAL merged main Figure 3 -- spatial + statistical anatomy of the building-scale
favela signature.
  (a) TOP, signature anatomy across the OTHER EIGHT favelas: dominant morphological
      feature per favela building (map), original Figure-3 palette.
  (legend band between a and b: dominant feature . morphometric dimension)
  (b) BOTTOM, two contrasting favelas in detail: Rocinha (interbuilding-dominant)
      vs Barao (squareness-dominant), each x {dominant feature, SHAP map of
      interbuilding distance, SHAP map of squareness} as LARGE framed maps; each
      SHAP map carries a small SHAP-dependence INSET in its emptiest corner (one
      site, one indicator; dep_adjacency style: dot = building, bold = this favela,
      dashed = pooled) -> spatial AND statistical heterogeneity of importance.
Pure visualization of stored SHAP + footprint geometry. Arial, editable, raster maps.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":11,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
RAW=["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
     "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
     "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
     "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH=["shap_"+f for f in RAW]
PRETTY={'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','shared_walls':'shared walls','avg_weighted_dist':'avg wt. dist.','num_neighbors':'no. neighbours','area':'area','CAR':'cov-area ratio','tes_area':'tess. cell area','cwa':'compact. axis (t)','compact_weighted_axis':'compact. axis'}
GROUPS={'Dimension':['area','perimeter','lal'],'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],'Configuration':['tes_area','cell_alignment','num_neighbors'],'Connectivity':['fractal','avg_weighted_dist']}
FEAT_GROUP={f:g for g,fs in GROUPS.items() for f in fs}
GABBR={'Spatial distribution':'spatial','Shape':'shape','Connectivity':'connectivity','Configuration':'config','Dimension':'dimension'}
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
FEATCOL={'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249','perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418','shared_walls':'#8C4A28','rectangularity':'#D99441','avg_weighted_dist':'#4C9F70','fractal':'#1E7A45','num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566','area':'#0F4D92','perimeter':'#5B8FC0'}
SITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
GREY='#e8e8e8'; OTHER='#c8c8c8'; GORDER=['Spatial distribution','Shape','Configuration','Connectivity','Dimension']
CMAP=LinearSegmentedColormap.from_list("blue_orange",["#0F4D92","#F4F4F4","#E28E2C"])
FAVc="#E28E2C"; NFVc="#6BAED6"; TREND="#14746A"; POOL="#9e9e9e"
HERO=[("rocinha","Rocinha","#1f77b4"),("barao","Barão","#d62728")]; HNAMES=[h[0] for h in HERO]
THUMBS=[s for s in SITES if s not in HNAMES]
FEATB="squareness"; HFEAT=["mean_interbuilding_distance",FEATB]
ZOOM={'rocinha':([679720,680980],[7455500,7456680])}

print("loading all ten sites (geometry + 25 SHAP) ...")
data={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].copy()
    for c in (HFEAT+SH): g[c]=pd.to_numeric(g[c],errors='coerce')
    g['driver']=[RAW[i] for i in np.argmax(g[SH].values,axis=1)]
    data[s]=g; print(f"  {s:22s} {len(g):>7d}")
allfav=pd.concat([data[s][data[s].label=='favela']['driver'] for s in SITES])
top=sorted(allfav.value_counts().head(6).index,key=lambda f:(GORDER.index(FEAT_GROUP[f]),-int((allfav==f).sum())))
print("top-6 drivers:",[PRETTY.get(f,f) for f in top])
fnorm={f:TwoSlopeNorm(vmin=-(v:=float(np.nanpercentile(np.abs(np.concatenate([data[s]['shap_'+f].values for s in HNAMES])),99))),vcenter=0,vmax=v) for f in HFEAT}

def dom_map(ax,g):
    fav=g[g.label=='favela']
    g[g.label=='not-favela'].plot(ax=ax,color=GREY,ec='none',rasterized=True)
    fav[~fav.driver.isin(top)].plot(ax=ax,color=OTHER,ec='none',rasterized=True)
    for f in top:
        sub=fav[fav.driver==f]
        if len(sub): sub.plot(ax=ax,color=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]),ec='none',rasterized=True)
    ax.set_aspect('equal',adjustable='datalim')
def square_bounds(bnd,pad=0.04):
    minx,miny,maxx,maxy=bnd; cx,cy=(minx+maxx)/2,(miny+maxy)/2; half=max(maxx-minx,maxy-miny)/2*(1+pad)
    return (cx-half,cy-half,cx+half,cy+half)
def set_sq(ax,sqb): ax.set_xlim(sqb[0],sqb[2]); ax.set_ylim(sqb[1],sqb[3])
def frame_map(ax,col='#5a5a5a',lw=0.9):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(lw)
def boundary(g,ax):
    try:
        reg=g[g.label=='favela'].geometry.simplify(1).buffer(10).unary_union.buffer(-6).simplify(2)
        gpd.GeoSeries([reg],crs=g.crs).boundary.plot(ax=ax,color='#111111',lw=0.6,zorder=5)
    except Exception: pass
rng=np.random.default_rng(5); DEP={}
for f in HFEAT:
    allx=np.concatenate([data[s][f].values for s in SITES]); ally=np.concatenate([data[s]['shap_'+f].values for s in SITES])
    lo,hi=np.nanpercentile(allx,[1,98]); edges=np.linspace(lo,hi,17); cen=0.5*(edges[:-1]+edges[1:])
    def bm(xx,yy,lo=lo,hi=hi,edges=edges,minn=20):
        mm=np.isfinite(xx)&np.isfinite(yy)&(xx>=lo)&(xx<=hi); xx,yy=xx[mm],yy[mm]; idx=np.clip(np.digitize(xx,edges)-1,0,len(edges)-2)
        return np.array([yy[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
    DEP[f]=dict(lo=lo,hi=hi,cen=cen,bm=bm,pooled=bm(allx,ally))
def emptiest_corner(g,sqb):
    minx,miny,maxx,maxy=sqb; cx,cy=(minx+maxx)/2,(miny+maxy)/2
    c=g.geometry.centroid; x=c.x.values; y=c.y.values; m=(x>=minx)&(x<=maxx)&(y>=miny)&(y<=maxy); x,y=x[m],y[m]
    q={'tl':int(((x<cx)&(y>cy)).sum()),'tr':int(((x>cx)&(y>cy)).sum()),'bl':int(((x<cx)&(y<cy)).sum()),'br':int(((x>cx)&(y<cy)).sum())}
    return {'tl':[0.02,0.585,0.43,0.40],'tr':[0.55,0.585,0.43,0.40],'bl':[0.02,0.02,0.43,0.40],'br':[0.55,0.02,0.43,0.40]}[min(q,key=q.get)]
def dep_inset(ax,s,f,sqb):
    D=DEP[f]; iax=ax.inset_axes(emptiest_corner(data[s],sqb),zorder=20); iax.set_facecolor('white'); iax.patch.set_alpha(0.45)
    iax.axhline(0,color='#cfcfcf',lw=0.6,ls='--',zorder=1)
    iax.plot(D['cen'],D['pooled'],color=POOL,lw=1.3,ls=(0,(3,2)),zorder=3)
    d=data[s]
    for cls,col in [('not-favela',NFVc),('favela',FAVc)]:
        sub=d[d.label==cls]; xs=sub[f].values; ys=sub['shap_'+f].values
        mm=np.isfinite(xs)&np.isfinite(ys)&(xs>=D['lo'])&(xs<=D['hi']); xs,ys=xs[mm],ys[mm]
        if len(xs)>700: ii=rng.choice(len(xs),700,replace=False); xs,ys=xs[ii],ys[ii]
        iax.scatter(xs,ys,s=2.4,c=col,alpha=0.20,lw=0,rasterized=True,zorder=2)
    yv=D['bm'](d[f].values,d['shap_'+f].values); ok=np.isfinite(yv)
    iax.plot(D['cen'][ok],yv[ok],color=TREND,lw=1.9,marker='o',ms=2.4,zorder=5)
    iax.set_xlim(D['lo'],D['hi']); iax.tick_params(labelsize=6.5,length=2,pad=1); iax.locator_params(axis='both',nbins=3)
    for sp in iax.spines.values(): sp.set_color('#777'); sp.set_linewidth(0.6)
    iax.set_title(f'{DISP[s]} · SHAP dep.',fontsize=7,pad=1.5,color=FEATCOL[f])

fig=plt.figure(figsize=(17.5,21.0))
outer=fig.add_gridspec(3,1,height_ratios=[0.60,0.14,1.0],hspace=0.02)

# ===== (a) TOP: eight other sites, dominant feature map =====
gtop=outer[0].subgridspec(2,4,hspace=0.10,wspace=0.02)
for k,s in enumerate(THUMBS):
    axm=fig.add_subplot(gtop[k//4,k%4])
    dom_map(axm,data[s]); set_sq(axm,square_bounds(tuple(data[s].total_bounds))); frame_map(axm); axm.set_title(DISP[s],fontsize=13,pad=4)
fig.text(0.012,0.978,'a',fontsize=20,fontweight='bold')
fig.text(0.5,0.978,'Signature anatomy across the other eight favelas: dominant feature per building',ha='center',fontsize=13.5,fontweight='bold')

# ===== legend band between a and b =====
axL=fig.add_subplot(outer[1]); axL.axis('off')
domleg=[Patch(fc=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]),label=f'{PRETTY.get(f,f)} · {GABBR[FEAT_GROUP[f]]}') for f in top]+[Patch(fc=OTHER,label='other feature'),Patch(fc=GREY,label='non-favela')]
axL.legend(handles=domleg,loc='center',bbox_to_anchor=(0.5,0.24),ncol=8,fontsize=11.5,title='dominant feature  ·  morphometric dimension',title_fontsize=12.5,handlelength=1.2,columnspacing=1.3,handletextpad=0.5)

# ===== (b) BOTTOM: two hero favelas in detail; SHAP dependence inset on each SHAP map =====
gb=outer[2].subgridspec(3,3,height_ratios=[1,1,0.085],hspace=0.045,wspace=0.025)
COLT=['dominant feature',f'SHAP: {PRETTY.get(HFEAT[0],HFEAT[0])}',f'SHAP: {PRETTY.get(HFEAT[1],HFEAT[1])}']
for i,(s,name,scol) in enumerate(HERO):
    g=data[s]
    bnd=(ZOOM[s][0][0],ZOOM[s][1][0],ZOOM[s][0][1],ZOOM[s][1][1]) if s in ZOOM else tuple(g.total_bounds); sqb=square_bounds(bnd)
    ax=fig.add_subplot(gb[i,0]); dom_map(ax,g); set_sq(ax,sqb); frame_map(ax)
    ax.text(-0.045,0.5,name,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=17,fontweight='bold',color=scol)
    if i==0: ax.set_title(COLT[0],fontsize=15,fontweight='bold',pad=6)
    for j,f in enumerate(HFEAT):
        ax=fig.add_subplot(gb[i,1+j]); g.plot(ax=ax,column='shap_'+f,cmap=CMAP,norm=fnorm[f],ec='none',rasterized=True)
        boundary(g,ax); ax.set_aspect('equal',adjustable='datalim'); set_sq(ax,sqb); frame_map(ax)
        if i==0: ax.set_title(COLT[1+j],fontsize=15,fontweight='bold',color=FEATCOL[f],pad=6)
        dep_inset(ax,s,f,sqb)
# bottom strip: colourbars under the two SHAP-map columns
for j,f in enumerate(HFEAT):
    axc=fig.add_subplot(gb[2,1+j]); axc.axis('off'); cax=axc.inset_axes([0.16,0.45,0.68,0.32])
    sm=mpl.cm.ScalarMappable(cmap=CMAP,norm=fnorm[f]); sm.set_array([])
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.ax.tick_params(labelsize=9); cb.outline.set_visible(False)
    cb.set_label(f'SHAP of {PRETTY.get(f,f)}  ($\\leftarrow$ formal  |  favela $\\rightarrow$)',fontsize=10.5)
axn=fig.add_subplot(gb[2,0]); axn.axis('off')
axn.text(0.0,0.6,'inset = SHAP dependence\ndot = building (orange favela / blue non-favela)\nbold = this favela · dashed = pooled, all sites',ha='left',va='center',fontsize=8.5,color='#555')
fig.text(0.012,0.604,'b',fontsize=20,fontweight='bold')
fig.text(0.5,0.606,'Two contrasting favelas in detail: spatial \\& statistical heterogeneity of feature importance',ha='center',va='bottom',fontsize=13,fontweight='bold')
fig.subplots_adjust(left=0.03,right=0.996,top=0.952,bottom=0.008)
fig.savefig('/tmp/fig3_final_preview.png',dpi=78,bbox_inches='tight')
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig3_final.{ext}'),bbox_inches='tight',dpi=320)
plt.close(fig); print("[saved] Figure_X_fig3_final.pdf/.svg")
