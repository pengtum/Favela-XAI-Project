"""
DRAFT merged main Figure 3 -- spatial anatomy of the building-scale signal.
  (a) dominant morphological feature per favela building across the ten sites
      (which morphology, where) -- the signature anatomy [hero];
  (b) boundary spillover: probability of being classified favela as a smooth
      function of signed distance to the favela/non-favela boundary, pooled
      across sites -- continuity, not a step.
Signed distance is computed per building as the Euclidean distance (m) from its
centroid to the nearest edge of the OFFICIAL 2013 favela polygon (negative inside
favela, positive outside) -- identical to the mis-classification buffer analysis
and to main-text Figure 2d, so every "signed distance to favela boundary" axis in
the paper is on one scale.
Pure visualization of stored SHAP + footprint geometry. Arial, editable text.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd, shapely.geometry as sg
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "legend.frameon": False,
    "axes.linewidth": 0.6,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES = os.path.join(PROJ,"results","all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
POLY = gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp"))  # official 2013 favela polygon
RAW = ["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
       "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
       "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
       "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH = ["shap_"+f for f in RAW]
PRETTY = {'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall',
          'squareness':'squareness','area':'area','perimeter':'perimeter','shared_walls':'shared walls',
          'num_neighbors':'no. neighbours','CAR':'cov-area ratio','tes_area':'tess. cell area','neighbour_dist':'neighbour dist.',
          'shape_idx':'shape index','convexity':'convexity','elongation':'elongation','rectangularity':'rectangularity',
          'corners':'corners','compact_weighted_axis':'compact. axis','cwa':'compact. axis (t)','fractal':'fractal',
          'cell_alignment':'cell align.','avg_weighted_dist':'avg wt. dist.','lal':'longest axis','facade_ratio':'facade ratio',
          'rect_idx':'rect. index','square_compact':'square compact'}
GROUPS = {'Dimension':['area','perimeter','lal'],
 'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
 'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
 'Configuration':['tes_area','cell_alignment','num_neighbors'],'Connectivity':['fractal','avg_weighted_dist']}
FEAT_GROUP={f:g for g,fs in GROUPS.items() for f in fs}
GABBR={'Spatial distribution':'spatial','Shape':'shape','Connectivity':'connectivity','Configuration':'config','Dimension':'dimension'}
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
FEATCOL={'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249',
 'perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418',
 'shared_walls':'#8C4A28','rectangularity':'#D99441','avg_weighted_dist':'#4C9F70','fractal':'#1E7A45',
 'num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566','area':'#0F4D92','perimeter':'#5B8FC0'}
# sites by per-site TEST ACCURACY high->low (Supplementary Table S1) = main-text Fig 2b order
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
GREY='#e8e8e8'; OTHER='#c8c8c8'; GORDER=['Spatial distribution','Shape','Configuration','Connectivity','Dimension']

print("loading ...")
data={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label']+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    g['driver']=[RAW[i] for i in np.argmax(g[SH].values,axis=1)]
    c=g.geometry.centroid; fav=(g['label']=='favela').values
    pl=POLY.to_crs(g.crs); bb=g.total_bounds                                   # signed dist to nearest edge of official 2013 polygon
    near=pl[pl.intersects(sg.box(bb[0]-300,bb[1]-300,bb[2]+300,bb[3]+300))]
    dist=c.distance(near.boundary.union_all()).values
    g['signed_dist']=np.where(fav,-dist,dist); g['pred_fav']=(g['predicted_label']=='favela').astype(float)
    data[s]=g; print(f"  {DISP[s]:18s} {len(g):>7d}")

allfav=pd.concat([data[s][data[s].label=='favela']['driver'] for s in SITES])
top=sorted(allfav.value_counts().head(6).index,key=lambda f:(GORDER.index(FEAT_GROUP[f]),-int((allfav==f).sum())))
print("top drivers:",[(PRETTY.get(f,f),GABBR[FEAT_GROUP[f]]) for f in top])

# ---- layout: 3 map-rows (+thin comp bars) on top, spillover curve on bottom ----
fig=plt.figure(figsize=(11.5,14.6))
gs=fig.add_gridspec(8,4,height_ratios=[1,0.05,1,0.05,1,0.05,0.10,0.95],hspace=0.10,wspace=0.04)
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
    comp=fav['driver'].value_counts(normalize=True); left=0.0
    for f in top:
        w=float(comp.get(f,0.0))
        if w>0: axb.barh(0,w,left=left,height=1,color=FEATCOL.get(f,GBASE[FEAT_GROUP[f]])); left+=w
    ow=float(comp[~comp.index.isin(top)].sum())
    if ow>0: axb.barh(0,ow,left=left,height=1,color=OTHER)
    axb.set_xlim(0,1); axb.set_ylim(-0.5,0.5); axb.set_xticks([]); axb.set_yticks([])
    for sp in axb.spines.values(): sp.set_linewidth(0.4); sp.set_color('#bbbbbb')
# legend (dominant feature) in the two spare cells of the 3rd map row
lax=fig.add_subplot(gs[4,2:4]); lax.axis('off')
handles=[Patch(fc=FEATCOL.get(f,GBASE[FEAT_GROUP[f]]),label=f'{PRETTY.get(f,f)} · {GABBR[FEAT_GROUP[f]]}') for f in top]
handles+=[Patch(fc=OTHER,label='other feature'),Patch(fc=GREY,label='non-favela')]
lax.legend(handles=handles,loc='center',ncol=2,fontsize=8,title='(a)  dominant feature  ·  morphometric dimension',title_fontsize=8.5)

# ---- (b) SIGNATURE spillover per key indicator: cross-site mean +/- 1 SD ----
DLO,DHI=-40,40; edges=np.linspace(DLO,DHI,33); cen=0.5*(edges[:-1]+edges[1:])
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)
KEY=['mean_interbuilding_distance','adjacency','perimeter_wall','squareness']
def binmean(d,y):
    m=np.isfinite(d)&np.isfinite(y)&(d>=DLO)&(d<=DHI); d,y=d[m],y[m]
    idx=np.clip(np.digitize(d,edges)-1,0,len(edges)-2)
    return np.array([y[idx==b].mean() if (idx==b).sum()>=20 else np.nan for b in range(len(edges)-1)])
ax_b0=None
for j,f in enumerate(KEY):
    ax=fig.add_subplot(gs[7,j]); col=FEATCOL.get(f,'#666666')
    if j==0: ax_b0=ax
    M=np.vstack([binmean(data[s]['signed_dist'].values,data[s]['shap_'+f].values) for s in SITES])
    mu=np.nanmean(M,0); sd=np.nanstd(M,0); ok=np.isfinite(mu)
    ax.axvspan(DLO,0,color='#E28E2C',alpha=0.05); ax.axvspan(0,DHI,color='#6BAED6',alpha=0.05)
    ax.fill_between(cen[ok],(mu-sd)[ok],(mu+sd)[ok],color=col,alpha=0.22,lw=0)
    ax.plot(cen[ok],mu[ok],color=col,lw=2.0)
    ax.axvline(0,color='#B64342',lw=0.9,ls='--'); ax.axhline(0,color='#999999',lw=0.5,ls=':')
    ax.set_xlim(DLO,DHI); ax.set_title(PRETTY.get(f,f),fontsize=8,color=col,fontweight='bold',pad=2)
    ax.tick_params(labelsize=6.5); ax.set_xlabel('signed dist. (m)',fontsize=7)
    if j==0: ax.set_ylabel('mean SHAP\n($\\rightarrow$ favela)',fontsize=7)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

fig.suptitle('Figure 3 (draft) — Spatial anatomy of the building-scale morphological signal',fontsize=11,y=0.997)
fig.tight_layout(rect=[0,0,1,0.98])
pb=ax_b0.get_position()
fig.text(0.5,pb.y1+0.011,'(b)  Morphological signature (SHAP of key indicators) across the boundary — line = cross-site mean, band = $\\pm$1 SD across the ten sites; red dashed = legal boundary',ha='center',fontsize=8.0)
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig3_merged.{ext}'),bbox_inches='tight',dpi=350)
fig.savefig('/tmp/fig3_merged_preview.png',dpi=120,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_fig3_merged.pdf/.svg")

# ---- standalone APPENDIX figure: classification-probability spillover ----
def pcurve(df):
    d=df['signed_dist'].values; p=df['pred_fav'].values; m=np.isfinite(d)&(d>=DLO)&(d<=DHI); d,p=d[m],p[m]
    idx=np.clip(np.digitize(d,edges)-1,0,len(edges)-2)
    return np.array([p[idx==b].mean() if (idx==b).sum()>=25 else np.nan for b in range(len(edges)-1)])
figB,axb2=plt.subplots(figsize=(6.4,3.7))
Mb=np.vstack([pcurve(data[s]) for s in SITES]); lo=np.nanmin(Mb,0); hi=np.nanmax(Mb,0); ok=np.isfinite(lo)&np.isfinite(hi)
pp=pcurve(ALL); ppk=np.isfinite(pp)
axb2.axvspan(DLO,0,color='#E28E2C',alpha=0.06); axb2.axvspan(0,DHI,color='#6BAED6',alpha=0.06)
axb2.fill_between(cen[ok],lo[ok],hi[ok],color='#444444',alpha=0.13,lw=0,label='per-site range (10 sites)')
axb2.plot(cen[ppk],pp[ppk],color='#222222',lw=2.4,label='pooled')
axb2.axvline(0,color='#B64342',lw=1.0,ls='--'); axb2.axhline(0.5,color='#bbbbbb',lw=0.6,ls=':')
axb2.text(-38,0.94,'inside favela',fontsize=7.5,color='#9c5a16',va='top')
axb2.text(38,0.94,'outside (formal)',fontsize=7.5,color='#2f5d80',va='top',ha='right')
axb2.set_xlim(DLO,DHI); axb2.set_ylim(0,1.0)
axb2.set_xlabel('signed distance to boundary  (m;  $<0$ inside,  $>0$ outside)',fontsize=8)
axb2.set_ylabel('P(building classified favela)',fontsize=8)
axb2.set_title('Boundary spillover of the favela classification probability',fontsize=8.5,loc='left')
axb2.legend(fontsize=7,loc='center right')
for sp in ['top','right']: axb2.spines[sp].set_visible(False)
figB.tight_layout()
for ext in ('pdf','svg'): figB.savefig(os.path.join(OUT,f'Figure_X_spillover_prob.{ext}'),bbox_inches='tight',dpi=300)
figB.savefig('/tmp/spillover_prob_preview.png',dpi=130,bbox_inches='tight'); plt.close(figB)
print("[saved] Figure_X_spillover_prob.pdf/.svg (appendix)")

# ---- ALL-25 boundary spillover (companion to the dependence figure) ----
impall=ALL[SH].abs().mean()
gm={g:float(np.mean([impall['shap_'+f] for f in fs])) for g,fs in GROUPS.items()}
gord=sorted(GROUPS,key=lambda g:-gm[g])
order25=[f for g in gord for f in sorted(GROUPS[g],key=lambda f:-impall['shap_'+f])]
figC,axesC=plt.subplots(5,5,figsize=(12.5,13.0)); axfC=axesC.ravel()
for k,f in enumerate(order25):
    ax=axfC[k]; gc=GBASE[FEAT_GROUP[f]]
    M=np.vstack([binmean(data[s]['signed_dist'].values,data[s]['shap_'+f].values) for s in SITES])
    mu=np.nanmean(M,0); sd=np.nanstd(M,0); ok=np.isfinite(mu)
    ax.axvspan(DLO,0,color='#E28E2C',alpha=0.05); ax.axvspan(0,DHI,color='#6BAED6',alpha=0.05)
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
figC.suptitle('Boundary spillover of every morphological indicator: SHAP vs signed distance to the favela boundary ($x<0$ inside, $x>0$ outside)',fontsize=12.5,y=1.0)
figC.tight_layout(rect=[0,0,1,0.95])
for ext in ('pdf','svg'): figC.savefig(os.path.join(OUT,f'Figure_X_spillover_all25.{ext}'),bbox_inches='tight',dpi=300)
figC.savefig('/tmp/spillover_all25_preview.png',dpi=125,bbox_inches='tight'); plt.close(figC)
print("[saved] Figure_X_spillover_all25.pdf/.svg")
