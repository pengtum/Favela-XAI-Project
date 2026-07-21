"""
SHAP dependence for the top-5 morphological indicators (publication restyle of
the old shap_grid_all_sites.png, originally graphing.ipynb cells 5-6).
Instead of a 10x5 = 50-panel grid, this shows ONE panel per top feature:
the deep-feature contribution (SHAP, y) as a function of the feature value
(x), pooled over all ten sites, with a band spanning the per-site trends so
that BOTH the monotonic sign and the cross-site consistency read at a glance.
Pure visualization of stored SHAP from the per-site gpkg (positive SHAP ->
favela). Arial, editable text, raster scatter / vector lines. Exports PDF+SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 11,
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES  = os.path.join(PROJ,"results","all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
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
FAV='#E28E2C'; NFV='#6BAED6'; TREND='#14746A'; REF='#555555'
# 5 morphometric dimensions + canonical group palette (shared with the other figures); 25->5 mapping per Supplementary S3
GROUPS={'Dimension':['area','perimeter','lal'],
        'Shape':['shape_idx','compact_weighted_axis','cwa','convexity','facade_ratio','shared_walls','perimeter_wall','corners','rect_idx','rectangularity','squareness','square_compact','elongation'],
        'Spatial distribution':['neighbour_dist','mean_interbuilding_distance','adjacency','CAR'],
        'Configuration':['tes_area','cell_alignment','num_neighbors'],
        'Connectivity':['fractal','avg_weighted_dist']}
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}
FEAT_GROUP={f:g for g,fs in GROUPS.items() for f in fs}
# sites ordered by per-site TEST ACCURACY high->low (Supplementary Table S1), matching main-text Fig 2b
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]

print("loading per-site buildings (both classes) ...")
frames=[]
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH)
    d=pd.DataFrame({f:pd.to_numeric(g[f],errors='coerce').values for f in RAW})
    for f in RAW: d['shap_'+f]=pd.to_numeric(g['shap_'+f],errors='coerce').values
    d['site']=s; d['fav']=(g['label']=='favela').values
    frames.append(d); print(f"  {s:24s} {len(d):>7d}")
A=pd.concat(frames,ignore_index=True)

# ALL 25 features, GROUPED by morphometric dimension; groups AND within-group features ordered by mean|SHAP| (desc)
imp=A[SH].abs().mean()  # indexed by 'shap_<feature>'
gmean={g:float(np.mean([imp['shap_'+f] for f in fs])) for g,fs in GROUPS.items()}
gorder=sorted(GROUPS,key=lambda g:-gmean[g])
order=[f for g in gorder for f in sorted(GROUPS[g],key=lambda f:-imp['shap_'+f])]
print("group order (by mean per-feature |SHAP|):",[(g,round(gmean[g],3)) for g in gorder])
print("feature order:",[(PRETTY.get(f,f),FEAT_GROUP[f]) for f in order])

def binned(x,y,edges,minn=20):
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    idx=np.clip(np.digitize(x,edges)-1,0,len(edges)-2)
    out=np.full(len(edges)-1,np.nan)
    for b in range(len(edges)-1):
        sel=idx==b
        if sel.sum()>=minn: out[b]=np.nanmean(y[sel])
    return out

rng=np.random.default_rng(42)
NR,NC=5,5
fig,axes=plt.subplots(NR,NC,figsize=(12.0,12.6)); axf=axes.ravel()
for k,f in enumerate(order):
    ax=axf[k]; sh='shap_'+f
    x=A[f].values; y=A[sh].values; m=np.isfinite(x)&np.isfinite(y)
    xlo,xhi=np.percentile(x[m],[1,99])
    if xhi<=xlo: xhi=xlo+1e-9
    edges=np.linspace(xlo,xhi,13); cen=0.5*(edges[:-1]+edges[1:])
    # faint class-coloured scatter (subsampled, rasterized)
    for cls,col in [(False,NFV),(True,FAV)]:
        sub=A[A.fav==cls]; xs=sub[f].values; ys=sub[sh].values
        mm=np.isfinite(xs)&np.isfinite(ys)&(xs>=xlo)&(xs<=xhi); xs,ys=xs[mm],ys[mm]
        if len(xs)>1200: sel=rng.choice(len(xs),1200,replace=False); xs,ys=xs[sel],ys[sel]
        ax.scatter(xs,ys,s=1.4,c=col,alpha=0.14,lw=0,rasterized=True)
    # per-site trend band (min-max across the 10 sites) + pooled trend
    M=np.vstack([binned(A[A.site==s][f].values,A[A.site==s][sh].values,edges) for s in SITES])
    lo=np.nanmin(M,0); hi=np.nanmax(M,0); ok=np.isfinite(lo)&np.isfinite(hi)
    ax.fill_between(cen[ok],lo[ok],hi[ok],color=TREND,alpha=0.15,lw=0)
    pooled=binned(x[m],y[m],edges); pk=np.isfinite(pooled)
    ax.plot(cen[pk],pooled[pk],color=TREND,lw=1.8,solid_capstyle='round')
    ax.axhline(0,color=REF,lw=0.6,ls='--')
    ax.set_xlim(xlo,xhi)
    gc=GBASE[FEAT_GROUP[f]]
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(gc); sp.set_linewidth(1.7)
    ax.set_title(f'{PRETTY.get(f,f)}',fontsize=11.5,fontweight='bold',pad=3,color=gc)
    ax.tick_params(labelsize=9,color='#999999'); ax.locator_params(axis='both',nbins=3)
    if k%NC==0: ax.set_ylabel('SHAP ($\\rightarrow$ fav.)',fontsize=10.5)
for j in range(len(order),NR*NC): axf[j].axis('off')
handles=[Patch(fc=FAV,label='favela building'),Patch(fc=NFV,label='non-favela building'),
         Line2D([0],[0],color=TREND,lw=1.8,label='pooled trend'),
         Patch(fc=TREND,alpha=0.15,label='per-site range (10 sites)'),
         Line2D([0],[0],color=REF,lw=0.6,ls='--',label='SHAP = 0  (formal / favela)')]
fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=12.5,bbox_to_anchor=(0.5,-0.004))
gleg=[Patch(ec=GBASE[g],fc='none',lw=3.0,label=g) for g in gorder]
fig.legend(handles=gleg,loc='upper center',ncol=5,fontsize=12.5,bbox_to_anchor=(0.5,0.965),
           title='morphometric dimension (panel-frame colour) — groups & within-group features ordered by mean |SHAP|',title_fontsize=11.5)
fig.suptitle('Deep-feature SHAP dependence for all 25 morphological indicators (x = feature value, y = SHAP, positive $\\rightarrow$ favela; all ten sites)',fontsize=13.5,y=1.0)
fig.tight_layout(rect=[0,0.028,1,0.95])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_shap_dependence.{ext}'),bbox_inches='tight',dpi=350)
fig.savefig('/tmp/shap_dependence_preview.png',dpi=125,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_shap_dependence.pdf/.svg")
