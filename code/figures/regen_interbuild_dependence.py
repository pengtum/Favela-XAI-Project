"""
Per-site dependence of ONE indicator (mean interbuilding distance), S7c-style:
a few TYPICAL favelas side by side, each showing how the deep-feature
contribution (SHAP) depends on the indicator value -- to make the cross-site
HETEROGENEITY of a single feature legible (main-text §2.2). Selected cases span
the contrast: Rocinha (textbook short->favela), Bar\~ao (reversal: large
spacing -> favela), Morro da Gua\'iba (steep short->favela), Catumbi (weak /
mixed). Each panel: faint class-coloured scatter + bold per-site binned-mean
trend + faint pooled reference. Pure visualization of stored SHAP. PDF+SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8.5,"axes.linewidth":0.6,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False,
})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
FEAT="mean_interbuilding_distance"; SHCOL="shap_"+FEAT
ALLSITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
SELECT=[("rocinha","Rocinha","textbook: short spacing $\\rightarrow$ favela"),
        ("barao","Barão","reversal: large spacing $\\rightarrow$ favela"),
        ("morro_da_guaiba","Morro da Guaíba","steep: short spacing $\\rightarrow$ favela"),
        ("catumbi","Catumbi","weak / mixed, near zero")]
FAV="#E28E2C"; NFV="#6BAED6"; TREND="#14746A"; POOL="#9e9e9e"

print("loading (no geometry) ...")
data={}
for s in ALLSITES:
    try: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label',FEAT,SHCOL],ignore_geometry=True)
    except Exception: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label',FEAT,SHCOL]).drop(columns='geometry',errors='ignore')
    d=pd.DataFrame({'x':pd.to_numeric(g[FEAT],errors='coerce'),'v':pd.to_numeric(g[SHCOL],errors='coerce'),'fav':(g['label']=='favela').values})
    data[s]=d.dropna(subset=['x','v']); print(f"  {s:22s} {len(data[s]):>7d}")
ALL=pd.concat([data[s] for s in ALLSITES],ignore_index=True)
DLO,DHI=np.percentile(ALL['x'].values,[1,98]); edges=np.linspace(DLO,DHI,19); cen=0.5*(edges[:-1]+edges[1:])
def binmean(x,y,minn=25):
    m=(x>=DLO)&(x<=DHI); x,y=x[m],y[m]; idx=np.clip(np.digitize(x,edges)-1,0,len(edges)-2)
    return np.array([y[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
pooled=binmean(ALL['x'].values,ALL['v'].values)
rng=np.random.default_rng(7)

fig,axes=plt.subplots(2,2,figsize=(8.6,7.0),sharex=True,sharey=True); axf=axes.ravel()
for k,(s,name,tag) in enumerate(SELECT):
    ax=axf[k]; d=data[s]
    ax.axhline(0,color='#bbbbbb',lw=0.7,ls='--')
    pk=np.isfinite(pooled); ax.plot(cen[pk],pooled[pk],color=POOL,lw=1.6,ls=(0,(4,2)),zorder=3,label='pooled (all sites)')
    for cls,col in [(False,NFV),(True,FAV)]:
        sub=d[d.fav==cls]; xs=sub['x'].values; ys=sub['v'].values
        mm=(xs>=DLO)&(xs<=DHI)
        xs,ys=xs[mm],ys[mm]
        if len(xs)>1800: sel=rng.choice(len(xs),1800,replace=False); xs,ys=xs[sel],ys[sel]
        ax.scatter(xs,ys,s=3,c=col,alpha=0.16,lw=0,rasterized=True,zorder=1)
    yv=binmean(d['x'].values,d['v'].values); ok=np.isfinite(yv)
    ax.plot(cen[ok],yv[ok],color=TREND,lw=2.6,marker='o',ms=3.2,zorder=5,label='this favela')
    ax.set_title(name,fontsize=10,fontweight='bold',pad=2,loc='left')
    ax.text(0.985,0.04,tag,transform=ax.transAxes,ha='right',va='bottom',fontsize=7.6,style='italic',color='#555555')
    ax.set_xlim(DLO,DHI); ax.tick_params(labelsize=7.5)
    if k%2==0: ax.set_ylabel('SHAP of interbuild. dist.\n($\\rightarrow$ favela)',fontsize=8.3)
    if k//2==1: ax.set_xlabel('mean interbuilding distance  (m)',fontsize=8.3)
handles=[Line2D([0],[0],color=TREND,lw=2.6,marker='o',ms=3.2,label='this favela (binned mean)'),
         Line2D([0],[0],color=POOL,lw=1.6,ls=(0,(4,2)),label='pooled, all ten sites'),
         Line2D([0],[0],marker='o',color='none',markerfacecolor=FAV,markersize=6,label='favela building'),
         Line2D([0],[0],marker='o',color='none',markerfacecolor=NFV,markersize=6,label='non-favela building')]
fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(0.5,-0.02))
fig.suptitle('The same indicator, different dependence across favelas: mean interbuilding distance',fontsize=10.8,y=0.995)
fig.tight_layout(rect=[0,0.03,1,0.97])
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_interbuild_dependence.{ext}'),bbox_inches='tight',dpi=400)
fig.savefig('/tmp/interbuild_dep_select_preview.png',dpi=135,bbox_inches='tight'); plt.close(fig)
print("[saved] Figure_X_interbuild_dependence.pdf/.svg")
