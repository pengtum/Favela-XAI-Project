"""
Refined, S7c-style per-site SHAP dependence panels for the indicators with the
strongest cross-site HETEROGENEITY (interbuilding distance, adjacency, perimeter
wall, squareness). One figure per indicator: 2x2 of FOUR favelas auto-selected to
span the contrast -- one that follows the pooled trend + the three that deviate
most from it. Each panel: faint class-coloured scatter + bold per-site binned-mean
trend + faint pooled reference. Pure visualization of stored SHAP. PDF+SVG.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8.5,"axes.linewidth":0.6,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
FEATS=[("mean_interbuilding_distance","mean interbuilding distance  (m)","interbuild"),
       ("adjacency","building adjacency","adjacency"),
       ("perimeter_wall","perimeter wall","perimeterwall"),
       ("squareness","squareness","squareness")]
RAW=[f for f,_,_ in FEATS]; SH=["shap_"+f for f in RAW]
SITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
FAV="#E28E2C"; NFV="#6BAED6"; TREND="#14746A"; POOL="#9e9e9e"

print("loading (no geometry) ...")
data={}
for s in SITES:
    try: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH,ignore_geometry=True)
    except Exception: g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label']+RAW+SH).drop(columns='geometry',errors='ignore')
    d=pd.DataFrame({'fav':(g['label']=='favela').values})
    for f in RAW: d[f]=pd.to_numeric(g[f],errors='coerce'); d['shap_'+f]=pd.to_numeric(g['shap_'+f],errors='coerce')
    data[s]=d; print(f"  {s:22s} {len(d):>7d}")
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)
rng=np.random.default_rng(7)

def panels(feat,xlab,slug):
    x=ALL[feat].values; lo,hi=np.nanpercentile(x,[1,98]); edges=np.linspace(lo,hi,19); cen=0.5*(edges[:-1]+edges[1:])
    def bm(xx,yy,minn=25):
        m=np.isfinite(xx)&np.isfinite(yy)&(xx>=lo)&(xx<=hi); xx,yy=xx[m],yy[m]
        idx=np.clip(np.digitize(xx,edges)-1,0,len(edges)-2)
        return np.array([yy[idx==b].mean() if (idx==b).sum()>=minn else np.nan for b in range(len(edges)-1)])
    pooled=bm(x,ALL['shap_'+feat].values)
    per={s:bm(data[s][feat].values,data[s]['shap_'+feat].values) for s in SITES}
    dev={s:np.nanmean(np.abs(per[s]-pooled)) for s in SITES}
    canon=min(dev,key=dev.get); diverg=[s for s in sorted(SITES,key=lambda s:-dev[s]) if s!=canon][:3]
    sel=[canon]+diverg
    def tag(s):
        v=per[s]; p=pooled; ok=np.isfinite(v)&np.isfinite(p)
        if not ok.any(): return ''
        # opposite tail sign vs pooled -> reversal
        last=np.where(ok)[0][-1]
        if np.sign(v[last])!=np.sign(p[last]) and abs(v[last])>0.4: return 'reverses vs pooled'
        return 'follows pooled' if s==canon else 'diverges from pooled'
    ymax=np.nanmax([np.nanmax(np.abs(per[s])) for s in sel]+[np.nanmax(np.abs(pooled))])*1.08
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.0),sharex=True,sharey=True); axf=axes.ravel()
    for k,s in enumerate(sel):
        ax=axf[k]; d=data[s]
        ax.axhline(0,color='#bbbbbb',lw=0.7,ls='--')
        pk=np.isfinite(pooled); ax.plot(cen[pk],pooled[pk],color=POOL,lw=1.6,ls=(0,(4,2)),zorder=3)
        for cls,col in [(False,NFV),(True,FAV)]:
            sub=d[d.fav==cls]; xs=sub[feat].values; ys=sub['shap_'+feat].values
            mm=np.isfinite(xs)&np.isfinite(ys)&(xs>=lo)&(xs<=hi); xs,ys=xs[mm],ys[mm]
            if len(xs)>1800: ii=rng.choice(len(xs),1800,replace=False); xs,ys=xs[ii],ys[ii]
            ax.scatter(xs,ys,s=3,c=col,alpha=0.16,lw=0,rasterized=True,zorder=1)
        v=per[s]; ok=np.isfinite(v); ax.plot(cen[ok],v[ok],color=TREND,lw=2.6,marker='o',ms=3.2,zorder=5)
        ax.set_title(DISP[s],fontsize=10,fontweight='bold',pad=2,loc='left')
        ax.text(0.985,0.04,tag(s),transform=ax.transAxes,ha='right',va='bottom',fontsize=7.6,style='italic',color='#555555')
        ax.set_xlim(lo,hi); ax.set_ylim(-ymax,ymax); ax.tick_params(labelsize=7.5)
        if k%2==0: ax.set_ylabel('SHAP  ($\\rightarrow$ favela)',fontsize=8.3)
        if k//2==1: ax.set_xlabel(xlab,fontsize=8.3)
    handles=[Line2D([0],[0],color=TREND,lw=2.6,marker='o',ms=3.2,label='this favela (binned mean)'),
             Line2D([0],[0],color=POOL,lw=1.6,ls=(0,(4,2)),label='pooled, all ten sites'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor=FAV,markersize=6,label='favela building'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor=NFV,markersize=6,label='non-favela building')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(f'Same indicator, different dependence across favelas: {feat.replace("_"," ")}',fontsize=10.8,y=0.995)
    fig.tight_layout(rect=[0,0.03,1,0.97])
    for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_dep_{slug}.{ext}'),bbox_inches='tight',dpi=400)
    fig.savefig(f'/tmp/dep_{slug}_preview.png',dpi=135,bbox_inches='tight'); plt.close(fig)
    print(f"[saved] Figure_X_dep_{slug}  sites={[DISP[s] for s in sel]}")

for f,xlab,slug in FEATS: panels(f,xlab,slug)
print("done.")
