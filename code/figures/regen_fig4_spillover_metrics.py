"""
NEW VARIANT of Figure 4 (does NOT touch Figure_X_spillover_all25.pdf): the same
polygon-distance SHAP spillover grid, but each box is annotated with three
boundary-transition metrics derived from the cross-site-mean profile S(x):
  - x0  = |S'|-weighted centroid of the change  -> WHERE it flips (m; <0 inside, >0 spillover-outside)
  - w   = |S'|-weighted std of the change       -> HOW gradual/fuzzy (m; small=sharp, large=fuzzy)
  - dS  = S_inside - S_outside                  -> boundary-discriminative STRENGTH
Per box: a dark horizontal bar at x0 +/- w (transition location & width) + text.
Plus a companion summary scatter: 25 features in the (x0, w) plane, size=|dS|,
colour=morphometric dimension.
Outputs: Figure_X_spillover_all25_metrics.pdf  and  Figure_X_spillover_metrics_scatter.pdf
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd, shapely.geometry as sg
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"legend.frameon":False,"axes.linewidth":0.6})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
POLY=gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp"))
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

print("loading + polygon signed distance ...")
data={}
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label']+SH)
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    c=g.geometry.centroid; fav=(g['label']=='favela').values
    pl=POLY.to_crs(g.crs); bb=g.total_bounds
    near=pl[pl.intersects(sg.box(bb[0]-300,bb[1]-300,bb[2]+300,bb[3]+300))]
    dist=c.distance(near.boundary.union_all()).values
    g['signed_dist']=np.where(fav,-dist,dist)
    data[s]=g; print(f"  {DISP[s]:18s} {len(g):>7d}")
ALL=pd.concat([data[s] for s in SITES],ignore_index=True)

DLO,DHI=-40,40; edges=np.linspace(DLO,DHI,33); cen=0.5*(edges[:-1]+edges[1:])
def binmean(d,y):
    m=np.isfinite(d)&np.isfinite(y)&(d>=DLO)&(d<=DHI); d,y=d[m],y[m]
    idx=np.clip(np.digitize(d,edges)-1,0,len(edges)-2)
    return np.array([y[idx==b].mean() if (idx==b).sum()>=20 else np.nan for b in range(len(edges)-1)])
def profile(f):
    M=np.vstack([binmean(data[s]['signed_dist'].values,data[s]['shap_'+f].values) for s in SITES])
    return np.nanmean(M,0), np.nanstd(M,0)
def xflip(x,y):   # first +->- crossing of SHAP = where the feature stops voting favela (favela->formal); inside->outside
    for i in range(len(x)-1):
        if y[i]>=0 and y[i+1]<0 and y[i]!=y[i+1]:
            return float(x[i]+(0-y[i])/(y[i+1]-y[i])*(x[i+1]-x[i]))
    return np.nan
def metrics(mu):
    ok=np.isfinite(mu)
    if ok.sum()<6: return np.nan,np.nan,np.nan
    x=cen[ok]; y=gaussian_filter1d(mu[ok],sigma=1.0)
    dy=np.gradient(y,x); wgt=np.abs(dy)
    w=float(np.sqrt(np.sum(wgt*(x-np.sum(wgt*x)/np.sum(wgt))**2)/np.sum(wgt))) if wgt.sum()>0 else np.nan
    Sin=np.nanmean(mu[cen<=-25]); Sout=np.nanmean(mu[cen>=25]); dS=float(Sin-Sout)
    return xflip(x,y),w,dS   # x0 = SHAP sign-flip distance (zero-crossing)

impall=ALL[SH].abs().mean()
gm={g:float(np.mean([impall['shap_'+f] for f in fs])) for g,fs in GROUPS.items()}
gord=sorted(GROUPS,key=lambda g:-gm[g])
order25=[f for g in gord for f in sorted(GROUPS[g],key=lambda f:-impall['shap_'+f])]
MET={}
print(f"\n{'feature':20s} {'flip(m)':>8s} {'w(m)':>6s} {'|dS|':>6s}")
for f in order25:
    mu,sd=profile(f); x0,w,dS=metrics(mu); MET[f]=(mu,sd,x0,w,dS)
    print(f"{PRETTY.get(f,f):20s} {x0:8.1f} {w:6.1f} {abs(dS):6.2f}")

# ---------- figure A: annotated 5x5 grid ----------
figC,axesC=plt.subplots(5,5,figsize=(12.5,13.0)); axfC=axesC.ravel()
for k,f in enumerate(order25):
    ax=axfC[k]; gc=GBASE[FEAT_GROUP[f]]; mu,sd,x0,w,dS=MET[f]; ok=np.isfinite(mu)
    ax.axvspan(DLO,0,color='#3F83A2',alpha=0.06); ax.axvspan(0,DHI,color='#42A969',alpha=0.06)
    ax.fill_between(cen[ok],(mu-sd)[ok],(mu+sd)[ok],color=gc,alpha=0.20,lw=0)
    ax.plot(cen[ok],mu[ok],color=gc,lw=1.8)
    ax.axvline(0,color='#B64342',lw=0.8,ls='--'); ax.axhline(0,color='#999999',lw=0.5,ls=':')
    # mark the SHAP sign-flip: a dot ON the curve where it crosses 0 (favela-vote -> formal-vote) + its distance.
    # shown for any resolved transition (|dS|>=0.10); dot/label crispness scales with |dS| (strong = solid, weak = faint).
    if np.isfinite(x0) and abs(dS)>=0.10:
        aa=float(np.clip(abs(dS)/0.6,0.42,1.0))
        ax.plot(x0,0,marker='o',ms=8,mfc=(0.1,0.1,0.1,0.40*aa),mec=(0.1,0.1,0.1,aa),mew=1.4,zorder=9)  # translucent: line shows through
        ha='left' if x0<25 else 'right'; dx=7 if x0<25 else -7
        ax.annotate(f'$x_0$={x0:+.1f} m',(x0,0),xytext=(dx,9),textcoords='offset points',ha=ha,fontsize=7.3,color='#1a1a1a',alpha=aa,fontweight='bold',zorder=9)
    ax.set_xlim(DLO,DHI)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(gc); sp.set_linewidth(1.6)
    ax.set_title(PRETTY.get(f,f),fontsize=9.0,color=gc,fontweight='bold',pad=4)
    ax.tick_params(labelsize=7.5); ax.locator_params(axis='both',nbins=3)
    if k%5==0: ax.set_ylabel('SHAP ($\\rightarrow$ fav.)',fontsize=8.5)
gleg=[Patch(ec=GBASE[g],fc='none',lw=2.8,label=g) for g in gord]
gleg+=[Line2D([0],[0],marker='o',color='none',mfc='#1a1a1a',mec='white',ms=8,label='$x_0$ = SHAP sign-flip (favela $\\to$ formal); label = its distance from boundary')]
figC.legend(handles=gleg,loc='upper center',ncol=3,fontsize=10,bbox_to_anchor=(0.5,0.965),
            title='frame colour = morphometric dimension; line = cross-site mean, band = $\\pm$1 SD; red dashed = official 2013 favela boundary',title_fontsize=9.5)
figC.suptitle('Boundary spillover of every indicator: SHAP vs signed distance to the official favela boundary — dot marks where each feature flips favela$\\to$formal',fontsize=12,y=1.0)
figC.tight_layout(rect=[0,0,1,0.95])
for ext in ('pdf','svg'): figC.savefig(os.path.join(OUT,f'Figure_X_spillover_all25_metrics.{ext}'),bbox_inches='tight',dpi=300)
figC.savefig('/tmp/spillover_metrics_grid.png',dpi=120,bbox_inches='tight'); plt.close(figC)
print("[saved] Figure_X_spillover_all25_metrics.pdf")

# ---------- figure B: summary scatter in the (x0, w) plane ----------
figS,axS=plt.subplots(figsize=(8.4,6.6))
dSmax=max(abs(MET[f][4]) for f in order25)
axS.axvspan(-40,0,color='#3F83A2',alpha=0.05); axS.axvspan(0,40,color='#42A969',alpha=0.05)
axS.axvline(0,color='#B64342',lw=1.0,ls='--')
for f in order25:
    mu,sd,x0,w,dS=MET[f]
    if not (np.isfinite(x0) and np.isfinite(w)): continue
    gc=GBASE[FEAT_GROUP[f]]; size=50+780*abs(dS)/dSmax; aa=float(np.clip(abs(dS)/0.5,0.28,0.85))
    axS.scatter(x0,w,s=size,color=gc,alpha=aa,ec='white',lw=0.8,zorder=3)
    if abs(dS)>=0.2:   # label only indicators with a real boundary signal (weak ones: x0,w are noise)
        axS.annotate(PRETTY.get(f,f),(x0,w),fontsize=6.8,color='#222',ha='center',va='center',fontweight='bold',
                     xytext=(0,-np.sqrt(size)/2-6),textcoords='offset points',zorder=5)
xs=[MET[f][2] for f in order25 if np.isfinite(MET[f][2])]; ws=[MET[f][3] for f in order25 if np.isfinite(MET[f][3])]
axS.set_xlim(min(xs)-4,max(xs)+5); axS.set_ylim(11,max(ws)*1.12)
axS.set_xlabel('SHAP sign-flip distance  $x_0$  (m;  where the feature stops voting favela;  $>0$ = flips outside the boundary = spillover)',fontsize=9)
axS.set_ylabel('transition width  $w$  (m;  small = sharp boundary,  large = fuzzy / graded)',fontsize=9)
axS.set_title('Where each morphological indicator flips favela$\\to$formal, and how gradual the change is',fontsize=10,loc='left')
axS.text(0.5,0.045,'The dominant indicators (interbuild, adjacency, perimeter wall, squareness) flip $\\sim$+3 to +5 m OUTSIDE the legal line —\nthe signature spills a few metres into the formal fabric — over a consistent $\\sim\\pm$17 m width.  (Exception: avg wt. dist. flips $\\sim$6 m inside.)',
         transform=axS.transAxes,ha='center',va='bottom',fontsize=7.6,color='#555')
gleg=[Line2D([0],[0],marker='o',color='none',mfc=GBASE[g],ms=8,label=g) for g in gord]
gleg+=[Line2D([0],[0],marker='o',color='none',mfc='#bbb',ms=ms,label=lab) for ms,lab in [(6,'|ΔS| small'),(12,'|ΔS| large')]]
axS.legend(handles=gleg,loc='upper right',fontsize=7.5,title='dimension · point size = strength |ΔS|',title_fontsize=7.8)
for sp in ['top','right']: axS.spines[sp].set_visible(False)
figS.tight_layout()
for ext in ('pdf','svg'): figS.savefig(os.path.join(OUT,f'Figure_X_spillover_metrics_scatter.{ext}'),bbox_inches='tight',dpi=300)
figS.savefig('/tmp/spillover_metrics_scatter.png',dpi=135,bbox_inches='tight'); plt.close(figS)
print("[saved] Figure_X_spillover_metrics_scatter.pdf")

# ---------- LaTeX table of the transition metrics ----------
def row(f):
    mu,sd,x0,w,dS=MET[f]
    xs=(f'${x0:+.1f}$' if (abs(dS)>=0.10 and np.isfinite(x0)) else '--')
    ws=(f'{w:.1f}'     if (abs(dS)>=0.10 and np.isfinite(w))  else '--')
    return f'{PRETTY.get(f,f)} & {FEAT_GROUP[f]} & {abs(dS):.2f} & {xs} & {ws} \\\\'
body='\n'.join(row(f) for f in sorted(order25,key=lambda f:-abs(MET[f][4])))
tex=(r'''\begin{table}[H]\centering
\caption{\textbf{Boundary-transition metrics for the 25 morphological indicators.} For each indicator,
the cross-site-mean Tree-SHAP contribution $S(x)$ is taken in 2.5\,m bins of signed distance $x$ to the
official 2013 favela boundary ($x<0$ inside, $x>0$ outside; EPSG:31983). $|\Delta S|=|S_{\mathrm{in}}-S_{\mathrm{out}}|$
is the boundary-discriminative \emph{strength} (deep-inside minus deep-outside SHAP level); $x_0$ is the
\emph{sign-flip distance} where $S$ crosses zero, i.e.\ where the indicator stops voting favela
($x_0>0$ = outside the legal line, a spillover into the formal fabric); $w=\sqrt{\sum_x|S'(x)|\,(x-\bar{x})^2/\sum_x|S'(x)|}$
is the $|S'|$-weighted transition \emph{width} (small = sharp, large = graded). $x_0$ and $w$ are reported only where a
transition is resolved ($|\Delta S|\ge0.1$); for the remaining indicators the profile is essentially flat. Rows are
ordered by $|\Delta S|$.}
\label{tab:spillover_metrics}
\begin{tabular}{llrrr}
\toprule
Indicator & Morphometric dimension & $|\Delta S|$ & $x_0$ (m) & $w$ (m) \\
\midrule
'''+body+r'''
\bottomrule
\end{tabular}
\end{table}
''')
open(os.path.join(OUT,'spillover_metrics_table.tex'),'w').write(tex)
print("[saved] spillover_metrics_table.tex\n"); print(tex)
