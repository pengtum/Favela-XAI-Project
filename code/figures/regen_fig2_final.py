"""
Assembled main Figure 2 (favela-detection performance), accuracy-focused:
  a  study area: site locations in Rio + per-site building counts (favela/non-favela)   [original 2a, rebuilt]
  b  Figure-S5 case studies: four building clusters, classification outcome (FP red / FN purple)
  c  per-site test metrics (lollipop)  +  errors graded at the boundary (decay curve)   [V1 b + d]
Pure visualization of stored predictions + geometry. Caveat: predicted_label is in-sample;
the boundary curve plots a within-error SHARE distribution (location of errors), never an in-sample rate.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
import shapely.geometry as sg
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":7.5,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
FAV="#E28E2C"; NFV="#6BAED6"; NFVg="#CFCECE"; ERRc="#D62728"; FPc="#B64342"; FNc="#9A4D8E"; INK="#222222"; ACC="#2A6F97"
METRICS={'Rocinha':(0.977,0.981,0.978,0.980,0.977),'Morro da Guaíba':(0.962,0.919,0.942,0.930,0.952),
 'Barão':(0.958,0.929,0.911,0.920,0.946),'Jacarezinho':(0.954,0.940,0.929,0.934,0.949),
 'Faz Quem Quer':(0.949,0.906,0.914,0.910,0.937),'Catumbi':(0.934,0.913,0.921,0.917,0.931),
 'Palmeiras':(0.931,0.839,0.884,0.861,0.907),'Fazenda Coqueiro':(0.920,0.935,0.943,0.939,0.911),
 'Nova Cidade':(0.881,0.767,0.811,0.788,0.853),'Grotão de C.B.':(0.866,0.834,0.871,0.852,0.865)}
ORDER=list(METRICS); MEANACC=0.933
KEY={'Rocinha':'rocinha','Morro da Guaíba':'morro_da_guaiba','Barão':'barao','Jacarezinho':'jacarezinho','Faz Quem Quer':'faz_quem_quer','Catumbi':'catumbi','Palmeiras':'palmeras','Fazenda Coqueiro':'fazena_coqueiro','Nova Cidade':'nova_cidade','Grotão de C.B.':'grotao_de_costa_barros'}
CASES=[("Rocinha · boundary spillover",'rocinha',680475,7456125,190),("Rocinha · within-site cluster",'rocinha',679783,7456979,320),
       ("Grotão de C.B. · perimeter wall",'grotao_de_costa_barros',666525,7474875,200),("Grotão de C.B. · false negatives",'grotao_de_costa_barros',666825,7474575,190)]

print("loading ...")
CENT=pd.read_csv(os.path.join(OUT,'fig2_site_centroids.csv')).set_index('disp')
BE=pd.read_parquet(os.path.join(OUT,'boundary_error_data.parquet'))
RIO=gpd.read_file(os.path.join(OUT,'rio_boundary.geojson'))
CD={}
for s in set(c[1] for c in CASES):
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label'])
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    g['_x']=g.geometry.centroid.x; g['_y']=g.geometry.centroid.y
    g['FP']=(g.label=='not-favela')&(g.predicted_label=='favela'); g['FN']=(g.label=='favela')&(g.predicted_label=='not-favela')
    CD[s]=g

# ---------- a: locator + count bars ----------
def locator(ax):
    RIO.plot(ax=ax,color='#f4f4f4',zorder=0); RIO.boundary.plot(ax=ax,color='#b9b9b9',lw=0.7,zorder=1)
    for disp in ORDER:
        r=CENT.loc[disp]; ax.scatter(r['lon'],r['lat'],s=26,c=ACC,ec='white',lw=0.6,zorder=3)
        ax.annotate(disp,(r['lon'],r['lat']),xytext=(3,2),textcoords='offset points',fontsize=6,color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('Ten study sites, Rio de Janeiro',fontsize=8.5,fontweight='bold',loc='left')

def count_bars(ax):
    y=np.arange(len(ORDER))[::-1]
    nf=[CENT.loc[d,'n_fav'] for d in ORDER]; nn=[CENT.loc[d,'n_non'] for d in ORDER]
    ax.barh(y,nf,color=FAV,label='favela'); ax.barh(y,nn,left=nf,color=NFV,label='non-favela')
    ax.set_yticks(y); ax.set_yticklabels(ORDER,fontsize=6.5); ax.set_xlabel('buildings per site',fontsize=7.5)
    ax.set_title('Building counts',fontsize=8.5,fontweight='bold',loc='left'); ax.legend(loc='lower right',fontsize=6.5)
    ax.set_xlim(0,46000); ax.tick_params(labelsize=6.5)

# ---------- b: case-study windows ----------
def case_window(ax,case):
    title,site,cx,cy,r=case; g=CD[site]; w=g[(g._x>cx-r)&(g._x<cx+r)&(g._y>cy-r)&(g._y<cy+r)]
    w[w.label=='not-favela'].plot(ax=ax,color=NFVg,ec='white',lw=0.06,zorder=1)
    w[w.label=='favela'].plot(ax=ax,color=FAV,ec='white',lw=0.06,zorder=2)
    if w.FP.any(): w[w.FP].boundary.plot(ax=ax,color=FPc,lw=0.9,zorder=3)
    if w.FN.any(): w[w.FN].boundary.plot(ax=ax,color=FNc,lw=0.9,zorder=3)
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim(); xs=x0+(x1-x0)*0.06; ys=y0+(y1-y0)*0.07
    ax.plot([xs,xs+100],[ys,ys],color='k',lw=1.3,solid_capstyle='butt'); ax.text(xs+50,ys+(y1-y0)*0.02,'100 m',ha='center',va='bottom',fontsize=5.2)
    ax.set_title(title,fontsize=7,fontweight='bold'); ax.set_aspect('equal'); ax.axis('off')

# ---------- c: lollipop + boundary decay ----------
def lollipop(ax):
    y=np.arange(len(ORDER))[::-1]
    for i,disp in enumerate(ORDER):
        a,p,r,f1,fm=METRICS[disp]
        ax.plot([0.8,a],[y[i],y[i]],color='#cfcfcf',lw=1.2,zorder=1)
        ax.scatter(p,y[i],marker='|',s=110,c=NFV,zorder=2); ax.scatter(r,y[i],marker='|',s=110,c=FAV,zorder=2)
        ax.scatter(a,y[i],s=42,c=ACC,ec='white',lw=0.6,zorder=3)
    ax.axvline(MEANACC,color='#888',ls='--',lw=0.9)
    ax.text(MEANACC,len(ORDER)-0.1,f'mean {MEANACC:.3f}',fontsize=6.3,color='#666',ha='center',va='bottom')
    ax.set_yticks(y); ax.set_yticklabels(ORDER,fontsize=6.8); ax.set_xlim(0.8,1.005); ax.set_ylim(-0.6,len(ORDER)-0.1)
    ax.set_xlabel('test accuracy (dot) · precision $|$ recall (ticks)',fontsize=7)
    ax.set_title('Per-site detection performance',fontsize=8.5,fontweight='bold',loc='left')
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',mfc=ACC,ms=5.5,label='accuracy'),
        Line2D([0],[0],marker='|',color=NFV,label='precision'),Line2D([0],[0],marker='|',color=FAV,label='recall')],loc='lower right',fontsize=6.2)

def boundary_decay(ax,lo=-90,hi=90,bw=6):
    edges=np.arange(lo,hi+bw,bw); cen=0.5*(edges[:-1]+edges[1:])
    def dens(d):
        d=d[(d>=lo)&(d<=hi)]; h,_=np.histogram(d,bins=edges); return h/max(h.sum(),1)/bw*100
    ax.axvspan(lo,0,color=FAV,alpha=0.05); ax.axvspan(0,hi,color=NFV,alpha=0.05)
    M=np.vstack([dens(BE[(BE.site==KEY[d])&(BE.error)]['signed_dist'].values) for d in ORDER])
    ax.fill_between(cen,np.percentile(M,10,axis=0),np.percentile(M,90,axis=0),color='#cdd7e0',alpha=0.7,lw=0,label='per-site 10–90%')
    ax.plot(cen,dens(BE[BE.error]['signed_dist'].values),color=INK,lw=2.3,zorder=5,label='all sites (pooled)')
    ax.axvline(0,color='#444',ls='--',lw=1.0)
    ax.set_xlim(lo,hi); ax.set_xlabel('signed distance to favela boundary (m)  —  favela $\\leftarrow$ | $\\rightarrow$ formal',fontsize=7)
    ax.set_ylabel('share of misclassified buildings (%/m)',fontsize=7)
    ax.set_title('Errors are graded at the boundary',fontsize=8.5,fontweight='bold',loc='left')
    ax.text(0.97,0.93,'misclassifications collapse\nonto the favela edge\n(Rocinha 70.9% within 50 m)',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,color='#555')
    ax.legend(loc='upper left',fontsize=6.4)

# ---------- assemble ----------
fig=plt.figure(figsize=(7.6,11.0))
outer=fig.add_gridspec(3,1,height_ratios=[1.0,1.85,1.15],hspace=0.26)
ga=outer[0].subgridspec(1,2,width_ratios=[1.25,1.0],wspace=0.28); locator(fig.add_subplot(ga[0,0])); count_bars(fig.add_subplot(ga[0,1]))
gb=outer[1].subgridspec(2,2,hspace=0.16,wspace=0.06)
for k,case in enumerate(CASES): case_window(fig.add_subplot(gb[k//2,k%2]),case)
fig.legend(handles=[Patch(fc=FAV,label='favela (ground truth)'),Patch(fc=NFVg,label='non-favela'),
    Line2D([0],[0],color=FPc,lw=1.3,label='false positive (formal$\\to$favela)'),Line2D([0],[0],color=FNc,lw=1.3,label='false negative (favela$\\to$formal)')],
    loc='center',ncol=4,fontsize=6.6,bbox_to_anchor=(0.5,0.385))
gc=outer[2].subgridspec(1,2,width_ratios=[1.0,1.3],wspace=0.26); lollipop(fig.add_subplot(gc[0,0])); boundary_decay(fig.add_subplot(gc[0,1]))
for x,y,t in [(0.012,0.985,'a'),(0.012,0.66,'b'),(0.012,0.30,'c')]: fig.text(x,y,t,fontsize=14,fontweight='bold')
fig.suptitle('Figure 2 (assembled): study area · classification case studies (S5) · per-site performance & boundary-error decay',fontsize=9,y=1.005)
fig.savefig('/tmp/fig2_final.png',dpi=95,bbox_inches='tight')
for e in('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig2_assembled.{e}'),bbox_inches='tight',dpi=320)
plt.close(fig); print('[saved] Figure_X_fig2_assembled.pdf/.svg')
