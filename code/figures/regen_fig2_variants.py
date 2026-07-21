"""
Three candidate redesigns of main Figure 2 (favela-detection performance),
accuracy-focused (the old feature-importance panel moves to the merged Fig 3).
Shared panels:
  - study-area map: Rio outline + 10 site dots (area = building count, orange
    favela / blue non-favela split)
  - per-site test metrics (lollipop or grouped bars) from Supplementary Table S2
    (reported numbers, verbatim)
  - boundary-graded errors: density of MISCLASSIFIED buildings vs signed distance
    to the official 2013 favela polygon (negative inside favela / positive
    outside); pooled + per-site band. Caveat-safe: a within-error SHARE
    distribution (where errors are), never an in-sample accuracy/rate.
  - spatial proof: one site, correct grey / error red, with the polygon edge.
V1 narrative 4-panel, V2 dashboard 2x2, V3 minimal 2-panel.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
FAV="#E28E2C"; NFV="#6BAED6"; ERRc="#D62728"; OKc="#d9d9d9"; INK="#222222"; ACC="#2A6F97"
# Supplementary Table S2 (reported held-out test metrics) -- verbatim, accuracy-descending
METRICS={ # site:(accuracy,precision,recall,f1_fav,f1_macro)
 'Rocinha':(0.977,0.981,0.978,0.980,0.977),'Morro da Guaíba':(0.962,0.919,0.942,0.930,0.952),
 'Barão':(0.958,0.929,0.911,0.920,0.946),'Jacarezinho':(0.954,0.940,0.929,0.934,0.949),
 'Faz Quem Quer':(0.949,0.906,0.914,0.910,0.937),'Catumbi':(0.934,0.913,0.921,0.917,0.931),
 'Palmeiras':(0.931,0.839,0.884,0.861,0.907),'Fazenda Coqueiro':(0.920,0.935,0.943,0.939,0.911),
 'Nova Cidade':(0.881,0.767,0.811,0.788,0.853),'Grotão de C.B.':(0.866,0.834,0.871,0.852,0.865)}
ORDER=list(METRICS); MEANACC=0.933
KEY={'Rocinha':'rocinha','Morro da Guaíba':'morro_da_guaiba','Barão':'barao','Jacarezinho':'jacarezinho','Faz Quem Quer':'faz_quem_quer','Catumbi':'catumbi','Palmeiras':'palmeras','Fazenda Coqueiro':'fazena_coqueiro','Nova Cidade':'nova_cidade','Grotão de C.B.':'grotao_de_costa_barros'}

print("loading prepped data ...")
CENT=pd.read_csv(os.path.join(OUT,'fig2_site_centroids.csv'))
BE=pd.read_parquet(os.path.join(OUT,'boundary_error_data.parquet'))
RIO=gpd.read_file(os.path.join(OUT,'rio_boundary.geojson'))

# ---------- panel functions ----------
def study_area(ax,labelsize=7,title=True):
    RIO.boundary.plot(ax=ax,color='#b9b9b9',lw=0.7,zorder=1)
    RIO.plot(ax=ax,color='#f4f4f4',zorder=0)
    cc=CENT.set_index('disp')
    smax=cc['n'].max()
    for disp in ORDER:
        r=cc.loc[disp]; s_tot=420*r['n']/smax+18; s_fav=420*r['n_fav']/smax+8
        ax.scatter(r['lon'],r['lat'],s=s_tot,c=NFV,ec='white',lw=0.5,zorder=3)
        ax.scatter(r['lon'],r['lat'],s=s_fav,c=FAV,ec='none',zorder=4)
        ax.annotate(disp,(r['lon'],r['lat']),xytext=(3,3),textcoords='offset points',fontsize=labelsize,color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if title: ax.set_title('Ten study sites in Rio de Janeiro',fontsize=labelsize+2,fontweight='bold',loc='left')
    h=[Line2D([0],[0],marker='o',color='none',mfc=NFV,ms=9,label='buildings (size $\\propto$ count)'),
       Line2D([0],[0],marker='o',color='none',mfc=FAV,ms=7,label='favela share')]
    ax.legend(handles=h,loc='lower left',fontsize=labelsize-1,handletextpad=0.3,labelspacing=0.3)

def lollipop(ax):
    y=np.arange(len(ORDER))[::-1]
    for i,disp in enumerate(ORDER):
        a,p,r,f1,fm=METRICS[disp]
        ax.plot([0.8,a],[y[i],y[i]],color='#cfcfcf',lw=1.2,zorder=1)
        ax.scatter(p,y[i],marker='|',s=120,c=NFV,zorder=2); ax.scatter(r,y[i],marker='|',s=120,c=FAV,zorder=2)
        ax.scatter(a,y[i],s=46,c=ACC,ec='white',lw=0.6,zorder=3)
    ax.axvline(MEANACC,color='#888',ls='--',lw=0.9,zorder=0)
    ax.text(MEANACC,len(ORDER)-0.2,f'mean {MEANACC:.3f}',fontsize=6.5,color='#666',ha='center',va='bottom')
    ax.set_yticks(y); ax.set_yticklabels(ORDER,fontsize=7); ax.set_xlim(0.8,1.005); ax.set_ylim(-0.6,len(ORDER)-0.2)
    ax.set_xlabel('test accuracy (dot) · precision $|$ recall (ticks)',fontsize=7.5)
    ax.set_title('Per-site detection performance',fontsize=9,fontweight='bold',loc='left')
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',mfc=ACC,ms=6,label='accuracy'),
        Line2D([0],[0],marker='|',color=NFV,label='precision'),Line2D([0],[0],marker='|',color=FAV,label='recall')],
        loc='lower right',fontsize=6.5,ncol=1)

def bars(ax):
    x=np.arange(len(ORDER)); w=0.26
    P=[METRICS[d][1] for d in ORDER]; R=[METRICS[d][2] for d in ORDER]; F=[METRICS[d][3] for d in ORDER]; A=[METRICS[d][0] for d in ORDER]
    ax.bar(x-w,P,w,color=NFV,label='precision'); ax.bar(x,R,w,color=FAV,label='recall'); ax.bar(x+w,F,w,color='#7a7a7a',label='F$_1$ (favela)')
    ax.scatter(x,A,marker='D',s=22,c=ACC,ec='white',lw=0.5,zorder=5,label='accuracy')
    ax.axhline(MEANACC,color='#888',ls='--',lw=0.9); ax.text(len(ORDER)-0.5,MEANACC+0.003,f'mean acc {MEANACC:.3f}',fontsize=6.5,color='#666',ha='right')
    ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=40,ha='right',fontsize=6.8); ax.set_ylim(0.74,1.0)
    ax.set_ylabel('score',fontsize=8); ax.set_title('Per-site detection performance (held-out test set)',fontsize=9,fontweight='bold',loc='left')
    ax.legend(loc='lower left',fontsize=6.5,ncol=4,columnspacing=1.0)

def boundary_decay(ax,lo=-90,hi=90,bw=6,annotate=True):
    edges=np.arange(lo,hi+bw,bw); cen=0.5*(edges[:-1]+edges[1:])
    def dens(d):
        d=d[(d>=lo)&(d<=hi)]; h,_=np.histogram(d,bins=edges); return h/max(h.sum(),1)/bw*100  # %/m
    ax.axvspan(lo,0,color=FAV,alpha=0.05); ax.axvspan(0,hi,color=NFV,alpha=0.05)
    perr=[BE[(BE.site==KEY[d])&(BE.error)]['signed_dist'].values for d in ORDER]
    M=np.vstack([dens(e) for e in perr]); band_lo=np.percentile(M,10,axis=0); band_hi=np.percentile(M,90,axis=0)
    ax.fill_between(cen,band_lo,band_hi,color='#cdd7e0',alpha=0.7,lw=0,label='per-site 10–90%')
    pooled=dens(BE[BE.error]['signed_dist'].values)
    ax.plot(cen,pooled,color=INK,lw=2.4,zorder=5,label='all sites (pooled)')
    ax.axvline(0,color='#444',ls='--',lw=1.0); ax.axhline(0,color='#ddd',lw=0.6)
    ax.set_xlim(lo,hi); ax.set_xlabel('signed distance to favela boundary (m)   —  favela $\\leftarrow$ | $\\rightarrow$ formal',fontsize=7.5)
    ax.set_ylabel('share of misclassified buildings (%/m)',fontsize=7.5)
    ax.set_title('Errors are graded at the boundary',fontsize=9,fontweight='bold',loc='left')
    if annotate:
        ax.text(0.97,0.93,'misclassifications collapse\nonto the favela edge\n(Rocinha 70.9% within 50 m)',transform=ax.transAxes,ha='right',va='top',fontsize=6.8,color='#555')
    ax.legend(loc='upper left',fontsize=6.6)

TNc="#dbe3ea"  # correct non-favela (light)
def spatial_proof(ax,site='catumbi',disp='Catumbi',inset=False):
    # Figure-S2 style per-building classification-outcome map (all buildings; errors hug the boundary)
    g=gpd.read_file(os.path.join(RES,site,f"{site}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label'])
    g=g[g.geometry.is_valid & ~g.geometry.is_empty]
    lab=g['label'].values; pred=g['predicted_label'].values
    tp=(lab=='favela')&(pred=='favela'); tn=(lab=='not-favela')&(pred=='not-favela'); err=(lab!=pred)
    g[tn].plot(ax=ax,color=TNc,ec='none',rasterized=True,zorder=1)      # correct non-favela
    g[tp].plot(ax=ax,color=FAV,ec='none',rasterized=True,zorder=2)      # correct favela
    if err.sum(): g[err].plot(ax=ax,color=ERRc,ec='none',rasterized=True,zorder=3)  # misclassified
    try:
        poly=gpd.read_file(os.path.join(PROJ,'Data','Limite_Favelas_2013','Limite_Favelas_2013.shp')).to_crs(g.crs)
        import shapely.geometry as sg; bb=g.total_bounds
        near=poly[poly.intersects(sg.box(*bb))]; near.boundary.plot(ax=ax,color='#111',lw=0.6,zorder=4)
    except Exception: pass
    ax.set_aspect('equal'); ax.axis('off')
    fs=6.5 if inset else 8.5
    ax.set_title(f'{disp}: classification outcome',fontsize=fs,fontweight='bold',loc='left',color=ERRc if inset else INK)
    if not inset:
        ax.legend(handles=[Patch(fc=FAV,label='favela (correct)'),Patch(fc=TNc,label='non-favela (correct)'),
            Patch(fc=ERRc,label='misclassified'),Line2D([0],[0],color='#111',lw=0.8,label='2013 favela polygon')],
            loc='lower left',fontsize=6.6,labelspacing=0.25)

def panel_letter(fig,x,y,t): fig.text(x,y,t,fontsize=13,fontweight='bold')

# ---------- V1: narrative four-beat ----------
def V1():
    fig=plt.figure(figsize=(7.2,6.2))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],width_ratios=[1,1.25],hspace=0.28,wspace=0.22)
    study_area(fig.add_subplot(gs[0,0]))
    lollipop(fig.add_subplot(gs[0,1]))
    boundary_decay(fig.add_subplot(gs[1,1]))
    spatial_proof(fig.add_subplot(gs[1,0]))
    for x,y,t in [(0.02,0.965,'a'),(0.44,0.965,'b'),(0.02,0.49,'c'),(0.44,0.49,'d')]: panel_letter(fig,x,y,t)
    fig.suptitle('Figure 2 (V1, narrative): from a sharp line to a graded edge',fontsize=10,y=0.995)
    fig.savefig('/tmp/fig2_v1.png',dpi=150,bbox_inches='tight');
    for e in('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig2_v1.{e}'),bbox_inches='tight',dpi=300)
    plt.close(fig); print('[V1] saved')

# ---------- V2: dashboard 2x2 ----------
def V2():
    fig=plt.figure(figsize=(7.4,6.6))
    gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.24)
    study_area(fig.add_subplot(gs[0,0]))
    bars(fig.add_subplot(gs[0,1]))
    boundary_decay(fig.add_subplot(gs[1,0]))
    spatial_proof(fig.add_subplot(gs[1,1]))
    for x,y,t in [(0.02,0.965,'a'),(0.52,0.965,'b'),(0.02,0.47,'c'),(0.52,0.47,'d')]: panel_letter(fig,x,y,t)
    fig.suptitle('Figure 2 (V2, dashboard): map · per-site metrics · boundary-error decay · spatial proof',fontsize=9.5,y=0.997)
    fig.savefig('/tmp/fig2_v2.png',dpi=150,bbox_inches='tight')
    for e in('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig2_v2.{e}'),bbox_inches='tight',dpi=300)
    plt.close(fig); print('[V2] saved')

# ---------- V3: minimal two-panel (map | decay + spatial inset) ----------
def V3():
    fig=plt.figure(figsize=(7.6,3.7))
    gs=fig.add_gridspec(1,2,width_ratios=[0.85,1.3],wspace=0.2)
    study_area(fig.add_subplot(gs[0,0]),title=True)
    axd=fig.add_subplot(gs[0,1]); boundary_decay(axd,annotate=True)
    iax=axd.inset_axes([0.60,0.40,0.38,0.56]); spatial_proof(iax,inset=True)
    fig.text(0.18,0.02,'10 sites · ~238k buildings · mean held-out test accuracy 93.3%',ha='center',fontsize=7,color='#555')
    for x,y,t in [(0.02,0.95,'a'),(0.42,0.95,'b')]: panel_letter(fig,x,y,t)
    fig.suptitle('Figure 2 (V3, minimal): study area + the boundary-error curve',fontsize=9.5,y=1.02)
    fig.savefig('/tmp/fig2_v3.png',dpi=150,bbox_inches='tight')
    for e in('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig2_v3.{e}'),bbox_inches='tight',dpi=300)
    plt.close(fig); print('[V3] saved')

V1(); V2(); V3(); print('done — 3 variants rendered')
