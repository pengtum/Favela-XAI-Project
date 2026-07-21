"""
Main Figure 2 (favela detection), 2x2, with the per-site building COUNTS and the
per-site ACCURACY merged into one chart:
  a  study area (original map, embedded)
  b  per-site: stacked building counts (favela/non-favela)  +  accuracy dot, precision|recall ticks   [MERGED]
  c  classification outcome for two representative sites (Rocinha, Catumbi; cropped from S2)
  d  errors graded at the boundary (share of misclassified vs signed distance to the favela polygon)
Images (a, c) are embedded RGB; b, d are matplotlib. Caveat: boundary curve is a within-error
SHARE distribution (location of errors), not an in-sample rate.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Patch, Circle
OUT=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
# palette matched to the Supplementary S5 classification maps (panel b): cool favela so warm errors pop
FAV="#3F83A2"; NFV="#42A969"; ERR="#DE913D"; ERR2="#C0392B"; INK="#222222"; ACC="#2F4B5E"
METRICS={'Rocinha':(0.977,0.981,0.978,0.980,0.977),'Morro da Guaíba':(0.962,0.919,0.942,0.930,0.952),
 'Barão':(0.958,0.929,0.911,0.920,0.946),'Jacarezinho':(0.954,0.940,0.929,0.934,0.949),
 'Faz Quem Quer':(0.949,0.906,0.914,0.910,0.937),'Catumbi':(0.934,0.913,0.921,0.917,0.931),
 'Palmeiras':(0.931,0.839,0.884,0.861,0.907),'Fazenda Coqueiro':(0.920,0.935,0.943,0.939,0.911),
 'Nova Cidade':(0.881,0.767,0.811,0.788,0.853),'Grotão de C.B.':(0.866,0.834,0.871,0.852,0.865)}
ORDER=list(METRICS); MEANACC=0.933
KEY={'Rocinha':'rocinha','Morro da Guaíba':'morro_da_guaiba','Barão':'barao','Jacarezinho':'jacarezinho','Faz Quem Quer':'faz_quem_quer','Catumbi':'catumbi','Palmeiras':'palmeras','Fazenda Coqueiro':'fazena_coqueiro','Nova Cidade':'nova_cidade','Grotão de C.B.':'grotao_de_costa_barros'}
CENT=pd.read_csv(os.path.join(OUT,'fig2_site_centroids.csv')).set_index('disp')
BE=pd.read_parquet(os.path.join(OUT,'boundary_error_data.parquet'))
import matplotlib.patheffects as pe
RIOm=gpd.read_file(os.path.join(OUT,'rio_detailed.geojson')).to_crs(3857)   # detailed municipality boundary (osmnx)
CENTg=gpd.GeoDataFrame(CENT.reset_index(),geometry=gpd.points_from_xy(CENT.lon,CENT.lat),crs=4326).to_crs(3857)
NMAX=np.sqrt(max(CENT.n_fav.max(),CENT.n_non.max()))
BMAP=np.load(os.path.join(OUT,'_fig2_basemap.npy')); _m=np.load(os.path.join(OUT,'_fig2_basemap_meta.npy'))
BEXT=(_m[0],_m[1],_m[2],_m[3]); MAPX=(_m[4],_m[5]); MAPY=(_m[6],_m[7]); RIO_AR=(MAPX[1]-MAPX[0])/(MAPY[1]-MAPY[0])
HALO=[pe.withStroke(linewidth=2.7,foreground='white')]
def halfcircle_map(ax,Rmax=5200):
    ax.imshow(BMAP,extent=BEXT,zorder=0,interpolation='bilinear')                 # OpenStreetMap (CartoDB Positron) basemap
    RIOm.boundary.plot(ax=ax,color='#6f6f6f',lw=0.9,zorder=1,alpha=0.85)          # detailed Rio boundary
    POS={'Grotão de C.B.':'left','Morro da Guaíba':'above','Faz Quem Quer':'right','Catumbi':'right','Jacarezinho':'below','Barão':'left'}  # nudge crowded labels apart
    for _,r in CENTg.iterrows():
        x,y=r.geometry.x,r.geometry.y; disp=r['disp']
        rf=Rmax*np.sqrt(r.n_fav)/NMAX; rn=Rmax*np.sqrt(r.n_non)/NMAX; rr=max(rf,rn); off=600
        ax.add_patch(Wedge((x,y),rf,90,270,fc=FAV,ec='white',lw=0.7,zorder=4))    # left half = favela
        ax.add_patch(Wedge((x,y),rn,270,90,fc=NFV,ec='white',lw=0.7,zorder=4))    # right half = non-favela
        p=POS.get(disp,'below')
        if   p=='below': tx,ty,ha,va=x,y-rr-off,'center','top'
        elif p=='above': tx,ty,ha,va=x,y+rr+off,'center','bottom'
        elif p=='left':  tx,ty,ha,va=x-rf-off,y,'right','center'
        else:            tx,ty,ha,va=x+rn+off,y,'left','center'
        ax.text(tx,ty,disp,ha=ha,va=va,fontsize=8.6,color='#1a1a1a',zorder=5,path_effects=HALO)
    ax.set_xlim(*MAPX); ax.set_ylim(*MAPY); ax.set_aspect('equal'); ax.axis('off')
    # graduated size key (lower-left, over open water): half-disk RADIUS encodes per-class building count (area ∝ count)
    refs=[5000,15000,30000]; lx=MAPX[0]+(MAPX[1]-MAPX[0])*0.085; lyb=MAPY[0]+(MAPY[1]-MAPY[0])*0.075
    rbig=Rmax*np.sqrt(refs[-1])/NMAX
    for c in refs:
        rr=Rmax*np.sqrt(c)/NMAX
        ax.add_patch(Circle((lx,lyb+rr),rr,fill=False,ec='#6f6f6f',lw=0.8,zorder=4))
        ax.plot([lx,lx+rbig+900],[lyb+2*rr,lyb+2*rr],color='#9a9a9a',lw=0.5,zorder=4)
        ax.text(lx+rbig+1050,lyb+2*rr,f'{c:,}',ha='left',va='center',fontsize=6,color='#444',zorder=5,path_effects=HALO)
    ax.text(lx,lyb+2*rbig+1500,'buildings per class',ha='center',va='bottom',fontsize=6.3,color='#222',zorder=5,path_effects=HALO,fontweight='bold')
    ax.legend(handles=[Patch(fc=FAV,label='favela (left half)'),Patch(fc=NFV,label='non-favela (right half)')],loc='upper right',fontsize=7,title='ground-truth class',title_fontsize=6.8,frameon=True,framealpha=0.9,facecolor='white',edgecolor='#cccccc')

def accuracy(ax):   # per-site held-out test accuracy only (clean Nature dot plot, sorted high->low)
    y=np.arange(len(ORDER))[::-1]; accs=[METRICS[d][0] for d in ORDER]
    ax.hlines(y,0.85,accs,color='#dfe2e6',lw=1.7,zorder=1)
    ax.scatter(accs,y,s=54,c=ACC,ec='white',lw=0.7,zorder=3)
    for yi,a in zip(y,accs): ax.text(a+0.004,yi,f'{a:.3f}',va='center',ha='left',fontsize=6.6,color='#555',zorder=4)
    ax.axvline(MEANACC,color='#9aa0a6',ls='--',lw=0.9,zorder=2)
    ax.text(MEANACC,len(ORDER)-0.1,f'mean {MEANACC:.3f}',fontsize=6.6,color='#777',ha='center',va='bottom')
    ax.set_xlim(0.85,1.012); ax.set_ylim(-0.6,len(ORDER)-0.05)
    ax.set_yticks(y); ax.set_yticklabels(ORDER,fontsize=8); ax.set_xticks([0.85,0.90,0.95,1.00])
    ax.set_xlabel('held-out test accuracy',fontsize=8.5); ax.tick_params(axis='x',labelsize=7.4); ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_color('#bbbbbb')

def boundary_decay(ax,lo=-90,hi=90,bw=6):
    edges=np.arange(lo,hi+bw,bw); cen=0.5*(edges[:-1]+edges[1:])
    def dens(d):
        d=d[(d>=lo)&(d<=hi)]; h,_=np.histogram(d,bins=edges); return h/max(h.sum(),1)/bw*100
    ax.axvspan(lo,0,color=FAV,alpha=0.05); ax.axvspan(0,hi,color=NFV,alpha=0.05)
    M=np.vstack([dens(BE[(BE.site==KEY[d])&(BE.error)]['signed_dist'].values) for d in ORDER])
    ax.fill_between(cen,np.percentile(M,10,axis=0),np.percentile(M,90,axis=0),color='#cdd7e0',alpha=0.7,lw=0,label='per-site 10–90%')
    ax.plot(cen,dens(BE[BE.error]['signed_dist'].values),color=INK,lw=2.3,zorder=5,label='all sites (pooled)')
    ax.axvline(0,color='#444',ls='--',lw=1.0)
    ax.set_xlim(lo,hi); ax.set_xlabel('signed distance to favela boundary (m)  —  favela $\\leftarrow$ | $\\rightarrow$ formal',fontsize=7.3)
    ax.set_ylabel('share of misclassified buildings (%/m)',fontsize=7.3)
    ax.text(0.97,0.94,'misclassifications collapse\nonto the favela edge\n(Rocinha 70.9% within 50 m)',transform=ax.transAxes,ha='right',va='top',fontsize=6.6,color='#555')
    ax.legend(loc='upper left',fontsize=6.5)

# 3-row layout, rows aspect-matched so a / b / (c,d) share the same left & right edges:
#   a half-disk study-area map | b two classification maps (Barão, Morro) | c accuracy + d boundary curve
_t=plt.imread(os.path.join(OUT,'_cand_FazQuemQuer.png')); TILE_AR=2*_t.shape[1]/_t.shape[0]   # two tiles side by side (w/h)
W=7.0; HR=[1/RIO_AR,1/TILE_AR,0.50]
fig=plt.figure(figsize=(W,W*sum(HR)+0.9))
gs=fig.add_gridspec(3,1,height_ratios=HR,hspace=0.11,left=0.012,right=0.988,top=0.965,bottom=0.045)
# a: half-disk study-area map
halfcircle_map(fig.add_subplot(gs[0]))
fig.axes[-1].set_title('Study area: ten favelas across Rio de Janeiro',fontsize=9.5,fontweight='bold',loc='left')
# b: two classification maps, full width
gb=gs[1].subgridspec(1,2,wspace=0.04)
for cell,img,nm in [(gb[0,0],'_cand_FazQuemQuer.png','Faz Quem Quer'),(gb[0,1],'_cand_Morro.png','Morro da Guaíba')]:
    a=fig.add_subplot(cell); a.imshow(plt.imread(os.path.join(OUT,img))); a.axis('off'); a.set_title(nm,fontsize=9.5,fontweight='bold')
# c: per-site accuracy (left)   d: boundary errors (right)
gc=gs[2].subgridspec(1,2,width_ratios=[1.0,1.12],wspace=0.2)
axacc=fig.add_subplot(gc[0]); accuracy(axacc); axacc.set_title('Per-site detection performance',fontsize=9.5,fontweight='bold',loc='left')
axd=fig.add_subplot(gc[1]); boundary_decay(axd); axd.set_title('Errors are graded at the boundary',fontsize=9.5,fontweight='bold',loc='left')
# panel letters (positions derived from the gridspec rows)
rows=[gs[0].get_position(fig),gs[1].get_position(fig),gs[2].get_position(fig)]
for (cell,t) in [(rows[0],'a'),(rows[1],'b'),(rows[2],'c')]: fig.text(0.004,cell.y1-0.012,t,fontsize=14,fontweight='bold')
fig.text(0.5,rows[2].y1-0.012,'d',fontsize=14,fontweight='bold')
fig.legend(handles=[Patch(fc=FAV,label='favela — correct'),Patch(fc=NFV,label='non-favela — correct'),
    Patch(fc=ERR,label='misclassified (orange / red)'),Line2D([0],[0],color='k',lw=1.2,label='2013 favela boundary')],
    loc='center',ncol=4,fontsize=7,bbox_to_anchor=(0.5,rows[1].y0-0.006),frameon=False,columnspacing=1.7,handletextpad=0.5)
fig.savefig('/tmp/fig2_v2.png',dpi=120,bbox_inches='tight')
for e in('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_fig2_merged.{e}'),bbox_inches='tight',dpi=300)
plt.close(fig); print('[saved] Figure_X_fig2_merged.pdf  aspect~',round(7.4/9.2,2))
