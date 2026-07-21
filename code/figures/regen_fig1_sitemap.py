"""Figure-1 element: study-area half-disk map (10 favelas across Rio), extracted from
regen_fig2_v2.py's halfcircle_map. Colours unified with the other Fig-1 maps
(favela blue / formal-settlement green); legend relabelled to 'favela (informal)' /
'formal settlement'. Clean (no title) vector PDF + PNG for manual assembly."""
import os, numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.patches import Wedge, Patch, Circle
OUT=os.path.dirname(os.path.abspath(__file__))
OUTLINE=False   # True = all text -> vector OUTLINES (转曲: font-independent vector shapes); False = live editable Helvetica text
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"path" if OUTLINE else "none","font.size":8})
if not OUTLINE: plt.rcParams["pdf.use14corefonts"]=True   # editable mode = non-embedded base-14 Helvetica
FAV="#3F83A2"; NFV="#42A969"                                   # favela blue / formal-settlement green (= Fig 2 class colours)
CENT=pd.read_csv(os.path.join(OUT,'fig2_site_centroids.csv')).set_index('disp')
RIOm=gpd.read_file(os.path.join(OUT,'rio_detailed.geojson')).to_crs(3857)
CENTg=gpd.GeoDataFrame(CENT.reset_index(),geometry=gpd.points_from_xy(CENT.lon,CENT.lat),crs=4326).to_crs(3857)
NMAX=np.sqrt(max(CENT.n_fav.max(),CENT.n_non.max()))
BMAP=np.load(os.path.join(OUT,'_fig2_basemap.npy')); _m=np.load(os.path.join(OUT,'_fig2_basemap_meta.npy'))
BEXT=(_m[0],_m[1],_m[2],_m[3]); MAPX=(_m[4],_m[5]); MAPY=(_m[6],_m[7]); RIO_AR=(MAPX[1]-MAPX[0])/(MAPY[1]-MAPY[0])
HALO=[pe.withStroke(linewidth=2.7,foreground='white')]
LBL=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.72)  # editable-text-safe legibility: white bbox INSTEAD of a path-effect stroke (path effects outline glyphs -> non-editable text)

def halfcircle_map(ax,Rmax=5200):
    ax.imshow(BMAP,extent=BEXT,zorder=0,interpolation='bilinear')                 # CartoDB Positron (light) basemap
    RIOm.boundary.plot(ax=ax,color='#6f6f6f',lw=0.9,zorder=1,alpha=0.85)
    POS={'Grotão de C.B.':'left','Morro da Guaíba':'above','Faz Quem Quer':'right','Catumbi':'right','Jacarezinho':'below','Barão':'left'}
    for _,r in CENTg.iterrows():
        x,y=r.geometry.x,r.geometry.y; disp=r['disp']
        rf=Rmax*np.sqrt(r.n_fav)/NMAX; rn=Rmax*np.sqrt(r.n_non)/NMAX; rr=max(rf,rn); off=600
        ax.add_patch(Wedge((x,y),rf,90,270,fc=FAV,ec='white',lw=0.7,zorder=4))    # left half = favela
        ax.add_patch(Wedge((x,y),rn,270,90,fc=NFV,ec='white',lw=0.7,zorder=4))    # right half = formal
        p=POS.get(disp,'below')
        if   p=='below': tx,ty,ha,va=x,y-rr-off,'center','top'
        elif p=='above': tx,ty,ha,va=x,y+rr+off,'center','bottom'
        elif p=='left':  tx,ty,ha,va=x-rf-off,y,'right','center'
        else:            tx,ty,ha,va=x+rn+off,y,'left','center'
        ax.text(tx,ty,disp,ha=ha,va=va,fontsize=8.6,color='#1a1a1a',zorder=5,bbox=LBL)
    ax.set_xlim(*MAPX); ax.set_ylim(*MAPY); ax.set_aspect('equal'); ax.axis('off')
    # graduated size key: half-disk RADIUS encodes per-class building count (area proportional to count)
    refs=[5000,15000,30000]; lx=MAPX[0]+(MAPX[1]-MAPX[0])*0.085; lyb=MAPY[0]+(MAPY[1]-MAPY[0])*0.075
    rbig=Rmax*np.sqrt(refs[-1])/NMAX
    for c in refs:
        rr=Rmax*np.sqrt(c)/NMAX
        ax.add_patch(Circle((lx,lyb+rr),rr,fill=False,ec='#6f6f6f',lw=0.8,zorder=4))
        ax.plot([lx,lx+rbig+900],[lyb+2*rr,lyb+2*rr],color='#9a9a9a',lw=0.5,zorder=4)
        ax.text(lx+rbig+1050,lyb+2*rr,f'{c:,}',ha='left',va='center',fontsize=6,color='#444',zorder=5,bbox=LBL)
    ax.text(lx,lyb+2*rbig+1500,'buildings per class',ha='center',va='bottom',fontsize=6.3,color='#222',zorder=5,bbox=LBL,fontweight='bold')
    ax.legend(handles=[Patch(fc=FAV,label='favela (informal) - left half'),Patch(fc=NFV,label='formal settlement - right half')],
              loc='upper right',fontsize=7,title='ground-truth class',title_fontsize=6.8,frameon=True,framealpha=0.9,facecolor='white',edgecolor='#cccccc')

fig,ax=plt.subplots(figsize=(6.0,6.0/RIO_AR))
halfcircle_map(ax)
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
if OUTLINE:
    import matplotlib.text as mtext
    for _t in fig.findobj(mtext.Text): _t.set_path_effects([pe.Normal()])   # outline every glyph -> pure vector paths (no embedded font)
fig.savefig(os.path.join(OUT,'fig1_map_sites.svg'),bbox_inches='tight')   # SVG = live <text> (svg.fonttype none) -> fully editable type in Illustrator
fig.savefig(os.path.join(OUT,'fig1_map_sites.pdf'),bbox_inches='tight')
fig.savefig(os.path.join(OUT,'fig1_map_sites.png'),bbox_inches='tight',dpi=300)
print("[saved] fig1_map_sites .svg (editable text) + .pdf + .png")
