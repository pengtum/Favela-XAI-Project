"""Figure 1 — pipeline: building footprints + favela boundary -> train AI model -> XAI extracts the
morphological signature. Light-tone maps (Rocinha bottom-right district = Fig 3b), colours unified
with Fig 2/3. Vector frame + inline map axes; bottom thumbnails = real Fig 3/4/5."""
import os, warnings, numpy as np, geopandas as gpd, shapely.geometry as sg
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.colors as mc
import fitz
from PIL import Image
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "pdf.fonttype":42,"svg.fonttype":"none","font.size":9})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
INP="#2A9D8F"; MODELc="#E2700E"; XAIc="#9A4D8E"; SIGc="#C0504D"
FAV="#3F83A2"; FORMAL="#e8e8e8"
DIMCOL={'Dimension':'#0F4D92','Spatial distribution':'#2A9D8F'}
def seq(c): r,g_,b=mc.to_rgb(c); t=lambda a:(1-a+a*r,1-a+a*g_,1-a+a*b); return LinearSegmentedColormap.from_list("s",[t(0.10),t(0.5),c])
SHAPDIV=LinearSegmentedColormap.from_list("blue_orange",["#0F4D92","#F4F4F4","#E28E2C"])
def tint(h,a=0.05): r,g_,b=mc.to_rgb(h); return (1-a+a*r,1-a+a*g_,1-a+a*b)

# ---- data: Rocinha bottom-right district + 2013 boundary ----
print("loading ...")
FEATS=['area','adjacency','mean_interbuilding_distance']
cols=['label']+FEATS+['shap_'+f for f in FEATS]
g=gpd.read_file(os.path.join(RES,"rocinha","rocinha_updated_buildings_with_shap.gpkg"),columns=cols)
g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
ZB=(680290,7455500,680980,7456150); cx,cy=(ZB[0]+ZB[2])/2,(ZB[1]+ZB[3])/2; half=max(ZB[2]-ZB[0],ZB[3]-ZB[1])/2*1.04
SQB=(cx-half,cy-half,cx+half,cy+half)
c=g.geometry.centroid; g=g[(c.x>=SQB[0])&(c.x<=SQB[2])&(c.y>=SQB[1])&(c.y<=SQB[3])].reset_index(drop=True)
fav=(g['label']=='favela').values
POLY=gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp")).to_crs(g.crs)
zbox=sg.box(*SQB); BND=gpd.GeoSeries([POLY[POLY.intersects(zbox)].boundary.union_all().intersection(zbox)],crs=g.crs)
print(f"  district {len(g)} buildings ({fav.sum()} fav / {(~fav).sum()} formal)")

def mbase(ax): BND.plot(ax=ax,color='#1a1a1a',lw=1.0,zorder=6); ax.set_xlim(SQB[0],SQB[2]); ax.set_ylim(SQB[1],SQB[3]); ax.set_aspect('equal'); ax.axis('off')
def m_input(ax):
    g[~fav].plot(ax=ax,color=FORMAL,ec='white',lw=0.16); g[fav].plot(ax=ax,color=FAV,ec='white',lw=0.16); mbase(ax)
def m_feat(ax,col,dim):
    v=g[col].values.astype(float); lo,hi=np.nanpercentile(v,[2,98]); g.plot(ax=ax,column=col,cmap=seq(DIMCOL[dim]),vmin=lo,vmax=hi,ec='white',lw=0.16); mbase(ax)
def m_shap(ax,col):
    v=g['shap_'+col].values.astype(float); a=np.abs(v[np.isfinite(v)])
    m=float(np.percentile(a,98)) if a.size else 0.0
    if not np.isfinite(m) or m<=0: m=float(a.max()) if (a.size and a.max()>0) else 1.0
    g.plot(ax=ax,column='shap_'+col,cmap=SHAPDIV,norm=TwoSlopeNorm(vmin=-m,vcenter=0.0,vmax=m),ec='white',lw=0.16); mbase(ax)

# ---- bottom thumbnails: crop real Fig 3/4/5 ----
def crop(src,box,out,jpg=False):
    if jpg:
        im=Image.open(os.path.join(OUT,src)).convert('RGB')
    else:
        d=fitz.open(os.path.join(OUT,src)); pg=d[0]; pm=pg.get_pixmap(matrix=fitz.Matrix(2.4,2.4)); im=Image.frombytes('RGB',[pm.width,pm.height],pm.samples)
    W,H=im.size; im.crop((int(box[0]*W),int(box[1]*H),int(box[2]*W),int(box[3]*H))).save(os.path.join('/tmp',out))
crop('Figure_X_fig3_final.pdf',(0.0,0.04,0.50,0.40),'th_fig3.png')              # two dominant-feature maps
crop('Figure_X_spillover_all25_metrics.pdf',(0.0,0.06,0.42,0.40),'th_fig4.png') # a few SHAP-vs-distance panels
crop('Figure_X_Favela_DNA_new.jpg',(0.30,0.10,0.72,0.92),'th_fig5.png',jpg=True)# PC1-PC2 signature scatter

# ================= figure =================
FW,FH=13.8,8.4
fig=plt.figure(figsize=(FW,FH)); bg=fig.add_axes([0,0,1,1]); bg.set_xlim(0,100); bg.set_ylim(0,100); bg.axis('off')
def block(x,y,w,h,title,ec):
    bg.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=2.2",fc=tint(ec),ec=ec,lw=1.6))
    bg.add_patch(FancyBboxPatch((x+0.6,y+h-4.6),w-1.2,3.8,boxstyle="round,pad=0.1,rounding_size=1",fc=ec,ec='none'))
    bg.text(x+w/2,y+h-2.6,title,ha='center',va='center',fontsize=10,fontweight='bold',color='white')
def node(x,y,w,h,txt,ec,fs=8.0,bold=False):
    bg.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.22,rounding_size=1.1",fc='white',ec=ec,lw=1.1))
    bg.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color='#222',fontweight='bold' if bold else 'normal')
def arrow(x1,y1,x2,y2): bg.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=22,lw=2.6,color='#444',shrinkA=2,shrinkB=2))
def sq(x,y,w,h):  # square inline axes centered in 0-100 box region
    rw,rh=w/100*FW,h/100*FH; s=min(rw,rh); ccx,ccy=(x+w/2)/100,(y+h/2)/100
    return fig.add_axes([ccx-s/FW/2,ccy-s/FH/2,s/FW,s/FH])
def thumb(x,y,w,h,png):  # imshow a thumbnail PNG, aspect-preserved, into 0-100 box
    im=plt.imread(os.path.join('/tmp',png)); ar=im.shape[1]/im.shape[0]; rw,rh=w/100*FW,h/100*FH
    bw,bh=(rw,rw/ar) if rw/ar<=rh else (rh*ar,rh); ccx,ccy=(x+w/2)/100,(y+h/2)/100
    ax=fig.add_axes([ccx-bw/FW/2,ccy-bh/FH/2,bw/FW,bh/FH]); ax.imshow(im); ax.axis('off'); return ax

bg.text(50,98.4,'From building footprints + favela boundary, an explainable AI yields a building-scale morphological signature',ha='center',fontsize=11.5,fontweight='bold')

# Block 1 INPUT
block(1,40,30,53,'INPUT',INP)
node(2.5,82.5,27,4.2,'Open building footprints',INP); node(2.5,77,27,4.2,'2013 favela boundary',INP)
axI=sq(2,44,28,30); m_input(axI)
bg.text(16,42.3,'footprints + boundary  →  favela / formal',ha='center',fontsize=7.4,color='#555')
arrow(31.5,66,35.5,66)
# Block 2 AI MODEL
block(36,40,30,53,'TRAIN AI MODEL',MODELc)
node(37.5,82.5,27,4.2,'25 morphometric features (momepy)',MODELc,fs=7.6)
ax2a=sq(37.5,60,13,15); m_feat(ax2a,'area','Dimension'); bg.text(44,59,'building area',ha='center',fontsize=6.4,color='#555')
ax2b=sq(51,60,13,15); m_feat(ax2b,'mean_interbuilding_distance','Spatial distribution'); bg.text(57.5,59,'interbuild. dist.',ha='center',fontsize=6.4,color='#555')
node(37.5,51,27,5.0,'XGBoost classifier · favela vs formal',MODELc,fs=7.8,bold=True)
node(37.5,44.5,27,4.6,'held-out accuracy  mean 0.933',MODELc,fs=7.6)
arrow(66.5,66,70.5,66)
# Block 3 XAI
block(71,40,28,53,'EXPLAIN (XAI)',XAIc)
node(72.3,82.5,25.4,4.6,'Tree SHAP → per-building attribution',XAIc,fs=7.5)
ax3a=sq(72.3,60,12.2,15); m_shap(ax3a,'area'); bg.text(78.4,59,'SHAP · area',ha='center',fontsize=6.4,color='#555')
ax3b=sq(85,60,12.2,15); m_shap(ax3b,'mean_interbuilding_distance'); bg.text(91.1,59,'SHAP · interbuild.',ha='center',fontsize=6.4,color='#555')
node(72.3,49.5,25.4,6.2,'building-scale\nMORPHOLOGICAL SIGNATURE',XAIc,fs=8.2,bold=True)
arrow(85,40,85,35.5)
# shared colour key (under blocks)
bg.text(33.5,37.0,'Feature value: light → dimension colour (blue = dimension, teal = spatial)        SHAP: blue = formal ← 0 → orange = favela        — = 2013 favela boundary',
        ha='center',fontsize=6.8,color='#555')

# Bottom: THE SIGNATURE REVEALS
block(1,2,98,32,'THE SIGNATURE REVEALS',SIGc)
def out(x,t,png,fg):
    node(x,26,29,4.6,t,SIGc,fs=8.0,bold=True); thumb(x,8.5,29,15,png); bg.text(x+14.5,4.6,fg,ha='center',fontsize=8.8,fontweight='bold',color=SIGc)
out(3.5,'(i) Site-specific signatures','th_fig3.png','→  Figure 3')
out(35.5,'(ii) Graded boundary / spillover','th_fig4.png','→  Figure 4')
out(67.5,'(iii) 2D signature space','th_fig5.png','→  Figure 5')

for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_concept_redesign.{ext}'),bbox_inches='tight',dpi=300)
fig.savefig('/tmp/fig1_full.png',dpi=130,bbox_inches='tight'); print("[saved] Figure_X_concept_redesign.pdf + /tmp/fig1_full.png")
