"""Light-tone (white-background) building maps for the Figure-1 pipeline redesign,
matching the Fig 2/3 visual language (NOT the old black-background version).
Same site + same three features as the CURRENT Figure 1: building area, adjacency,
mean interbuilding distance — each as a feature-value map AND a Tree-SHAP map.
Saved individually as vector PDF + a combined preview PNG."""
import os, warnings, numpy as np, geopandas as gpd, shapely.geometry as sg
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe, matplotlib.text as mtext
OUTLINE=True   # True = all text -> vector OUTLINES (转曲, matches fig1_map_sites); False = editable base-14 Helvetica text
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"path" if OUTLINE else "none","font.size":8})
if not OUTLINE: plt.rcParams["pdf.use14corefonts"]=True   # editable mode = non-embedded base-14 Helvetica
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
SITE="rocinha"; DISP="Rocinha"                                   # current Figure-1 region (large iconic favela)
# colours unified with the other main figures: Fig 2 favela class + Fig 3 grey non-favela / dimension palette / blue-orange SHAP
FAV="#3F83A2"; FORMAL="#42A969"                                  # favela blue / formal-settlement green (both = Fig 2 class colours)
GBASE={'Dimension':'#0F4D92','Shape':'#E2700E','Spatial distribution':'#2A9D8F','Configuration':'#9A4D8E','Connectivity':'#4C9F70'}  # GBASE dimension palette (= Fig 3/4); used ONLY for the small per-map dimension tag
def seq(c):
    import matplotlib.colors as mc; rr,gg,bb=mc.to_rgb(c); t=lambda a:(1-a+a*rr,1-a+a*gg,1-a+a*bb)
    return LinearSegmentedColormap.from_list("s",[t(0.10),t(0.5),c])
SHAPDIV=LinearSegmentedColormap.from_list("blue_orange",["#0F4D92","#F4F4F4","#E28E2C"])  # = Fig 3 SHAP map CMAP
FEATCMAP=LinearSegmentedColormap.from_list("red_fig1",["#fcdccd","#f4926e","#cb2418"])  # (kept as fallback) single light-RED value-ramp
def seqdim(c):  # per-dimension building ramp: visible light tint -> full GBASE dimension colour
    import matplotlib.colors as mc; rr,gg,bb=mc.to_rgb(c); t=lambda a:(1-a+a*rr,1-a+a*gg,1-a+a*bb)
    return LinearSegmentedColormap.from_list("sd",[t(0.16),t(0.52),c])
FEATS=[("area","Building area",'Dimension'),("elongation","Elongation",'Shape'),
       ("mean_interbuilding_distance","Mean interbuilding distance",'Spatial distribution'),("num_neighbors","No. of neighbours",'Configuration'),
       ("avg_weighted_dist","Avg. weighted distance",'Connectivity')]

print("loading",SITE,"...")
RAW25=["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners","shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity","rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area","shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH25=['shap_'+f for f in RAW25]
cols=['label']+[f for f,_,_ in FEATS]+SH25
g=gpd.read_file(os.path.join(RES,SITE,f"{SITE}_updated_buildings_with_shap.gpkg"),columns=cols)
g=g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
# focus on the BOTTOM-RIGHT sub-block of the Fig-3b Rocinha district (tighter zoom, clearer buildings)
ZB=(680290,7455500,680980,7456150)
def square_bounds(b,pad=0.04):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; half=max(b[2]-b[0],b[3]-b[1])/2*(1+pad); return (cx-half,cy-half,cx+half,cy+half)
SQB=square_bounds(ZB)
c=g.geometry.centroid; sel=(c.x>=SQB[0])&(c.x<=SQB[2])&(c.y>=SQB[1])&(c.y<=SQB[3])
g=g[sel].reset_index(drop=True); fav=(g['label']=='favela').values
print(f"  Rocinha bottom-right district (= Fig 3b): {len(g)} buildings ({fav.sum()} favela / {(~fav).sum()} formal)")
# PC1 of the 25-dim SHAP vector = the building-scale morphological SIGNATURE axis (oriented so favela = positive)
X=np.nan_to_num(g[SH25].values.astype(float)); Xs=(X-X.mean(0))/(X.std(0)+1e-9)
pc1=Xs@np.linalg.svd(Xs,full_matrices=False)[2][0]
if pc1[fav].mean()<pc1[~fav].mean(): pc1=-pc1
g['pc1']=pc1; print(f"  PC1 of SHAP signature computed (favela mean {pc1[fav].mean():+.2f} / formal {pc1[~fav].mean():+.2f})")
# official 2013 favela boundary clipped to the view = the line that assigns favela/formal labels (drawn on every map, as in Fig 3b)
POLY=gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp")).to_crs(g.crs)
zbox=sg.box(SQB[0],SQB[1],SQB[2],SQB[3])
BND=gpd.GeoSeries([POLY[POLY.intersects(zbox)].boundary.union_all().intersection(zbox)],crs=g.crs)

def base(ax):
    BND.plot(ax=ax,color='#1a1a1a',lw=1.1,zorder=6)
    ax.set_xlim(SQB[0],SQB[2]); ax.set_ylim(SQB[1],SQB[3]); ax.set_aspect('equal'); ax.axis('off')
def cbar(ax,sm,label,diverging=False):
    cb=plt.colorbar(sm,ax=ax,orientation='horizontal',fraction=0.046,pad=0.02,aspect=22)
    cb.set_label(label,fontsize=6.6); cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.5)
    cb.set_ticks([sm.norm.vmin,0,sm.norm.vmax] if diverging else [sm.norm.vmin,sm.norm.vmax])
def render_input(ax):
    g[~fav].plot(ax=ax,color=FORMAL,ec='white',lw=0.22); g[fav].plot(ax=ax,color=FAV,ec='white',lw=0.22)
    base(ax); ax.set_title('Input · building footprints + labels',fontsize=8.3,fontweight='bold')
    ax.legend(handles=[Patch(fc=FAV,label='favela (informal)'),Patch(fc=FORMAL,label='formal settlement'),
              Line2D([0],[0],color='#1a1a1a',lw=1.1,label='2013 favela boundary (land-tenure line)')],
              loc='lower left',fontsize=6.0,frameon=False)
def render_feat(ax,col,title,dim):
    cmap=seqdim(GBASE[dim]); v=g[col].values.astype(float); lo,hi=np.nanpercentile(v,[2,98])
    g.plot(ax=ax,column=col,cmap=cmap,vmin=lo,vmax=hi,ec='white',lw=0.22)
    base(ax); ax.set_title('Feature · '+title,fontsize=8.3,fontweight='bold')
    ax.scatter([0.06],[0.94],s=60,c=[GBASE[dim]],transform=ax.transAxes,clip_on=False,zorder=10,edgecolors='white',linewidths=0.6)
    ax.text(0.115,0.94,dim,transform=ax.transAxes,fontsize=6.6,va='center',ha='left',color=GBASE[dim],fontweight='bold',zorder=10,
            bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.65))
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(lo,hi)); cbar(ax,sm,'feature value (low to high)')
def render_shap(ax,col,title):
    v=g['shap_'+col].values.astype(float); a=np.abs(v[np.isfinite(v)])
    m=float(np.percentile(a,98)); m=m if m>0 else float(a.max() or 1.0)
    norm=TwoSlopeNorm(vmin=-m,vcenter=0.0,vmax=m)
    g.plot(ax=ax,column='shap_'+col,cmap=SHAPDIV,norm=norm,ec='white',lw=0.22)
    base(ax); ax.set_title('SHAP · '+title,fontsize=8.3,fontweight='bold')
    sm=plt.cm.ScalarMappable(cmap=SHAPDIV,norm=norm); cbar(ax,sm,'SHAP  (formal <- 0 -> favela)',diverging=True)
def render_pc1(ax):
    v=g['pc1'].values; a=np.abs(v[np.isfinite(v)]); m=float(np.percentile(a,98)); m=m if m>0 else 1.0
    norm=TwoSlopeNorm(vmin=-m,vcenter=0.0,vmax=m)
    g.plot(ax=ax,column='pc1',cmap=SHAPDIV,norm=norm,ec='white',lw=0.22)
    base(ax); ax.set_title('Signature · PC1 of SHAP',fontsize=8.3,fontweight='bold')
    sm=plt.cm.ScalarMappable(cmap=SHAPDIV,norm=norm); cbar(ax,sm,'PC1  (formal <- 0 -> favela signature)',diverging=True)

# individual vector PDFs (input + 3 feature + 3 SHAP + PC1 signature)
def save_one(name,plot):
    fig,ax=plt.subplots(figsize=(3.0,3.3)); plot(ax); ax.set_title('')   # no title: clean map for manual assembly
    if OUTLINE:
        for _t in fig.findobj(mtext.Text): _t.set_path_effects([pe.Normal()])   # outline glyphs -> pure vector paths (font-independent)
    fig.savefig(os.path.join(OUT,f'fig1_map_{name}.svg'),bbox_inches='tight')
    fig.savefig(os.path.join(OUT,f'fig1_map_{name}.pdf'),bbox_inches='tight')
    fig.savefig(os.path.join(OUT,f'fig1_map_{name}.png'),bbox_inches='tight',dpi=300)
    plt.close(fig)
save_one('input',render_input)
for col,title,dim in FEATS:
    save_one('feat_'+col,lambda ax,c=col,t=title,d=dim:render_feat(ax,c,t,d))
    save_one('shap_'+col,lambda ax,c=col,t=title:render_shap(ax,c,t))
save_one('pc1',render_pc1)
# combined preview: row1 = input + N feature (grey + dim tag), row2 = N SHAP + PC1 signature
n=len(FEATS)
fig,axes=plt.subplots(2,n+1,figsize=(3.0*(n+1),6.6)); axf=axes.ravel()
render_input(axf[0])
for i,(col,title,dim) in enumerate(FEATS): render_feat(axf[1+i],col,title,dim)
for i,(col,title,dim) in enumerate(FEATS): render_shap(axf[(n+1)+i],col,title)
render_pc1(axf[(n+1)+n])
fig.suptitle(f'Figure-1 maps — {DISP} (Fig 3b sub-block); 4 GREY feature maps + GBASE dimension tag; SHAP/PC1 = reference only (NOT for Fig 1)',fontsize=10,fontweight='bold',y=1.0)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig('/tmp/fig1_maps_preview.png',dpi=125,bbox_inches='tight'); plt.close(fig)
print(f"[saved] fig1_map_input + {n} feat + {n} shap + pc1  (+ /tmp/fig1_maps_preview.png)")
