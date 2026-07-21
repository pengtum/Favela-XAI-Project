"""
Case-study figures (Supplementary S5/S6), styled to publication guidelines.
  A  Figure_X_casestudies_class.pdf/.svg  -- footprints by favela/non-favela (+ error outlines)
  B  Figure_X_casestudies_driver.pdf/.svg -- model-favela buildings by dominant SHAP driver + per-building SHAP
Pure visualization of stored SHAP + footprint geometry; no modelling.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- publication rcParams (editable text, Arial, clean spines) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.linewidth": 0.6,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
# restrained palette: one neutral family + signal/accent hues; red/violet reserved for error cues
C = dict(favela="#E28E2C", nonfav="#CFCECE", fp="#B64342", fn="#9A4D8E",
         mean_interbuilding_distance="#14746A", adjacency="#7FCBC0",
         perimeter_wall="#E2700E", squareness="#B5341F", other="#CFCECE",
         prednonfav="#ECECEC", pos="#E28E2C", neg="#0F4D92")
# per-feature colours = shades within the canonical 5-group palette (shared with the map figure)
FEATCOL = {
 # spatial distribution (teal)
 'mean_interbuilding_distance':'#14746A','adjacency':'#7FCBC0','CAR':'#2A9D8F','neighbour_dist':'#0E5249',
 # shape (warm)
 'perimeter_wall':'#E2700E','squareness':'#B5341F','cwa':'#F2B84B','compact_weighted_axis':'#9C5418',
 'shared_walls':'#8C4A28','rectangularity':'#D99441','shape_idx':'#C77A2E','convexity':'#A0511C',
 'facade_ratio':'#E8A05A','corners':'#7A3A12','rect_idx':'#D9742E','square_compact':'#C46A1C','elongation':'#F0A868',
 # connectivity (green)
 'avg_weighted_dist':'#4C9F70','fractal':'#1E7A45',
 # configuration (violet)
 'num_neighbors':'#9A4D8E','cell_alignment':'#C58EBC','tes_area':'#6F3566',
 # dimension (blue)
 'area':'#0F4D92','perimeter':'#5B8FC0','lal':'#2C6FB0',
}

PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES = os.path.join(PROJ, "results", "all_features"); OUT = os.path.dirname(os.path.abspath(__file__))
RAW = ["area","perimeter","lal","neighbour_dist","mean_interbuilding_distance","adjacency","corners",
       "shape_idx","facade_ratio","compact_weighted_axis","squareness","square_compact","rectangularity",
       "rect_idx","perimeter_wall","num_neighbors","elongation","fractal","cwa","CAR","tes_area",
       "shared_walls","convexity","cell_alignment","avg_weighted_dist"]
SH = ["shap_"+f for f in RAW]
PRETTY = {'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','area':'area','perimeter':'perimeter','shared_walls':'shared walls','num_neighbors':'no. neighbours','CAR':'cov-area ratio','tes_area':'tess. cell area','neighbour_dist':'neighbour dist.','shape_idx':'shape index','convexity':'convexity','elongation':'elongation','rectangularity':'rectangularity','corners':'corners','compact_weighted_axis':'compact. axis','cwa':'compact. axis (t)','fractal':'fractal','cell_alignment':'cell align.','avg_weighted_dist':'avg wt. dist.','lal':'longest axis','facade_ratio':'facade ratio','rect_idx':'rect. index','square_compact':'square compact'}
DRIVERS = ['mean_interbuilding_distance','adjacency','perimeter_wall','squareness']
CASES = [("a","Rocinha · boundary spillover",'rocinha',680475,7456125,190),
         ("b","Rocinha · within-site cluster",'rocinha',679783,7456979,320),
         ("c","Grotão de C.B. · perimeter wall",'grotao_de_costa_barros',666525,7474875,200),
         ("d","Grotão de C.B. · false negatives",'grotao_de_costa_barros',666825,7474575,190)]

def load(site):
    g = gpd.read_file(os.path.join(RES,site,f"{site}_updated_buildings_with_shap.gpkg"),
                      columns=['label','predicted_label']+RAW+SH)
    g = g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)
    g['_x']=g.geometry.centroid.x; g['_y']=g.geometry.centroid.y
    g['FP']=(g.label=='not-favela')&(g.predicted_label=='favela')
    g['FN']=(g.label=='favela')&(g.predicted_label=='not-favela')
    g['driver']=[RAW[i] for i in np.argmax(g[SH].values,axis=1)]
    return g
def win(g,cx,cy,r): return g[(g._x>cx-r)&(g._x<cx+r)&(g._y>cy-r)&(g._y<cy+r)]
def panel(ax,letter): ax.text(-0.02,1.02,letter,transform=ax.transAxes,fontsize=9,fontweight='bold',ha='right',va='bottom')
def scalebar(ax,length=100):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    x=x0+(x1-x0)*0.06; y=y0+(y1-y0)*0.07
    ax.plot([x,x+length],[y,y],color='k',lw=1.4,solid_capstyle='butt')
    ax.text(x+length/2,y+(y1-y0)*0.02,f'{length} m',ha='center',va='bottom',fontsize=5.5)
def save(fig,name):
    for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'{name}.{ext}'),bbox_inches='tight')
    fig.savefig(f'/tmp/{name}_preview.png',dpi=110,bbox_inches='tight'); plt.close(fig)

DATA = {s: load(s) for s in set(c[2] for c in CASES)}

# ===== Figure A : classification outcome (2x2) =====
figA, axA = plt.subplots(2,2,figsize=(7.2,6.6))
for ax,(L,title,site,cx,cy,r) in zip(axA.ravel(),CASES):
    w=win(DATA[site],cx,cy,r)
    w[w.label=='favela'].plot(ax=ax,color=C['favela'],ec='white',lw=.06)
    w[w.label=='not-favela'].plot(ax=ax,color=C['nonfav'],ec='white',lw=.06)
    if w.FP.any(): w[w.FP].boundary.plot(ax=ax,color=C['fp'],lw=.9)
    if w.FN.any(): w[w.FN].boundary.plot(ax=ax,color=C['fn'],lw=.9)
    ax.set_title(title,fontsize=7.5); ax.set_aspect('equal'); scalebar(ax); ax.axis('off'); panel(ax,L)
figA.legend(handles=[Patch(fc=C['favela'],label='favela (ground truth)'),Patch(fc=C['nonfav'],label='non-favela'),
                     Line2D([0],[0],color=C['fp'],lw=1.3,label='false positive (formal→favela)'),
                     Line2D([0],[0],color=C['fn'],lw=1.3,label='false negative (favela→formal)')],
            loc='lower center',ncol=4,fontsize=6.5,bbox_to_anchor=(0.5,-0.015))
figA.tight_layout(rect=[0,0.04,1,1]); save(figA,"Figure_X_casestudies_class")

# ===== Figure B : dominant driver + per-building SHAP (4x2) =====
def shapbar(ax,row,title):
    s=pd.Series([row['shap_'+f] for f in RAW],index=RAW).sort_values(key=np.abs,ascending=False).head(8)[::-1]
    ax.barh([PRETTY.get(f,f) for f in s.index],s.values,
            color=[FEATCOL.get(f,'#9e9e9e') for f in s.index],height=.66)
    ax.axvline(0,color='#767676',lw=.6); ax.tick_params(labelsize=6); ax.set_title(title,fontsize=7)
    ax.set_xlabel('SHAP   ($\\leftarrow$ formal | favela $\\rightarrow$)',fontsize=6)
present=set()
figB, axB = plt.subplots(4,2,figsize=(7.4,11.4),gridspec_kw={'width_ratios':[1.1,1]})
for i,(L,title,site,cx,cy,r) in enumerate(CASES):
    w=win(DATA[site],cx,cy,r); ax=axB[i,0]
    w[w.predicted_label=='not-favela'].plot(ax=ax,color=C['prednonfav'],ec='white',lw=.06)
    predfav=w[w.predicted_label=='favela']
    for d,sub in predfav.groupby('driver'):
        sub.plot(ax=ax,color=C.get(d,C['other']),ec='white',lw=.06); present.add(d)
    if w.FP.any(): w[w.FP].boundary.plot(ax=ax,color='#272727',lw=.55,ls=':')
    ax.set_title(title,fontsize=7.5); ax.set_aspect('equal'); scalebar(ax); ax.axis('off'); panel(ax,L)
    if i==0:   rep=w[w.FP].assign(t=w[w.FP][SH].sum(1)).sort_values('t',ascending=False).iloc[0]; rl='false positive (non-favela)'
    elif i==1: a=predfav[predfav.driver=='adjacency']; rep=(a if len(a) else predfav).assign(t=lambda d:d['shap_adjacency']).sort_values('t',ascending=False).iloc[0]; rl='adjacency-driven building'
    elif i==2: p=predfav[predfav.driver=='perimeter_wall']; rep=(p if len(p) else predfav).assign(t=lambda d:d['shap_perimeter_wall']).sort_values('t',ascending=False).iloc[0]; rl='perimeter-wall–driven building'
    else:      rep=w[w.FN].assign(t=w[w.FN][SH].sum(1)).sort_values('t').iloc[0]; rl='false negative (favela)'
    shapbar(axB[i,1],rep,f'{L}  ·  {rl}')
handles=[Patch(fc=C[d],label=PRETTY[d]) for d in DRIVERS if d in present]
if present-set(DRIVERS): handles.append(Patch(fc=C['other'],label='other feature'))
handles+=[Patch(fc=C['prednonfav'],label='predicted non-favela'),
          Line2D([0],[0],color='#272727',lw=.9,ls=':',label='false positive')]
figB.legend(handles=handles,loc='lower center',ncol=4,fontsize=6.3,bbox_to_anchor=(0.5,-0.008))
figB.tight_layout(rect=[0,0.03,1,1]); save(figB,"Figure_X_casestudies_driver")
print("drivers present:",sorted(present)); print("[saved] class + driver  (pdf, svg)")
