"""
Failure-driver heatmap (Supplementary S4), styled to publication guidelines.
Visualizes stored SHAP only (mean SHAP among false positives, per feature per
site); no modelling. Loads precomputed per-site FP SHAP means.
"""
import os, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.6, "legend.frameon": False,
})
PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
OUT = os.path.dirname(os.path.abspath(__file__))
# sites ordered by per-site TEST ACCURACY high->low (Supplementary Table S1 / tab:site_metrics_test), matching main-text Fig 2b
SITES = ["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP = {'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}
PRETTY = {'mean_interbuilding_distance':'interbuild. dist.','adjacency':'adjacency','perimeter_wall':'perimeter wall','squareness':'squareness','area':'area','perimeter':'perimeter','shared_walls':'shared walls','num_neighbors':'no. neighbours','CAR':'cov-area ratio','tes_area':'tess. cell area','neighbour_dist':'neighbour dist.','shape_idx':'shape index','convexity':'convexity','elongation':'elongation','rectangularity':'rectangularity','corners':'corners','compact_weighted_axis':'compact. axis','cwa':'compact. axis (t)','fractal':'fractal','cell_alignment':'cell align.','avg_weighted_dist':'avg wt. dist.','lal':'longest axis','facade_ratio':'facade ratio','rect_idx':'rect. index','square_compact':'square compact'}
# blue (away) -> white -> orange (toward favela): matches the favela=orange convention
CMAP = LinearSegmentedColormap.from_list("blue_orange", ["#0F4D92","#F4F4F4","#E28E2C"])

out = pickle.load(open(os.path.join(PROJ,"results_stats","_failure_saved_shap.pkl"),"rb"))
order = SITES
top = set()
for s in SITES: top.update(out[s]['fp_shap'].nlargest(3).index)
feats = sorted(top, key=lambda f:-np.nanmean([out[s]['fp_shap'].get(f,np.nan) for s in SITES]))
M = np.array([[out[s]['fp_shap'].get(f,np.nan) for s in order] for f in feats])
vmax = np.nanmax(np.abs(M))

fig, ax = plt.subplots(figsize=(7.0,4.2))
im = ax.imshow(M, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(order))); ax.set_xticklabels([DISP[s] for s in order], rotation=35, ha='right', fontsize=6.5)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([PRETTY.get(f,f) for f in feats], fontsize=6.5)
ax.set_xticks(np.arange(-.5,len(order),1), minor=True); ax.set_yticks(np.arange(-.5,len(feats),1), minor=True)
ax.grid(which='minor', color='white', lw=0.8); ax.tick_params(which='both', length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for j,s in enumerate(order):
    col=[out[s]['fp_shap'].get(f,np.nan) for f in feats]; i=int(np.nanargmax(col))
    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec='#272727',lw=1.4))
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
cb.set_label('mean SHAP among false positives\n(positive → favela)', fontsize=6.5)
cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
ax.set_title('Feature driving false-positive (formal→favela) errors, by site',
             fontsize=7.5, pad=8, loc='left')
fig.tight_layout()
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'Figure_X_failure_shap.{ext}'),bbox_inches='tight')
fig.savefig('/tmp/failure_shap_preview.png',dpi=110,bbox_inches='tight')
print("[saved] Figure_X_failure_shap.pdf/.svg")
