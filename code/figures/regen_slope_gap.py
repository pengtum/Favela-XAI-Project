"""Re-make Supplementary Fig (slope gap vs per-site accuracy) — cleaner, publication style.
Data: Final analysis/terrain analysis/experiments/exp1_global_site_summary.csv (10 sites)."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "pdf.fonttype":42,"font.size":9,"axes.linewidth":0.8})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
CSV=os.path.join(PROJ,"Final analysis/terrain analysis/experiments/exp1_global_site_summary.csv")
OUT=os.path.join(PROJ,"Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/Figures/exp1_fig3_slope_gap_vs_accuracy.pdf")
DISP={'Barao':'Barão','Morro da Guaiba':'Morro da Guaíba','Grotao':'Grotão','Palmeras':'Palmeiras'}
FAV="#3F83A2"
d=pd.read_csv(CSV); d['name']=d['site'].map(lambda s:DISP.get(s,s))
rho,p=stats.spearmanr(d['slope_gap'],d['accuracy'])
print(f"Spearman rho={rho:.3f} p={p:.3f} (n={len(d)})")

fig,ax=plt.subplots(figsize=(6.3,5.0))
# faint reference trend (non-significant -> light grey, de-emphasised)
z=np.polyfit(d['slope_gap'],d['accuracy'],1); xl=np.linspace(d['slope_gap'].min()-0.2,d['slope_gap'].max()+0.45,50)
ax.plot(xl,np.poly1d(z)(xl),'--',color='#c2c2c2',lw=1.3,zorder=1)
ax.scatter(d['slope_gap'],d['accuracy'],s=130,c=FAV,ec='white',lw=1.0,zorder=5)
# per-site label placement (dx,dy in points; ha; leader=draw a thin connector for crowded ones)
LAB={'Rocinha':(-9,3,'right',0),'Faz Quem Quer':(11,0,'left',0),'Fazenda Coqueiro':(11,3,'left',0),
     'Barao':(10,6,'left',0),'Catumbi':(11,-1,'left',0),'Nova Cidade':(11,1,'left',0),'Grotao':(11,1,'left',0),
     'Morro da Guaiba':(46,22,'left',1),'Jacarezinho':(10,-17,'left',1),'Palmeras':(42,2,'left',1)}
for _,r in d.iterrows():
    dx,dy,ha,ld=LAB.get(r['site'],(8,5,'left',0))
    ax.annotate(r['name'],(r['slope_gap'],r['accuracy']),textcoords='offset points',xytext=(dx,dy),
                fontsize=7.8,color='#2b2b2b',ha=ha,va='center',
                arrowprops=(dict(arrowstyle='-',color='#b8b8b8',lw=0.6,shrinkA=1,shrinkB=3) if ld else None))
ax.set_xlabel('Slope gap between favela and non-favela buildings  (°)',fontsize=9.6)
ax.set_ylabel('Per-site classification accuracy  (%)',fontsize=9.6)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',color='#eeeeee',lw=0.8,zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(-0.3,5.4); ax.set_ylim(82,100)
ax.annotate(f'Spearman $\\rho$ = {rho:.2f},  $p$ = {p:.2f}  (n.s.)',xy=(0.96,0.06),xycoords='axes fraction',
            ha='right',va='bottom',fontsize=9.2,color='#444',
            bbox=dict(boxstyle='round,pad=0.45',fc='#f6f6f6',ec='#cfcfcf',lw=0.8))
fig.tight_layout()
fig.savefig(OUT,bbox_inches='tight'); fig.savefig('/tmp/slope_gap_new.png',dpi=150,bbox_inches='tight')
print("[saved]",os.path.basename(OUT))
