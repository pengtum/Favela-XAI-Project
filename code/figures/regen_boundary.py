"""Re-make Supplementary Fig S3 (Figure_X_boundary.pdf) — stacked bar version (clearer than lines).
Per site (+ pooled), the misclassified buildings split by distance to the 2013 favela boundary
into <=25 / 25-50 / 50-100 / >100 m bands, stacked to 100%. Data: boundary_error_data.parquet."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "pdf.fonttype":42,"font.size":9,"axes.linewidth":0.8})
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"Figure_X_boundary.pdf")
DISP={'barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro',
      'grotao_de_costa_barros':'Grotão','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba',
      'nova_cidade':'Nova Cidade','palmeras':'Palmeiras','rocinha':'Rocinha'}
BANDS=[(0,25,'≤25 m'),(25,50,'25–50 m'),(50,100,'50–100 m'),(100,np.inf,'>100 m')]
COLS=['#16415c','#3F83A2','#9cc1d4','#e1ebf1']   # near = dark -> far = light
d=pd.read_parquet(os.path.join(HERE,"boundary_error_data.parquet"))
mis=d[d['error']].copy(); mis['adist']=mis['signed_dist'].abs()
def fr(a): a=a.values; return [np.mean((a>=lo)&(a<hi))*100 for lo,hi,_ in BANDS]
rows=[(DISP[s],fr(g['adist']),len(g)) for s,g in mis.groupby('site')]
rows.sort(key=lambda r:r[1][0]+r[1][1])              # most near-boundary -> top
pooled=('All sites (pooled)',fr(mis['adist']),len(mis))
labels=[r[0] for r in rows]+['']+ [pooled[0]]
vals  =[r[1] for r in rows]+[[0,0,0,0]]+[pooled[1]]   # blank spacer row before pooled
ns    =[r[2] for r in rows]+[None]+[pooled[2]]
y=np.arange(len(labels))
print(f"misclassified n={len(mis):,}; pooled within 50 m = {pooled[1][0]+pooled[1][1]:.0f}%")

fig,ax=plt.subplots(figsize=(8.0,5.4))
left=np.zeros(len(labels))
for bi,(lo,hi,lab) in enumerate(BANDS):
    w=np.array([v[bi] for v in vals])
    ax.barh(y,w,left=left,color=COLS[bi],edgecolor='white',lw=0.6,height=0.74,zorder=3,
            label=lab)
    left+=w
# n annotations at right; bold the pooled row
for yi,(lb,n) in enumerate(zip(labels,ns)):
    if n is None: continue
    ax.text(101.5,yi,f'n={n:,}',va='center',ha='left',fontsize=7,color='#777')
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8.4)
for t,lb in zip(ax.get_yticklabels(),labels):
    if lb.startswith('All sites'): t.set_fontweight('bold')
ax.set_xlim(0,100); ax.set_xlabel('share of misclassified buildings  (%)',fontsize=9.6)
ax.set_ylim(-0.6,len(labels)-0.4)
ax.axvline(50,color='#999',ls=':',lw=1.0,zorder=4)
ax.spines[['top','right','left']].set_visible(False); ax.tick_params(left=False)
ax.legend(handles=[Patch(fc=COLS[i],label=BANDS[i][2]) for i in range(4)],
          ncol=4,frameon=False,fontsize=8.6,loc='lower center',bbox_to_anchor=(0.5,1.005),
          title='distance of building to the 2013 favela boundary',title_fontsize=8.8)
fig.tight_layout()
fig.savefig(OUT,bbox_inches='tight'); fig.savefig('/tmp/boundary_bar.png',dpi=150,bbox_inches='tight')
print("[saved]",os.path.basename(OUT))
