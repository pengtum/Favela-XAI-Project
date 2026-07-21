"""Re-make Supplementary Fig bettencourt_dist (Figure_X_bettencourt_k_distribution.pdf) — cleaner.
Same 4 panels (A k-histogram; B PC space by k; C PC space by class; D per-site mean k),
all from the per-building CSV. Paper palette, rasterised scatters, no caption change needed."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "pdf.fonttype":42,"font.size":9,"axes.linewidth":0.8})
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
DATA=os.path.join(PROJ,"Final analysis/Experiment_COVID19/Data")
OUT=os.path.join(PROJ,"Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/Figures/Figure_X_bettencourt_k_distribution.pdf")
FAV="#3F83A2"; NFV="#9aa0a6"; INK="#2b2b2b"
DISP={'barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro',
      'grotao_de_costa_barros':'Grotão','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba',
      'nova_cidade':'Nova Cidade','palmeras':'Palmeiras','rocinha':'Rocinha'}

d=pd.read_csv(os.path.join(DATA,"sensitivity_bettencourt_k.csv"))
v=d[d['k'].notna()].copy(); fav=(v['label']=='favela').values
ps=pd.read_csv(os.path.join(DATA,"sensitivity_bettencourt_k_distribution.csv")).sort_values('cohens_d',ascending=True)
rho,_=spearmanr(v['k'],v['PC1_shap'])
pooled_f, pooled_n, pooled_d = ps['pooled_k_mean_favela'].iloc[0], ps['pooled_k_mean_non_favela'].iloc[0], ps['pooled_cohens_d'].iloc[0]
print(f"valid n={len(v):,}  rho(k,PC1_shap)={rho:.2f}  pooled d={pooled_d:.2f}")

fig,axes=plt.subplots(2,2,figsize=(10.5,8.6)); (axA,axB),(axC,axD)=axes

# (A) pooled k histogram by class (density)
kmax=12; bins=np.arange(0.5,kmax+1.5,1)
axA.hist(np.clip(v.loc[~fav,'k'],0,kmax),bins=bins,density=True,color=NFV,alpha=0.75,label='non-favela')
axA.hist(np.clip(v.loc[fav,'k'],0,kmax),bins=bins,density=True,histtype='step',lw=2.0,edgecolor=FAV,label='favela')
axA.set_xlabel('block-complexity index $k$  (clipped at 12)'); axA.set_ylabel('density')
axA.legend(frameon=False,fontsize=8.5,loc='upper right')
axA.text(0.97,0.62,f'pooled mean $k$:\nfavela {pooled_f:.2f}  ·  non-favela {pooled_n:.2f}\nCohen’s $d$ = {pooled_d:.2f}',
         transform=axA.transAxes,ha='right',va='top',fontsize=7.8,color='#555')
axA.spines[['top','right']].set_visible(False)
axA.set_title('(A)',loc='left',fontweight='bold',fontsize=11)

# (B) SHAP PC space coloured by k
order=np.argsort(v['k'].values)  # low k under, high k on top
sc=axB.scatter(v['PC1_shap'].values[order],v['PC2_shap'].values[order],c=np.clip(v['k'].values[order],0,8),
               s=3,cmap='viridis',alpha=0.55,linewidths=0,rasterized=True)
cb=fig.colorbar(sc,ax=axB,fraction=0.046,pad=0.02); cb.set_label('$k$ (clipped at 8)',fontsize=8); cb.ax.tick_params(labelsize=7)
axB.set_xlabel('PC1$_{\\mathrm{SHAP}}$'); axB.set_ylabel('PC2$_{\\mathrm{SHAP}}$')
axB.text(0.03,0.97,f'Spearman $\\rho(k,$ PC1$_{{\\mathrm{{SHAP}}}})$ = {rho:.2f}',transform=axB.transAxes,va='top',fontsize=8,color='#444')
axB.set_title('(B)',loc='left',fontweight='bold',fontsize=11)

# (C) SHAP PC space coloured by class (non-favela under, favela over)
axC.scatter(v.loc[~fav,'PC1_shap'],v.loc[~fav,'PC2_shap'],s=3,c=NFV,alpha=0.35,linewidths=0,rasterized=True)
axC.scatter(v.loc[fav,'PC1_shap'],v.loc[fav,'PC2_shap'],s=3,c=FAV,alpha=0.45,linewidths=0,rasterized=True)
axC.set_xlabel('PC1$_{\\mathrm{SHAP}}$'); axC.set_ylabel('PC2$_{\\mathrm{SHAP}}$')
axC.legend(handles=[Patch(fc=FAV,label='favela'),Patch(fc=NFV,label='non-favela')],frameon=False,fontsize=8.5,loc='upper left')
axC.set_title('(C)',loc='left',fontweight='bold',fontsize=11)
for a in (axB,axC): a.set_xlim(*np.percentile(v['PC1_shap'],[0.5,99.5])); a.set_ylim(*np.percentile(v['PC2_shap'],[0.5,99.5]))

# (D) per-site mean k by class, sorted by Cohen's d (dumbbell)
y=np.arange(len(ps)); names=[DISP.get(s,s) for s in ps['site']]; mx=float(ps['k_mean_favela'].max())
for i,(_,r) in enumerate(ps.iterrows()):
    axD.plot([r['k_mean_non_favela'],r['k_mean_favela']],[i,i],color='#d0d0d0',lw=2,zorder=1)
axD.scatter(ps['k_mean_non_favela'],y,s=55,c=NFV,zorder=3,label='non-favela')
axD.scatter(ps['k_mean_favela'],y,s=55,c=FAV,zorder=3,label='favela')
for i,(_,r) in enumerate(ps.iterrows()):
    axD.text(mx+0.5,i,f"$d$={r['cohens_d']:+.2f}",va='center',ha='left',fontsize=7,color='#666')
axD.set_yticks(y); axD.set_yticklabels(names,fontsize=8); axD.set_xlabel('per-site mean $k$')
axD.set_xlim(0.8,mx+1.7); axD.set_ylim(-0.6,len(ps)-0.4)
axD.legend(frameon=False,fontsize=8.5,loc='lower right',bbox_to_anchor=(1.0,1.0),ncol=2); axD.spines[['top','right']].set_visible(False)
axD.set_title('(D)',loc='left',fontweight='bold',fontsize=11)

fig.tight_layout()
fig.savefig(OUT,bbox_inches='tight',dpi=200); fig.savefig('/tmp/bettencourt_new.png',dpi=120,bbox_inches='tight')
print("[saved]",os.path.basename(OUT))
