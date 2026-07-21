"""
Terrain separability vs detection accuracy (Supplementary S11), restyled to
the publication spec used for the other revision figures.
  (a) ridge plot of per-site slope distributions (favela vs non-favela)
  (b) ridge plot of per-site elevation distributions
  (c) terrain distribution overlap (OVL) vs morphological accuracy
Per-site accuracies match the Morph column of the S11 terrain table.
Exports PDF (referenced by the .tex) + editable SVG. No re-modelling.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
FAV="#E28E2C"; NFV="#0F4D92"; OVL_C="#8E7CB5"; REF="#767676"

PROJ = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
CSV = os.path.join(PROJ,"Final analysis","terrain analysis","experiments","exp1_pooled_test_buildings.csv")
OUT = os.path.dirname(os.path.abspath(__file__))

# per-site accuracy = Accuracy column of Supplementary Table S1 (tab:site_metrics_test) = main reported口径, matching Fig 2b
SITE_ACCURACY = {
    'rocinha_2019':97.7,'morro_da_guaiba_2019':96.2,'barao_2019':95.8,'jacarezinho_2019':95.4,
    'faz_quem_quer_2019':94.9,'catumbi_2019':93.4,'palmeras_2019':93.1,'fazena_coqueiro_2019':92.0,
    'nova_cidade_2019':88.1,'grotao_de_costa_barros_2019':86.6,
}

def kde_overlap(a,b,n=500):
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<3 or len(b)<3: return np.nan,None,None,None
    lo,hi=min(a.min(),b.min()),max(a.max(),b.max()); rng=hi-lo
    if rng==0: return np.nan,None,None,None
    x=np.linspace(lo-0.05*rng,hi+0.05*rng,n)
    try: ka=gaussian_kde(a,bw_method=0.3)(x); kb=gaussian_kde(b,bw_method=0.3)(x)
    except Exception: return np.nan,None,None,None
    return np.trapz(np.minimum(ka,kb),x),x,ka,kb

def draw_ridge(ax,df,ordered,var,xlabel,show_labels):
    n=len(ordered); sp=1.0
    for i,(sk,dn,acc,ovl) in enumerate(ordered):
        sub=df[df['site']==sk]
        fav=sub[sub['label']=='favela'][var].dropna().values
        nfav=sub[sub['label']=='not-favela'][var].dropna().values
        if len(fav)<5 or len(nfav)<5: continue
        _,x,kf,kn=kde_overlap(fav,nfav); yb=(n-1-i)*sp
        mx=max(kf.max(),kn.max()); sc=sp*0.85/mx if mx>0 else 1.0
        kf_s,kn_s=kf*sc,kn*sc
        ax.fill_between(x,yb,yb+kf_s,alpha=0.40,color=FAV,lw=0)
        ax.fill_between(x,yb,yb+kn_s,alpha=0.40,color=NFV,lw=0)
        ax.fill_between(x,yb,yb+np.minimum(kf_s,kn_s),alpha=0.55,color=OVL_C,lw=0)
        ax.plot(x,yb+kf_s,color=FAV,lw=0.8); ax.plot(x,yb+kn_s,color=NFV,lw=0.8)
        ax.axhline(yb,color='#cccccc',lw=0.3)
        ax.text(0.985,yb+sp*0.42,f'OVL {ovl:.2f}' if not np.isnan(ovl) else 'OVL n/a',
                fontsize=5.3,color=OVL_C,ha='right',va='center',transform=ax.get_yaxis_transform())
    yticks=[(n-1-i)*sp+sp*0.3 for i in range(n)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{dn} ({acc:.1f}%)' for _,dn,acc,_ in ordered] if show_labels else ['']*n,fontsize=6)
    ax.set_ylim(-0.3,n*sp); ax.set_xlabel(xlabel,fontsize=7)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

def panel(ax,L): ax.set_title(L,fontsize=8,fontweight='bold',loc='left',pad=4)

df=pd.read_csv(CSV)
rows=[]
for sk in df['site'].unique():
    sp=df[df['site']==sk]; dn=sp['site_display'].iloc[0]
    acc=SITE_ACCURACY.get(sk,sp['correct'].mean()*100)
    rows.append(dict(sk=sk,dn=dn,acc=acc,
                     ov_s=kde_overlap(sp[sp.label=='favela']['slope_mean'].values,sp[sp.label=='not-favela']['slope_mean'].values)[0],
                     ov_e=kde_overlap(sp[sp.label=='favela']['elev_mean'].values,sp[sp.label=='not-favela']['elev_mean'].values)[0]))
stats=pd.DataFrame(rows).sort_values('acc',ascending=False)
o_s=[(r.sk,r.dn,r.acc,r.ov_s) for r in stats.itertuples()]
o_e=[(r.sk,r.dn,r.acc,r.ov_e) for r in stats.itertuples()]

fig,axes=plt.subplots(1,3,figsize=(7.2,3.9),gridspec_kw={'width_ratios':[1,1,0.7]})
draw_ridge(axes[0],df,o_s,'slope_mean','mean slope (°)',True);  panel(axes[0],'a   Slope distribution')
draw_ridge(axes[1],df,o_e,'elev_mean','mean elevation (m)',False); panel(axes[1],'b   Elevation distribution')
ax=axes[2]
ax.scatter(stats['ov_s'],stats['acc'],s=26,c=FAV,edgecolors='black',lw=0.4,marker='o',zorder=5,label='slope OVL')
ax.scatter(stats['ov_e'],stats['acc'],s=26,c=NFV,edgecolors='black',lw=0.4,marker='s',zorder=5,label='elevation OVL')
xfit=np.linspace(0,1,50)
for col,c in [('ov_s',FAV),('ov_e',NFV)]:
    m=stats[col].notna()
    if m.sum()>=2: ax.plot(xfit,np.poly1d(np.polyfit(stats.loc[m,col],stats.loc[m,'acc'],1))(xfit),'--',color=c,lw=1.0,alpha=0.6)
rho_s,p_s=spearmanr(stats['ov_s'],stats['acc']); rho_e,p_e=spearmanr(stats['ov_e'],stats['acc'])
ax.text(0.04,0.04,f'slope: $\\rho$={rho_s:.2f}, p={p_s:.2f}\nelev:  $\\rho$={rho_e:.2f}, p={p_e:.2f}',
        transform=ax.transAxes,fontsize=6,va='bottom',color=REF)
ax.set_xlabel('terrain distribution overlap (OVL)',fontsize=7); ax.set_ylabel('accuracy (%)',fontsize=7)
ax.set_xlim(0.05,1.0); ax.set_ylim(82,100); ax.legend(fontsize=6,loc='upper right')
ax.tick_params(labelsize=6); panel(ax,'c   Overlap vs accuracy')

fig.legend(handles=[Patch(fc=FAV,alpha=.6,label='favela'),Patch(fc=NFV,alpha=.6,label='non-favela'),
                    Patch(fc=OVL_C,alpha=.7,label='overlap zone')],
           loc='upper center',ncol=3,fontsize=6.5,bbox_to_anchor=(0.5,1.04))
fig.tight_layout(rect=[0,0,1,0.95]); fig.subplots_adjust(wspace=0.28)
for ext in ('pdf','svg'): fig.savefig(os.path.join(OUT,f'terrain_separability_vs_accuracy.{ext}'),bbox_inches='tight')
fig.savefig('/tmp/terrain_sep_preview.png',dpi=130,bbox_inches='tight'); plt.close(fig)
print("OVL/acc by site:"); print(stats[['dn','acc','ov_s','ov_e']].round(2).to_string(index=False))
print(f"slope rho={rho_s:.2f} p={p_s:.2f} | elev rho={rho_e:.2f} p={p_e:.2f}")
print("[saved] terrain_separability_vs_accuracy.pdf/.svg")
