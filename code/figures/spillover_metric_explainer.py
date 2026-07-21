"""Explainer: what x0 (transition centre) and w (transition width) mean.
x0,w are the |S'(x)|-weighted mean and std of the SHAP profile -- i.e. the centre
and spread of WHERE the SHAP value changes along the distance-to-boundary axis."""
import numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9,"axes.linewidth":0.6})
FAVc="#3F83A2"; NFVc="#42A969"; CRV="#14746A"
x=np.linspace(-40,40,600)
def prof(x0,s,Sin=2.0,Sout=-0.6): return Sout+(Sin-Sout)*0.5*(1-np.tanh((x-x0)/s))
def moments(S):
    d=np.abs(np.gradient(S,x)); x0=np.sum(d*x)/np.sum(d); w=np.sqrt(np.sum(d*(x-x0)**2)/np.sum(d)); return x0,w,d

fig,(axL,axR)=plt.subplots(1,2,figsize=(11.4,4.4),gridspec_kw={'width_ratios':[1.15,1.0]})
# ---- LEFT: anatomy of one transition ----
S=prof(2.0,9.0); x0,w,d=moments(S)
axL.axvspan(-40,0,color=FAVc,alpha=.07); axL.axvspan(0,40,color=NFVc,alpha=.07)
axL.plot(x,S,color=CRV,lw=2.6,zorder=4)
axL.axhline(0,color='#bbb',ls=':',lw=.7); axL.axvline(0,color='#B64342',ls='--',lw=1.2,zorder=3)
base=-1.7; dens=d/d.max()*1.25
axL.fill_between(x,base,base+dens,color=CRV,alpha=.22,lw=0,zorder=2)
axL.plot(x,base+dens,color=CRV,lw=1.0,alpha=.6)
# x0 vertical + w span on the density
axL.plot([x0,x0],[base,base+dens.max()*0.82],color='#222',lw=1.6,zorder=5)
yw=base+0.32
axL.annotate('',xy=(x0-w,yw),xytext=(x0+w,yw),arrowprops=dict(arrowstyle='<->',color='#222',lw=1.4))
axL.text(x0,base+dens.max()+0.18,'$x_0$',ha='center',va='bottom',fontsize=12,color='#222',fontweight='bold')
axL.text(x0,yw-0.34,'width $2w$',ha='center',va='top',fontsize=9,color='#222')
axL.text(0.5,3.05,"$|S'(x)|$  =  how fast SHAP is changing  =  'where the transition happens'",ha='center',fontsize=8.5,color=CRV)
axL.text(-39,2.05,'favela side\n(inside)',fontsize=8,color='#2c6b86',va='top')
axL.text(39,2.05,'formal side\n(outside)',fontsize=8,color='#2d7a4f',va='top',ha='right')
axL.annotate('SHAP profile $S(x)$',(-24,prof(2,9)[np.argmin(abs(x+24))]),xytext=(-37,-0.9),fontsize=9,color=CRV,
             arrowprops=dict(arrowstyle='-',color=CRV,lw=.8))
axL.set_ylim(base-0.15,3.4); axL.set_xlim(-40,40)
axL.set_xlabel('signed distance to favela boundary  $x$  (m)',fontsize=9)
axL.set_ylabel('SHAP ($\\rightarrow$ favela)',fontsize=9)
axL.set_title('$x_0$ and $w$ = mean & spread of where SHAP changes',fontsize=9.5,loc='left')
for sp in ['top','right']: axL.spines[sp].set_visible(False)
# ---- RIGHT: what they distinguish ----
axR.axvspan(-40,0,color=FAVc,alpha=.07); axR.axvspan(0,40,color=NFVc,alpha=.07)
axR.axvline(0,color='#B64342',ls='--',lw=1.2)
for (x0i,si,c,lab) in [(0,3.0,'#0F4D92','sharp:  small $w$, $x_0\\approx0$'),
                       (0,14.0,'#E2700E','fuzzy:  large $w$, $x_0\\approx0$'),
                       (11,5.0,'#9A4D8E','spillover:  $x_0>0$ (flips outside)')]:
    S=prof(x0i,si); axR.plot(x,S,color=c,lw=2.4,label=lab)
    xm,wm,_=moments(S); axR.plot(xm,prof(x0i,si)[np.argmin(abs(x-xm))],marker='o',color=c,ms=6,mec='white',mew=.8,zorder=5)
axR.axhline(0,color='#bbb',ls=':',lw=.7); axR.set_xlim(-40,40); axR.set_ylim(-1.1,2.3)
axR.set_xlabel('signed distance to favela boundary  $x$  (m)',fontsize=9)
axR.set_title('what the two numbers distinguish (dot = $x_0$)',fontsize=9.5,loc='left')
axR.legend(fontsize=8.2,loc='lower left')
for sp in ['top','right']: axR.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(f'/tmp/metric_explainer.{ext}' if ext=='png' else __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'spillover_metric_explainer.pdf'),dpi=140,bbox_inches='tight')
print('saved')
