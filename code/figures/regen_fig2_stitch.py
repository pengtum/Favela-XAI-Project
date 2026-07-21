"""
Stitch main Figure 2 from existing vector PDFs (no re-rendering of a/b):
  a  = panel (a) cropped from the original Figure_X_accuracy_new.pdf (study area + counts)
  b  = Supplementary S5 whole figure (Figure_X_casestudies_class.pdf)
  c  = freshly rendered row c (per-site lollipop + boundary-error decay curve)
Composed top-to-bottom onto one page with fitz (PyMuPDF), preserving vectors.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=os.path.dirname(os.path.abspath(__file__))
FAV="#E28E2C"; NFV="#6BAED6"; INK="#222222"; ACC="#2A6F97"
CROP_A_BOTTOM=372  # pts from top of accuracy pdf where panel (a) ends (page is 1139.6 x 1415.8)

# ---------- 1) render row c standalone ----------
METRICS={'Rocinha':(0.977,0.981,0.978,0.980,0.977),'Morro da Guaíba':(0.962,0.919,0.942,0.930,0.952),
 'Barão':(0.958,0.929,0.911,0.920,0.946),'Jacarezinho':(0.954,0.940,0.929,0.934,0.949),
 'Faz Quem Quer':(0.949,0.906,0.914,0.910,0.937),'Catumbi':(0.934,0.913,0.921,0.917,0.931),
 'Palmeiras':(0.931,0.839,0.884,0.861,0.907),'Fazenda Coqueiro':(0.920,0.935,0.943,0.939,0.911),
 'Nova Cidade':(0.881,0.767,0.811,0.788,0.853),'Grotão de C.B.':(0.866,0.834,0.871,0.852,0.865)}
ORDER=list(METRICS); MEANACC=0.933
KEY={'Rocinha':'rocinha','Morro da Guaíba':'morro_da_guaiba','Barão':'barao','Jacarezinho':'jacarezinho','Faz Quem Quer':'faz_quem_quer','Catumbi':'catumbi','Palmeiras':'palmeras','Fazenda Coqueiro':'fazena_coqueiro','Nova Cidade':'nova_cidade','Grotão de C.B.':'grotao_de_costa_barros'}
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":7.5,"axes.linewidth":0.7,
    "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
BE=pd.read_parquet(os.path.join(OUT,'boundary_error_data.parquet'))
def lollipop(ax):
    y=np.arange(len(ORDER))[::-1]
    for i,d in enumerate(ORDER):
        a,p,r,f1,fm=METRICS[d]
        ax.plot([0.8,a],[y[i],y[i]],color='#cfcfcf',lw=1.2,zorder=1)
        ax.scatter(p,y[i],marker='|',s=110,c=NFV,zorder=2); ax.scatter(r,y[i],marker='|',s=110,c=FAV,zorder=2)
        ax.scatter(a,y[i],s=42,c=ACC,ec='white',lw=0.6,zorder=3)
    ax.axvline(MEANACC,color='#888',ls='--',lw=0.9); ax.text(MEANACC,len(ORDER)-0.1,f'mean {MEANACC:.3f}',fontsize=6.3,color='#666',ha='center',va='bottom')
    ax.set_yticks(y); ax.set_yticklabels(ORDER,fontsize=6.8); ax.set_xlim(0.8,1.005); ax.set_ylim(-0.6,len(ORDER)-0.1)
    ax.set_xlabel('test accuracy (dot) · precision $|$ recall (ticks)',fontsize=7)
    ax.set_title('Per-site detection performance',fontsize=8.5,fontweight='bold',loc='left')
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',mfc=ACC,ms=5.5,label='accuracy'),Line2D([0],[0],marker='|',color=NFV,label='precision'),Line2D([0],[0],marker='|',color=FAV,label='recall')],loc='lower right',fontsize=6.2)
def boundary_decay(ax,lo=-90,hi=90,bw=6):
    edges=np.arange(lo,hi+bw,bw); cen=0.5*(edges[:-1]+edges[1:])
    def dens(d):
        d=d[(d>=lo)&(d<=hi)]; h,_=np.histogram(d,bins=edges); return h/max(h.sum(),1)/bw*100
    ax.axvspan(lo,0,color=FAV,alpha=0.05); ax.axvspan(0,hi,color=NFV,alpha=0.05)
    M=np.vstack([dens(BE[(BE.site==KEY[d])&(BE.error)]['signed_dist'].values) for d in ORDER])
    ax.fill_between(cen,np.percentile(M,10,axis=0),np.percentile(M,90,axis=0),color='#cdd7e0',alpha=0.7,lw=0,label='per-site 10–90%')
    ax.plot(cen,dens(BE[BE.error]['signed_dist'].values),color=INK,lw=2.3,zorder=5,label='all sites (pooled)')
    ax.axvline(0,color='#444',ls='--',lw=1.0)
    ax.set_xlim(lo,hi); ax.set_xlabel('signed distance to favela boundary (m)  —  favela $\\leftarrow$ | $\\rightarrow$ formal',fontsize=7)
    ax.set_ylabel('share of misclassified buildings (%/m)',fontsize=7)
    ax.set_title('Errors are graded at the boundary',fontsize=8.5,fontweight='bold',loc='left')
    ax.text(0.97,0.93,'misclassifications collapse\nonto the favela edge\n(Rocinha 70.9% within 50 m)',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,color='#555')
    ax.legend(loc='upper left',fontsize=6.4)
fig=plt.figure(figsize=(7.6,2.9)); gs=fig.add_gridspec(1,2,width_ratios=[1.0,1.3],wspace=0.26)
lollipop(fig.add_subplot(gs[0,0])); boundary_decay(fig.add_subplot(gs[0,1]))
fig.tight_layout(); fig.savefig(os.path.join(OUT,'Figure_X_fig2_rowc.pdf'),bbox_inches='tight'); plt.close(fig)
print('[rowc] rendered')

# ---------- 2) compose with fitz: a (cropped PDF) + S2 (jpg image) + row c (PDF) ----------
import fitz
from PIL import Image
S2JPG=os.path.join(OUT,'Figure_X_spatial_mapping.jpg')
# crop 2 representative S2 tiles CORRECTLY: render full S2 to RGB via fitz (handles Adobe-CMYK), then crop
_d=fitz.open(); _pg=_d.new_page(width=4833/4,height=4075/4); _pg.insert_image(_pg.rect,filename=S2JPG)
_pm=_pg.get_pixmap(matrix=fitz.Matrix(4,4)); full=Image.frombytes('RGB',[_pm.width,_pm.height],_pm.samples)
cw,ch=full.size[0]/4,full.size[1]/3   # 4 cols x 3 rows
TILES=[]
for nm,box in [('rocinha',(int(cw*1),2,int(cw*2),int(ch))),('catumbi',(2,int(ch*2),int(cw),int(ch*3)-2))]:
    p=os.path.join(OUT,f'_s2_{nm}.jpg'); cr=full.crop(box); cr.save(p,quality=93); TILES.append((p,cr.size))
acc=fitz.open(os.path.join(OUT,'Figure_X_accuracy_new.pdf')); rc=fitz.open(os.path.join(OUT,'Figure_X_fig2_rowc.pdf'))
a_clip=fitz.Rect(0,0,acc[0].rect.width,CROP_A_BOTTOM); acc[0].get_pixmap(clip=a_clip,dpi=110).save('/tmp/fig2_acrop.png')
W=a_clip.width; PAD=12; GAP=14; ha=a_clip.height
Wb=0.60*W; tw=(Wb-GAP)/2; tile_ar=TILES[0][1][1]/TILES[0][1][0]; th=tw*tile_ar     # b: 2 tiles side-by-side, centered
hc=W*(rc[0].rect.height/rc[0].rect.width)
H=PAD+ha+PAD+th+PAD+hc+PAD
doc=fitz.open(); pg=doc.new_page(width=W,height=H)
pg.show_pdf_page(fitz.Rect(0,PAD,W,PAD+ha),acc,0,clip=a_clip)
yb=PAD+ha+PAD; x0=(W-Wb)/2
pg.insert_image(fitz.Rect(x0,yb,x0+tw,yb+th),filename=TILES[0][0])
pg.insert_image(fitz.Rect(x0+tw+GAP,yb,x0+2*tw+GAP,yb+th),filename=TILES[1][0])
yc=yb+th+PAD
pg.show_pdf_page(fitz.Rect(0,yc,W,yc+hc),rc,0)
for x,yy,t in [(10,PAD,'a'),(max(x0-16,6),yb,'b'),(10,yc,'c')]:
    pg.insert_text(fitz.Point(x,yy+15),t,fontsize=16,fontname='hebo',color=(0,0,0))
doc.save(os.path.join(OUT,'Figure_X_fig2_stitched.pdf'))
pg.get_pixmap(dpi=120).save('/tmp/fig2_stitched.png')
print(f'[stitched] W={W:.0f} H={H:.0f} aspect(H/W)={H/W:.2f}  (ha={ha:.0f} th={th:.0f} hc={hc:.0f})')
