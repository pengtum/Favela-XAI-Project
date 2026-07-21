"""
Precompute the data substrate for the Figure-2 (favela-detection performance)
redesign, so the actual figure scripts load fast:
 - boundary_error_data.parquet: per building -> site, signed distance to the
   favela/non-favela boundary (negative inside favela, positive outside; KD-tree
   to nearest opposite-class building centroid), error flag (label != predicted),
   label. (NB: predicted_label is in-sample; use error LOCATIONS, not rates.)
 - fig2_site_centroids.csv: per site lon/lat (EPSG:4326) + building counts.
Pure bookkeeping on stored predictions + geometry. No re-modelling.
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
import shapely.geometry as sg
warnings.filterwarnings("ignore")
PROJ="/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project"
RES=os.path.join(PROJ,"results","all_features"); OUT=os.path.dirname(os.path.abspath(__file__))
POLY=gpd.read_file(os.path.join(PROJ,"Data","Limite_Favelas_2013","Limite_Favelas_2013.shp"))  # official 2013 favela polygon (labeling source; matches Section 2.1)
SITES=["rocinha","morro_da_guaiba","barao","jacarezinho","faz_quem_quer","catumbi","palmeras","fazena_coqueiro","nova_cidade","grotao_de_costa_barros"]
DISP={'rocinha':'Rocinha','barao':'Barão','catumbi':'Catumbi','faz_quem_quer':'Faz Quem Quer','fazena_coqueiro':'Fazenda Coqueiro','grotao_de_costa_barros':'Grotão de C.B.','jacarezinho':'Jacarezinho','morro_da_guaiba':'Morro da Guaíba','nova_cidade':'Nova Cidade','palmeras':'Palmeiras'}

rows=[]; cent=[]
for s in SITES:
    g=gpd.read_file(os.path.join(RES,s,f"{s}_updated_buildings_with_shap.gpkg"),columns=['label','predicted_label'])
    g=g[g.geometry.is_valid & ~g.geometry.is_empty].copy()
    c=g.geometry.centroid; fav=(g['label']=='favela').values
    # signed distance to the OFFICIAL 2013 favela polygon boundary (matches Section 2.1); - inside favela / + outside
    pl=POLY.to_crs(g.crs); bb=g.total_bounds
    near=pl[pl.intersects(sg.box(bb[0]-300,bb[1]-300,bb[2]+300,bb[3]+300))]
    bnd=near.boundary.union_all()
    dist=c.distance(bnd).values
    sd=np.where(fav,-dist,dist)
    err=(g['label'].values!=g['predicted_label'].values)
    rows.append(pd.DataFrame({'site':s,'signed_dist':sd,'error':err,'fav':fav}))
    # per-site centroid in lon/lat + counts
    ll=gpd.GeoSeries([g.geometry.union_all().centroid],crs=g.crs).to_crs(4326)[0]
    near=int((err & (np.abs(sd)<=50)).sum()); tot_err=int(err.sum())
    cent.append({'site':s,'disp':DISP[s],'lon':ll.x,'lat':ll.y,'n':len(g),'n_fav':int(fav.sum()),'n_non':int((~fav).sum()),
                 'err':tot_err,'err_within50m':near,'err_within50m_pct':round(100*near/max(tot_err,1),1)})
    print(f"  {DISP[s]:18s} n={len(g):>6d} fav={int(fav.sum()):>6d} err={tot_err:>5d} err<=50m={cent[-1]['err_within50m_pct']:>5}% lon/lat={ll.x:.3f},{ll.y:.3f}")
df=pd.concat(rows,ignore_index=True)
df.to_parquet(os.path.join(OUT,'boundary_error_data.parquet'))
pd.DataFrame(cent).to_csv(os.path.join(OUT,'fig2_site_centroids.csv'),index=False)
print(f"[saved] boundary_error_data.parquet ({len(df)} rows) + fig2_site_centroids.csv")
print(f"pooled: {df['error'].mean()*100:.2f}% in-sample error; of errors, {(df['error']&(df['signed_dist'].abs()<=50)).sum()/df['error'].sum()*100:.1f}% within 50m of boundary")
