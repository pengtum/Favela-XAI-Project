"""
One-off prep for Figure-2 panel a (professional locator map):
 - rio_detailed.geojson : detailed Rio de Janeiro municipality boundary (osmnx, ~20k vertices, incl. islands), EPSG:3857
 - _fig2_basemap.npy     : CartoDB-Positron (OpenStreetMap-derived) basemap raster covering the favela-cluster extent
 - _fig2_basemap_meta.npy: [img_xmin,img_xmax,img_ymin,img_ymax, MAPX0,MAPX1,MAPY0,MAPY1]  (all EPSG:3857)
Run once; the figure script then just imshow()s the cached raster (offline, fast, reproducible).
"""
import os, warnings, numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")
import contextily as cx, osmnx as ox
OUT=os.path.dirname(os.path.abspath(__file__))

# detailed municipality boundary
rio=ox.geocode_to_gdf("Rio de Janeiro, Brazil").to_crs(3857)
rio[["geometry"]].to_file(os.path.join(OUT,"rio_detailed.geojson"),driver="GeoJSON")

# FULL Rio municipality extent so the recognizable, intricate coastline (Guanabara/Sepetiba bays,
# lagoons, offshore islands) is visible; the ten sites then spread across the real city.
rb=rio.total_bounds; w=rb[2]-rb[0]; h=rb[3]-rb[1]
MAPX=(rb[0]-0.015*w, rb[2]+0.015*w); MAPY=(rb[1]-0.03*h, rb[3]+0.07*h)
pw=(MAPX[1]-MAPX[0]); ph=(MAPY[1]-MAPY[0]); pad=0.015
img,iext=cx.bounds2img(MAPX[0]-pad*pw,MAPY[0]-pad*ph,MAPX[1]+pad*pw,MAPY[1]+pad*ph,
                       zoom=13,source=cx.providers.CartoDB.PositronNoLabels)
np.save(os.path.join(OUT,"_fig2_basemap.npy"),img)
np.save(os.path.join(OUT,"_fig2_basemap_meta.npy"),np.array([iext[0],iext[1],iext[2],iext[3],MAPX[0],MAPX[1],MAPY[0],MAPY[1]]))
print("rio_detailed.geojson saved; basemap",img.shape,"zoom13 PositronNoLabels (full municipality)")
print("MAPX",[round(v) for v in MAPX],"MAPY",[round(v) for v in MAPY],"aspect=%.3f"%(pw/ph))
