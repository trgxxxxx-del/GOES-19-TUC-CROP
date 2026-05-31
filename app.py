import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import base64

st.set_page_config(
    page_title="GOES-19 Tucumán",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ GOES-19 — Tucumán en vivo")

img_path  = Path("tucuman.png")
meta_path = Path("last_update.txt")

if not img_path.exists():
    st.info("⏳ Imagen no disponible todavía.")
    st.stop()

ts = meta_path.read_text().strip() if meta_path.exists() else "—"
st.caption(f"🕐 Última actualización: {ts}")

# Coordenadas geográficas del recorte
# Sector SSA: lon -81.30 a -34.00, lat -0.18 a -59.00
# Imagen: 7200x4320 px
# Crop: left=2730, top=1360, right=2986, bottom=1620

SSA_LON_LEFT  = -90.0
SSA_LON_RIGHT = -30.0
SSA_LAT_TOP   =  15.0
SSA_LAT_BOT   = -60.0
IMG_W = 7200
IMG_H = 4320

def px_to_coord(px_x, px_y):
    lon = SSA_LON_LEFT + (px_x / IMG_W) * (SSA_LON_RIGHT - SSA_LON_LEFT)
    lat = SSA_LAT_TOP  + (px_y / IMG_H) * (SSA_LAT_BOT   - SSA_LAT_TOP)
    return lat, lon

lat_top, lon_left  = px_to_coord(2730, 1360)
lat_bot, lon_right = px_to_coord(2986, 1620)

# Centro del mapa
lat_center = (lat_top + lat_bot) / 2
lon_center = (lon_left + lon_right) / 2

# Convertir imagen a base64
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Crear mapa Folium
m = folium.Map(
    location=[lat_center, lon_center],
    zoom_start=7,
    tiles="CartoDB dark_matter"
)

# Superponer imagen satelital
folium.raster_layers.ImageOverlay(
    image=f"data:image/png;base64,{img_b64}",
    bounds=[[lat_bot, lon_left], [lat_top, lon_right]],
    opacity=0.85,
    interactive=True,
    cross_origin=False,
    zindex=1
).add_to(m)

# Marcador San Miguel de Tucumán
folium.Marker(
    location=[-26.8241, -65.2226],
    popup="San Miguel de Tucumán",
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=900, height=650)
