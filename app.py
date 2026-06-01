import streamlit as st
from PIL import Image
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO

st.set_page_config(
    page_title="GOES-19 Tucumán",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ GOES-19 — Tucumán")

URL  = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"
CROP = (2679, 1344, 2985, 1639)

@st.cache_data(ttl=600)
def cargar_imagen_satelital():
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    last_modified = resp.headers.get("Last-Modified", "")
    if last_modified:
        dt_utc = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        dt_arg = dt_utc.astimezone(timezone(timedelta(hours=-3)))
        ts_str = dt_arg.strftime("%-d de %B %Y, %H:%M hs (Argentina)")
    else:
        ts_str = "—"
    img  = Image.open(BytesIO(resp.content))
    crop = img.crop(CROP)
    return crop, ts_str

try:
    crop, ts_str = cargar_imagen_satelital()
    st.caption(f"🕐 Última actualización: {ts_str}")
    st.image(crop, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
