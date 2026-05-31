import streamlit as st
from pathlib import Path
from PIL import Image
from datetime import datetime, timezone, timedelta
import re

st.set_page_config(
    page_title="GOES-19 Tucumán",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ GOES-19 — Tucumán")

def get_timestamp_from_filename(folder_path="."):
    archivos = sorted(Path(folder_path).glob("GOES19_*.jpg"), reverse=True)
    if not archivos:
        return None, None
    archivo = archivos[0]
    match = re.search(r'(\d{8})_(\d{4})Z', archivo.stem)
    if not match:
        return None, None
    dt_utc = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_arg = dt_utc.astimezone(timezone(timedelta(hours=-3)))
    return archivo, dt_arg

img_path, dt_arg = get_timestamp_from_filename()

if img_path:
    ts_str = dt_arg.strftime("%-d de %B %Y, %H:%M hs (Argentina)")
    st.caption(f"🕐 Última actualización: {ts_str}")
    st.image(Image.open(img_path), use_container_width=True)
else:
    st.info("⏳ Imagen no disponible todavía. El bot actualiza cada 20 minutos.")
