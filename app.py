import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="GOES-19 Tucumán",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ GOES-19 — Tucumán")

img_path  = Path("tucuman.png")
meta_path = Path("last_update.txt")

if img_path.exists():
    ts = meta_path.read_text().strip() if meta_path.exists() else "—"
    st.caption(f"🕐 Última actualización: {ts}")
    st.image(Image.open(img_path), use_container_width=True)
else:
    st.info("⏳ Imagen no disponible todavía. El bot actualiza cada 20 minutos.")
