import streamlit as st
from PIL import Image
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO
import base64
import json
from pathlib import Path

st.set_page_config(
    page_title="GOES-19 Tucumán",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ GOES-19 — Tucumán")

URL  = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"
CROP = (2679, 1344, 2985, 1639)

PROMPT = """Te envío dos imágenes: la primera es un mapa con los departamentos de la provincia de Tucumán (Argentina), y la segunda es una imagen satelital GOES-19 de la misma zona.
Usando el mapa como referencia geográfica, estimá el porcentaje de cobertura nubosa (0 a 100) para cada departamento.
Respondé SOLO en JSON con este formato exacto, sin texto adicional, sin bloques de código:
{"Capital": 80, "Yerba Buena": 60, "Tafí Viejo": 40, "Tafí del Valle": 20, "Trancas": 90, "Burruyacú": 70, "Cruz Alta": 50, "Leales": 30, "Simoca": 10, "Graneros": 80, "La Cocha": 60, "Juan Bautista Alberdi": 40, "Río Chico": 20, "Chicligasta": 90, "Monteros": 70, "Famaillá": 50, "Lules": 30}"""

def imagen_a_base64(img: Image.Image, fmt="JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

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

@st.cache_data(ttl=600)
def analizar_con_gemini(img_satelital_b64: str) -> dict:
    api_key  = st.secrets["GEMINI_API_KEY"]
    mapa     = Image.open(Path("departamentos_tucuman.jpg"))
    mapa_b64 = imagen_a_base64(mapa)

    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": mapa_b64}},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_satelital_b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.2}
    }

    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()

    texto = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    texto = texto.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(texto)

def color_nubosidad(pct: int) -> str:
    if pct >= 75:   return "#4a90d9"
    elif pct >= 50: return "#7fb3e0"
    elif pct >= 25: return "#f0c040"
    else:           return "#6abf6a"

try:
    crop, ts_str = cargar_imagen_satelital()
    st.caption(f"🕐 Última actualización: {ts_str}")

    col_img, col_tabla = st.columns([3, 2])

    with col_img:
        st.image(crop, use_container_width=True)

    with col_tabla:
        st.subheader("☁️ Nubosidad por departamento")
        with st.spinner("Analizando con Gemini..."):
            img_b64 = imagen_a_base64(crop)
            datos   = analizar_con_gemini(img_b64)

        for depto, pct in sorted(datos.items(), key=lambda x: -x[1]):
            color = color_nubosidad(pct)
            st.markdown(
                f"""<div style='display:flex; justify-content:space-between;
                    padding:4px 8px; margin:2px 0; border-radius:4px;
                    background:{color}20; border-left:4px solid {color}'>
                    <span>{depto}</span>
                    <strong>{pct}%</strong>
                </div>""",
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"⚠️ Error: {e}")
