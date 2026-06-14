
import streamlit as st
from PIL import Image
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Nubosidad en Tucumán",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stImage"] img {
        max-width: 400px !important;
        display: block;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Imágen satelital de Tucumán")

URL_GEOCOLOR = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"
URL_NIGHT    = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/16/7200x4320.jpg"
CROP            = (2717, 1382, 2932, 1600)
THRESHOLD_DIA   = 110
THRESHOLD_NOCHE = 110
MAT_PATH        = Path("matriz de departamentos.xlsx")
TZ_ARG          = timezone(timedelta(hours=-3))

DEPARTAMENTOS = {
    "San Miguel de Tucumán": 76,
    "Trancas":               175,
    "Burruyacú":             139,
    "Tafí Viejo":            97,
    "Tafí del Valle":        29,
    "Yerba Buena":           66,
    "Lules":                 92,
    "Cruz Alta":             164,
    "Leales":                174,
    "Famaillá":              102,
    "Monteros":              97,
    "Chicligasta":           192,
    "Simoca":                194,
    "Río Chico":             141,
    "Juan Bautista Alberdi": 164,
    "La Cocha":              127,
    "Graneros":              219,
}


def es_de_dia(dt_arg: datetime) -> bool:
    return 6 <= dt_arg.hour < 18


@st.cache_data(ttl=600)
def cargar_imagen_satelital():
    ahora_arg = datetime.now(TZ_ARG)
    diurno    = es_de_dia(ahora_arg)

    # Siempre se descarga GEOCOLOR para mostrar en pantalla
    resp_geo = requests.get(URL_GEOCOLOR, timeout=120)
    resp_geo.raise_for_status()

    last_modified = resp_geo.headers.get("Last-Modified", "")
    if last_modified:
        dt_utc = datetime.strptime(
            last_modified, "%a, %d %b %Y %H:%M:%S %Z"
        ).replace(tzinfo=timezone.utc)
        dt_arg  = dt_utc.astimezone(TZ_ARG)
        ts_str  = dt_arg.strftime("%-d de %B %Y, %H:%M hs (Argentina)")
        ts_key  = last_modified
    else:
        ts_str = "—"
        ts_key = ""

    img_geo  = Image.open(BytesIO(resp_geo.content))
    crop_geo = img_geo.crop(CROP)

    # Para el cálculo de nubosidad: de noche se usa el canal Night
    if diurno:
        crop_calculo = crop_geo
    else:
        resp_night   = requests.get(URL_NIGHT, timeout=120)
        resp_night.raise_for_status()
        img_night    = Image.open(BytesIO(resp_night.content))
        crop_calculo = img_night.crop(CROP)

    return crop_geo, crop_calculo, ts_str, ts_key, diurno


@st.cache_data(ttl=0)
def calcular_nubosidad(img_bytes: bytes, ts_key: str, diurno: bool):
    img = Image.open(BytesIO(img_bytes)).convert("L")

    df           = pd.read_excel(MAT_PATH, sheet_name=0, header=None)
    dept_matrix  = df.values.astype(int)
    mat_h, mat_w = dept_matrix.shape

    if img.size != (mat_w, mat_h):
        img = img.resize((mat_w, mat_h), Image.LANCZOS)

    gray         = np.array(img)
    threshold    = THRESHOLD_DIA if diurno else THRESHOLD_NOCHE
    mascara_nube = gray > threshold

    results = []
    for nombre, codigo in DEPARTAMENTOS.items():
        mask  = dept_matrix == codigo
        total = int(np.sum(mask))
        pct   = float(np.sum(mascara_nube & mask)) / total * 100 if total else 0.0
        results.append((nombre, round(pct, 1)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def color_nubosidad(pct: float) -> str:
    if pct >= 75:   return "#4a90d9"
    elif pct >= 50: return "#7fb3e0"
    elif pct >= 25: return "#f0c040"
    else:           return "#6abf6a"


def imagen_a_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


try:
    crop_geo, crop_calculo, ts_str, ts_key, diurno = cargar_imagen_satelital()

    modo = "☀️ GEOCOLOR (día)" if diurno else "🌙 Day/Night Cloud Combo (noche)"
    st.caption(f"🕐 Última actualización: **{ts_str}** · {modo}")

    if st.button("🔄 Recargar imagen"):
        st.cache_data.clear()
        st.rerun()

    col_img, col_tabla = st.columns([3, 2])

    with col_img:
        st.image(crop_geo, use_container_width=True)
        st.download_button(
            label="⬇️ Descargar imagen (215×216 px)",
            data=imagen_a_bytes(crop_geo, fmt="PNG"),
            file_name="tucuman_satelital.png",
            mime="image/png",
            use_container_width=True
        )

    with col_tabla:
        st.subheader("☁️ Nubosidad por departamento")

        if not MAT_PATH.exists():
            st.warning(
                "No se encontró **matriz de departamentos.xlsx** en el directorio. "
                "Subila al repositorio para activar el cálculo."
            )
        else:
            try:
                calculo_bytes = imagen_a_bytes(crop_calculo)
                datos         = calcular_nubosidad(calculo_bytes, ts_key, diurno)

                for nombre, pct in datos:
                    color = color_nubosidad(pct)
                    st.markdown(
                        f"""<div style='display:flex; justify-content:space-between;
                            padding:4px 8px; margin:2px 0; border-radius:4px;
                            background:{color}20; border-left:4px solid {color}'>
                            <span>{nombre}</span>
                            <strong>{pct:.1f}%</strong>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error en el cálculo: {e}")

except Exception as e:
    st.error(f"⚠️ Error al cargar la imagen: {e}")
