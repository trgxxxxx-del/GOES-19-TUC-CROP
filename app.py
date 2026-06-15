import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

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
        max-width: 750px !important;
        display: block;
        margin: auto;
    }
    div[data-testid="column"]:first-child {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-top: 3rem;
    }
    div[data-testid="column"]:last-child {
        margin-top: -3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Imágen satelital de Tucumán")

# ── URLs ─────────────────────────────────────────────────────────────────────
URL_GEOCOLOR = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"
URL_NIGHT    = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/16/7200x4320.jpg"
URL_BAND13   = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/13/7200x4320.jpg"

# ── Constantes ───────────────────────────────────────────────────────────────
CROP            = (2717, 1382, 2932, 1600)
THRESHOLD_DIA   = 110
THRESHOLD_NOCHE = 110
MAT_PATH        = Path("matriz de departamentos.xlsx")
MODEL_PATH      = Path("LapSRN_x2.pb")
TZ_ARG          = timezone(timedelta(hours=-3))

# ── Categorías de lluvia ──────────────────────────────────────────────────────
# Banda 13 (IR onda larga): imagen en escala de grises donde
#   píxel OSCURO (valor bajo)  → temperatura de brillo FRÍA → nube alta → lluvia
#   píxel CLARO (valor alto)   → temperatura de brillo CÁLIDA → superficie o nube baja
#
# Umbrales empíricos sobre imagen 0-255 (invertida: frío=oscuro):
#   < 40  → tops muy fríos < −60°C → Tormenta fuerte
#   40-70 → tops −45/−60°C         → Lluvia fuerte
#   70-95 → tops −30/−45°C         → Lluvia moderada
#   95-120→ tops −15/−30°C         → Lluvia leve
#   120-150→ tops 0/−15°C          → Alta nubosidad
#   > 150 → superficie/nube baja   → Sin lluvia significativa

LLUVIA_CATEGORIAS = [
    "Tormenta fuerte",
    "Lluvia fuerte",
    "Lluvia moderada",
    "Lluvia leve",
    "Alta nubosidad",
    "Sin lluvia",
]
LLUVIA_COLORES = {
    "Tormenta fuerte": "#e6271e",
    "Lluvia fuerte":   "#ed7a05",
    "Lluvia moderada": "#e8e703",
    "Lluvia leve":     "#6be709",
    "Alta nubosidad":  "#1122c0",
    "Sin lluvia":      "#6abf6a",
}
LLUVIA_ICONOS = {
    "Tormenta fuerte": "⛈️",
    "Lluvia fuerte":   "🌧️",
    "Lluvia moderada": "🌦️",
    "Lluvia leve":     "🌂",
    "Alta nubosidad":  "☁️",
    "Sin lluvia":      "🌤️",
}

# Umbrales de píxel (valor máximo para pertenecer a la categoría)
# Imagen B13 NOAA: oscuro=frío=nube alta, claro=cálido=sin nube
LLUVIA_PIXELES = {
    "Tormenta fuerte": 40,    # px < 40
    "Lluvia fuerte":   70,    # 40 ≤ px < 70
    "Lluvia moderada": 95,    # 70 ≤ px < 95
    "Lluvia leve":     120,   # 95 ≤ px < 120
    "Alta nubosidad":  150,   # 120 ≤ px < 150
    # > 150 → Sin lluvia
}

# % mínimo de píxeles del departamento en esa categoría para activarla
LLUVIA_UMBRAL_PCT = {
    "Tormenta fuerte": 2,
    "Lluvia fuerte":   4,
    "Lluvia moderada": 6,
    "Lluvia leve":     15,
    "Alta nubosidad":  10,
}

# ── Departamentos ─────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def es_de_dia(dt_arg: datetime) -> bool:
    return 6 <= dt_arg.hour < 18


@st.cache_resource
def cargar_modelo_sr():
    if not MODEL_PATH.exists():
        return None
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL_PATH))
    sr.setModel("lapsrn", 2)
    return sr


def mejorar_imagen(img: Image.Image, sr_model) -> Image.Image:
    if sr_model is None:
        w, h = img.size
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
    else:
        arr    = np.array(img.convert("RGB"))
        result = sr_model.upsample(arr)
        img    = Image.fromarray(result)

    arr = np.array(img)
    arr = cv2.bilateralFilter(arr, d=5, sigmaColor=30, sigmaSpace=30)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img


def imagen_a_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def color_nubosidad(pct: float) -> str:
    if pct >= 75:   return "#4a90d9"
    elif pct >= 50: return "#7fb3e0"
    elif pct >= 25: return "#f0c040"
    else:           return "#6abf6a"


def _mascaras_lluvia_b13(gray: np.ndarray) -> dict:
    """
    Clasifica píxeles de Banda 13 (escala de grises) por temperatura de brillo.
    Valor bajo (oscuro) = nube alta fría = mayor probabilidad de lluvia.
    """
    lim = LLUVIA_PIXELES
    mascaras = {
        "Tormenta fuerte": gray < lim["Tormenta fuerte"],
        "Lluvia fuerte":   (gray >= lim["Tormenta fuerte"]) & (gray < lim["Lluvia fuerte"]),
        "Lluvia moderada": (gray >= lim["Lluvia fuerte"])   & (gray < lim["Lluvia moderada"]),
        "Lluvia leve":     (gray >= lim["Lluvia moderada"]) & (gray < lim["Lluvia leve"]),
        "Alta nubosidad":  (gray >= lim["Lluvia leve"])     & (gray < lim["Alta nubosidad"]),
        "Sin lluvia":       gray >= lim["Alta nubosidad"],
    }
    return mascaras


# ── Carga de imágenes ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def cargar_imagen_satelital():
    ahora_arg = datetime.now(TZ_ARG)
    diurno    = es_de_dia(ahora_arg)

    resp_geo = requests.get(URL_GEOCOLOR, timeout=120)
    resp_geo.raise_for_status()

    last_modified = resp_geo.headers.get("Last-Modified", "")
    if last_modified:
        dt_utc = datetime.strptime(
            last_modified, "%a, %d %b %Y %H:%M:%S %Z"
        ).replace(tzinfo=timezone.utc)
        dt_arg = dt_utc.astimezone(TZ_ARG)
        ts_str = dt_arg.strftime("%-d de %B %Y, %H:%M hs (Argentina)")
        ts_key = last_modified
    else:
        ts_str = "—"
        ts_key = ""

    img_geo  = Image.open(BytesIO(resp_geo.content))
    crop_geo = img_geo.crop(CROP)

    if diurno:
        crop_calculo = crop_geo
    else:
        resp_night   = requests.get(URL_NIGHT, timeout=120)
        resp_night.raise_for_status()
        img_night    = Image.open(BytesIO(resp_night.content))
        crop_calculo = img_night.crop(CROP)

    resp_b13 = requests.get(URL_BAND13, timeout=120)
    resp_b13.raise_for_status()
    img_b13  = Image.open(BytesIO(resp_b13.content))
    crop_b13 = img_b13.crop(CROP)

    return crop_geo, crop_calculo, crop_b13, ts_str, ts_key, diurno


# ── Cálculos ──────────────────────────────────────────────────────────────────
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


@st.cache_data(ttl=0)
def calcular_lluvia(img_bytes_b13: bytes, ts_key: str):
    # Banda 13 se carga en escala de grises
    img = Image.open(BytesIO(img_bytes_b13)).convert("L")

    df           = pd.read_excel(MAT_PATH, sheet_name=0, header=None)
    dept_matrix  = df.values.astype(int)
    mat_h, mat_w = dept_matrix.shape

    if img.size != (mat_w, mat_h):
        img = img.resize((mat_w, mat_h), Image.LANCZOS)

    gray     = np.array(img)
    mascaras = _mascaras_lluvia_b13(gray)

    results = []
    for nombre, codigo in DEPARTAMENTOS.items():
        mask_dept = dept_matrix == codigo
        total     = int(np.sum(mask_dept))
        if total == 0:
            results.append((nombre, "Sin lluvia", {}))
            continue

        pcts = {cat: float(np.sum(m & mask_dept)) / total * 100
                for cat, m in mascaras.items()}

        categoria = "Sin lluvia"
        for cat in ["Tormenta fuerte", "Lluvia fuerte",
                    "Lluvia moderada", "Lluvia leve", "Alta nubosidad"]:
            if pcts[cat] >= LLUVIA_UMBRAL_PCT[cat]:
                categoria = cat
                break

        results.append((nombre, categoria, pcts))

    orden = {c: i for i, c in enumerate(reversed(LLUVIA_CATEGORIAS))}
    results.sort(key=lambda x: orden[x[1]], reverse=True)
    return results


# ── UI ────────────────────────────────────────────────────────────────────────
try:
    sr_model = cargar_modelo_sr()

    crop_geo, crop_calculo, crop_b13, ts_str, ts_key, diurno = cargar_imagen_satelital()

    with st.spinner("✨ Mejorando imagen..."):
        crop_display = mejorar_imagen(crop_geo, sr_model)

    modo = "☀️ GEOCOLOR (día)" if diurno else "🌙 Day/Night Cloud Combo (noche)"
    st.caption(f"🕐 Última actualización: **{ts_str}** · {modo}")

    if st.button("🔄 Recargar imagen"):
        st.cache_data.clear()
        st.rerun()

    col_img, col_tabla = st.columns([1, 1])

    with col_img:
        st.image(crop_display, use_container_width=True)
        st.download_button(
            label="⬇️ Descargar imagen mejorada",
            data=imagen_a_bytes(crop_display, fmt="PNG"),
            file_name="tucuman_satelital.png",
            mime="image/png",
            use_container_width=False
        )

    with col_tabla:

        tab_nubes, tab_lluvia = st.tabs(["☁️ Nubosidad", "🌧️ Lluvia"])

        with tab_nubes:
            st.subheader("☁️ Nubosidad por departamento")
            if not MAT_PATH.exists():
                st.warning(
                    "No se encontró **matriz de departamentos.xlsx**. "
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
                    st.error(f"Error en el cálculo de nubosidad: {e}")

        with tab_lluvia:
            st.subheader("🌧️ Probabilidad de lluvia por departamento")

            # Leyenda
            st.markdown(
                " &nbsp; ".join(
                    f"<span style='background:{LLUVIA_COLORES[c]}30; "
                    f"border-left:3px solid {LLUVIA_COLORES[c]}; "
                    f"padding:1px 6px; border-radius:3px; font-size:0.8em'>"
                    f"{LLUVIA_ICONOS[c]} {c}</span>"
                    for c in LLUVIA_CATEGORIAS
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

            if not MAT_PATH.exists():
                st.warning(
                    "No se encontró **matriz de departamentos.xlsx**. "
                    "Subila al repositorio para activar el cálculo."
                )
            else:
                try:
                    b13_bytes  = imagen_a_bytes(crop_b13)
                    datos_lluv = calcular_lluvia(b13_bytes, ts_key)

                    for nombre, categoria, pcts in datos_lluv:
                        color = LLUVIA_COLORES[categoria]
                        icono = LLUVIA_ICONOS[categoria]
                        # Tooltip con desglose de % por categoría
                        detalle = " | ".join(
                            f"{c}: {pcts.get(c, 0):.1f}%"
                            for c in LLUVIA_CATEGORIAS
                            if pcts.get(c, 0) > 0.5
                        )
                        st.markdown(
                            f"""<div style='display:flex; justify-content:space-between;
                                align-items:center; padding:4px 8px; margin:2px 0;
                                border-radius:4px; background:{color}20;
                                border-left:4px solid {color}'
                                title='{detalle}'>
                                <span>{nombre}</span>
                                <strong>{icono} {categoria}</strong>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    # Nota sobre calibración
                    st.markdown(
                        "<p style='font-size:0.75em; color:#888; margin-top:8px'>"
                        "Basado en temperatura de brillo Banda 13 (IR onda larga). "
                        "Pasá el cursor sobre cada departamento para ver el desglose.</p>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error en el cálculo de lluvia: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Error al cargar la imagen: {e}")
