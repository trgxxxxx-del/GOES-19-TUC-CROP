import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
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
# Un único canal: GEOCOLOR, tanto de día como de noche.
URL_GEOCOLOR = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"

# ── Constantes ───────────────────────────────────────────────────────────────
CROP            = (2717, 1382, 2932, 1600)
THRESHOLD_DIA   = 140
THRESHOLD_NOCHE = 110
MAT_PATH        = Path("matriz de departamentos.xlsx")
MODEL_PATH      = Path("LapSRN_x2.pb")
TZ_ARG          = timezone(timedelta(hours=-3))

# Umbrales para distinguir luces de ciudad (cálidas: R y G altos respecto a B)
# de nubes reales (blancas/azuladas) en el canal GEOCOLOR nocturno.
LUZ_R_MENOS_B  = 40
LUZ_G_MENOS_B  = 15

# Escala de agrandado para la imagen de máscara (la matriz de deptos es chica)
ESCALA_MASCARA = 3

# Ancho máximo (en píxeles) de "costura" entre dos departamentos vecinos que
# se considera parte del límite y no del fondo (evita bordes duplicados).
ANCHO_MAX_COSTURA = 2

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
CODIGOS_VALIDOS = list(DEPARTAMENTOS.values())


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


# ── Carga de imagen ───────────────────────────────────────────────────────────
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

    return crop_geo, ts_str, ts_key, diurno


# ── Cálculo ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def calcular_mascara_nube(img_bytes: bytes, ts_key: str, diurno: bool):
    """Devuelve (dept_matrix, mascara_nube) para reusar tanto en la tabla
    de porcentajes como en la imagen de máscara."""
    img_rgb = Image.open(BytesIO(img_bytes)).convert("RGB")

    df           = pd.read_excel(MAT_PATH, sheet_name=0, header=None)
    dept_matrix  = df.values.astype(int)
    mat_h, mat_w = dept_matrix.shape

    if img_rgb.size != (mat_w, mat_h):
        img_rgb = img_rgb.resize((mat_w, mat_h), Image.LANCZOS)

    arr = np.array(img_rgb).astype(np.int16)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Luminancia equivalente a convert("L") (ITU-R BT.601)
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    threshold    = THRESHOLD_DIA if diurno else THRESHOLD_NOCHE
    mascara_nube = gray > threshold

    if not diurno:
        # Las luces urbanas son cálidas (R y G bastante por encima de B).
        # Si se ven, el cielo está despejado ahí (una nube las taparía),
        # así que hay que excluirlas del conteo de nubosidad.
        luces_ciudad = (R - B > LUZ_R_MENOS_B) & (G - B > LUZ_G_MENOS_B)
        mascara_nube = mascara_nube & ~luces_ciudad

    return dept_matrix, mascara_nube


def calcular_tabla_nubosidad(dept_matrix: np.ndarray, mascara_nube: np.ndarray):
    results = []
    for nombre, codigo in DEPARTAMENTOS.items():
        mask  = dept_matrix == codigo
        total = int(np.sum(mask))
        pct   = float(np.sum(mascara_nube & mask)) / total * 100 if total else 0.0
        results.append((nombre, round(pct, 1)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _cerrar_costuras(dept_matrix: np.ndarray, max_ancho: int = ANCHO_MAX_COSTURA) -> np.ndarray:
    """La matriz de departamentos trae una 'costura' (celdas con código
    inválido, ej. 0) de uno o dos píxeles entre deptos vecinos, en vez de
    que un depto termine justo donde empieza el otro. Si se detecta el
    borde comparando códigos directamente, esa costura genera DOS líneas
    (depto→costura y costura→depto) en vez de una. Acá se le asigna a cada
    celda de costura el código del departamento válido más cercano, pero
    solo si la costura es angosta (<= max_ancho); las zonas anchas de
    fondo real (fuera del mapa) quedan intactas."""
    es_valido = np.isin(dept_matrix, CODIGOS_VALIDOS)
    if es_valido.all():
        return dept_matrix

    distancia, indices = ndimage.distance_transform_edt(
        ~es_valido, return_distances=True, return_indices=True
    )
    vecino_mas_cercano = dept_matrix[indices[0], indices[1]]
    a_rellenar = (~es_valido) & (distancia <= max_ancho)
    return np.where(a_rellenar, vecino_mas_cercano, dept_matrix)


def generar_imagen_mascara(dept_matrix: np.ndarray, mascara_nube: np.ndarray,
                            escala: int = ESCALA_MASCARA) -> Image.Image:
    """Fondo blanco, nubes detectadas en gris, y los límites entre
    departamentos (o entre departamento y fuera de mapa) en negro,
    con las costuras finas cerradas para que el borde no salga doble."""
    dept_sin_costura = _cerrar_costuras(dept_matrix)

    h, w = dept_sin_costura.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    canvas[mascara_nube] = (150, 150, 150)

    borde = np.zeros((h, w), dtype=bool)
    borde[:, :-1] |= dept_sin_costura[:, :-1] != dept_sin_costura[:, 1:]
    borde[:-1, :] |= dept_sin_costura[:-1, :] != dept_sin_costura[1:, :]
    canvas[borde] = (0, 0, 0)

    img = Image.fromarray(canvas, mode="RGB")
    if escala > 1:
        img = img.resize((w * escala, h * escala), Image.NEAREST)
    return img


# ── UI ────────────────────────────────────────────────────────────────────────
try:
    sr_model = cargar_modelo_sr()

    crop_geo, ts_str, ts_key, diurno = cargar_imagen_satelital()

    with st.spinner("✨ Mejorando imagen..."):
        crop_display = mejorar_imagen(crop_geo, sr_model)

    modo = "☀️ Día" if diurno else "🌙 Noche"
    st.caption(f"🕐 Última actualización: **{ts_str}** · GEOCOLOR ({modo})")

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
        tab_tabla, tab_mapa = st.tabs(["📊 Nubosidad", "🗺️ Mapa de nubes"])

        if not MAT_PATH.exists():
            with tab_tabla:
                st.warning(
                    "No se encontró **matriz de departamentos.xlsx**. "
                    "Subila al repositorio para activar el cálculo."
                )
        else:
            try:
                calculo_bytes            = imagen_a_bytes(crop_geo)
                dept_matrix, mascara_nube = calcular_mascara_nube(calculo_bytes, ts_key, diurno)

                with tab_tabla:
                    st.subheader("☁️ Nubosidad por departamento")
                    datos = calcular_tabla_nubosidad(dept_matrix, mascara_nube)
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

                with tab_mapa:
                    st.subheader("🗺️ Departamentos y nubes detectadas")
                    img_mascara = generar_imagen_mascara(dept_matrix, mascara_nube)
                    st.image(img_mascara, use_container_width=True)
                    st.download_button(
                        label="⬇️ Descargar mapa de nubes",
                        data=imagen_a_bytes(img_mascara, fmt="PNG"),
                        file_name="mapa_nubes.png",
                        mime="image/png",
                        use_container_width=False
                    )

            except Exception as e:
                st.error(f"Error en el cálculo de nubosidad: {e}")

except Exception as e:
    st.error(f"⚠️ Error al cargar la imagen: {e}")
