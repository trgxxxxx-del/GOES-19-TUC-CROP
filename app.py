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

    /* Alinea verticalmente las dos columnas (imagen satelital y mapa) */
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* Misma altura para ambas imágenes; el ancho se ajusta solo,
       preservando la relación de aspecto de cada una (sin deformar). */
    [data-testid="stImage"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stImage"] img {
        height: 520px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain;
        display: block;
        margin: auto;
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

# Filtro de forma para descartar líneas (límites provinciales de NOAA) que
# el umbral de brillo confunde con nube. Se agrupan píxeles conectados
# (incluida diagonal) en "manchas" y se descarta cada una según:
#   - AREA_MIN_NUBE: cantidad mínima de píxeles que debe tener la mancha.
#   - COMPACIDAD_MIN: área de la mancha / área de su rectángulo contenedor.
#     Una nube real es compacta (llena gran parte de su rectángulo); una
#     línea, aunque sea larga, ocupa un rectángulo mucho más grande que su
#     propia área (compacidad baja), incluso si es diagonal.
AREA_MIN_NUBE  = 2
COMPACIDAD_MIN = 0.4

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


def filtrar_formas_finas(mascara: np.ndarray,
                          area_min: int = AREA_MIN_NUBE,
                          compacidad_min: float = COMPACIDAD_MIN) -> np.ndarray:
    """Descarta manchas con forma de línea (los límites provinciales que
    trae dibujados el GEOCOLOR, que el umbral de brillo confunde con
    nube), preservando las manchas de nube reales.

    Se etiquetan los píxeles conectados (8-conectividad, para que una
    línea diagonal en 'escalera' quede unida en una sola mancha) y se
    descarta cada componente que sea chica o poco compacta: una línea,
    aunque sea larga, ocupa un rectángulo contenedor mucho más grande que
    su propia área (compacidad baja); una nube real llena buena parte de
    su rectángulo (compacidad alta). Esto detecta líneas diagonales
    completas, a diferencia de una simple apertura morfológica cuadrada
    que solo mira bloques locales de 2x2 y puede dejar pasar tramos en
    escalera."""
    estructura_8 = np.ones((3, 3), dtype=bool)
    etiquetas, n_etiquetas = ndimage.label(mascara, structure=estructura_8)
    if n_etiquetas == 0:
        return mascara

    salida = np.zeros_like(mascara)
    for slc, i in zip(ndimage.find_objects(etiquetas), range(1, n_etiquetas + 1)):
        if slc is None:
            continue
        comp = etiquetas[slc] == i
        area = int(comp.sum())
        area_rectangulo = comp.shape[0] * comp.shape[1]
        compacidad = area / area_rectangulo if area_rectangulo else 0.0
        if area >= area_min and compacidad >= compacidad_min:
            salida[slc][comp] = True

    return salida


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

    # Las líneas de límite provincial que NOAA dibuja sobre la imagen caen,
    # en algunos tramos, dentro del territorio válido de un departamento y
    # el umbral de brillo las toma como nube. Se descartan por su forma
    # (poco compactas, aunque sean diagonales) en vez de por su brillo,
    # así no se pierden nubes reales muy blancas.
    mascara_nube = filtrar_formas_finas(mascara_nube)

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

    # Solo se pinta como nube lo que cae DENTRO de algún departamento.
    # Fuera del mapa la imagen original trae contornos de provincias vecinas
    # (líneas grises/blancas) que el umbral de brillo confunde con nube;
    # como esa zona no entra en ningún departamento, se descarta acá.
    es_valido = np.isin(dept_matrix, CODIGOS_VALIDOS)
    canvas[mascara_nube & es_valido] = (150, 150, 150)

    borde = np.zeros((h, w), dtype=bool)
    borde[:, :-1] |= dept_sin_costura[:, :-1] != dept_sin_costura[:, 1:]
    borde[:-1, :] |= dept_sin_costura[:-1, :] != dept_sin_costura[1:, :]
    canvas[borde] = (0, 0, 0)

    img = Image.fromarray(canvas, mode="RGB")
    if escala > 1:
        img = img.resize((w * escala, h * escala), Image.NEAREST)
    return img


def generar_imagen_con_limites(img_base: Image.Image, dept_matrix: np.ndarray) -> Image.Image:
    """Superpone en blanco, sobre la imagen satelital mejorada, los
    límites entre departamentos. La máscara de bordes se calcula a la
    resolución de la matriz de deptos y se escala (NEAREST, para no
    generar bordes grises por interpolación) al tamaño de la imagen
    base, que suele tener mayor resolución por el mejorado/superresolución."""
    dept_sin_costura = _cerrar_costuras(dept_matrix)
    h, w = dept_sin_costura.shape

    borde = np.zeros((h, w), dtype=bool)
    borde[:, :-1] |= dept_sin_costura[:, :-1] != dept_sin_costura[:, 1:]
    borde[:-1, :] |= dept_sin_costura[:-1, :] != dept_sin_costura[1:, :]

    borde_img = Image.fromarray((borde * 255).astype(np.uint8), mode="L")
    borde_img = borde_img.resize(img_base.size, Image.NEAREST)
    borde_arr = np.array(borde_img) > 127

    resultado = np.array(img_base.convert("RGB")).copy()
    resultado[borde_arr] = (255, 255, 255)
    return Image.fromarray(resultado, mode="RGB")


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
        tab_tabla, tab_mapa, tab_bordes = st.tabs(
            ["📊 Nubosidad", "🗺️ Mapa de nubes", "🖼️ Imagen con límites"]
        )

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

                with tab_bordes:
                    st.subheader("🖼️ Imagen satelital con límites de departamentos")
                    img_con_bordes = generar_imagen_con_limites(crop_display, dept_matrix)
                    st.image(img_con_bordes, use_container_width=True)
                    st.download_button(
                        label="⬇️ Descargar imagen con límites",
                        data=imagen_a_bytes(img_con_bordes, fmt="PNG"),
                        file_name="tucuman_con_limites.png",
                        mime="image/png",
                        use_container_width=False
                    )

            except Exception as e:
                st.error(f"Error en el cálculo de nubosidad: {e}")

except Exception as e:
    st.error(f"⚠️ Error al cargar la imagen: {e}")
