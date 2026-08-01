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
# el umbral de brillo confunde con nube: apertura morfológica con un
# elemento estructurante en forma de CRUZ (4-conectividad: arriba, abajo,
# izquierda, derecha — sin diagonales). Con esa forma, un píxel que solo
# toca la línea en diagonal no tiene ningún vecino ortogonal que también
# sea parte de ella, así que la erosión la borra completa, sea recta o
# diagonal. Una nube real, aunque sea alargada o difusa (no compacta),
# sobrevive mientras tenga al menos algún píxel "macizo" en esas 4
# direcciones — a diferencia de filtrar por compacidad del rectángulo
# contenedor, esto no descarta nubes reales con forma irregular o alargada.

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


def filtrar_formas_finas(mascara: np.ndarray) -> np.ndarray:
    """Apertura morfológica (erosión + dilatación) con un elemento
    estructurante en forma de cruz: borra líneas de 1 píxel de ancho —
    como los límites provinciales dibujados por NOAA, rectas o
    diagonales— sin afectar nubes reales, aunque sean alargadas o de
    forma irregular."""
    estructura_cruz = ndimage.generate_binary_structure(2, 1)  # cruz (4-conectividad)
    return ndimage.binary_opening(mascara, structure=estructura_cruz, iterations=1)


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
def calcular_mascara_nube(img_bytes: bytes, ts_key: str, diurno: bool, mat_mtime: float):
    """Devuelve (dept_matrix, mascara_nube) para reusar tanto en la tabla
    de porcentajes como en la imagen de máscara.

    mat_mtime (fecha de modificación de MAT_PATH) se recibe solo para
    formar parte de la clave de caché: si se edita el xlsx, este valor
    cambia y st.cache_data recalcula en vez de devolver un resultado
    viejo cacheado con la matriz anterior."""
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
    # el umbral de brillo las toma como nube. Se descartan por su ancho
    # (1 píxel, incluso en diagonal) en vez de por su brillo, así no se
    # pierden nubes reales aunque sean alargadas o difusas.
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


def generar_imagen_con_limites(img_base: Image.Image, dept_matrix: np.ndarray) -> Image.Image:
    """Superpone en blanco, sobre la imagen satelital mejorada, los
    límites ENTRE departamentos (no el contorno externo de la provincia,
    que ya viene dibujado en la imagen satelital de NOAA). La máscara de
    bordes se calcula a la resolución de la matriz de deptos y se escala
    (NEAREST, para no generar bordes grises por interpolación) al tamaño
    de la imagen base, que suele tener mayor resolución por el
    mejorado/superresolución."""
    dept_sin_costura = _cerrar_costuras(dept_matrix)
    h, w = dept_sin_costura.shape
    es_valido = np.isin(dept_sin_costura, CODIGOS_VALIDOS)

    # Solo cuenta como límite si AMBOS lados son departamentos válidos y
    # distintos; si uno de los lados es "fuera del mapa" no se dibuja
    # (ese es el contorno provincial, ya presente en la imagen original).
    borde = np.zeros((h, w), dtype=bool)
    dif_horiz = (dept_sin_costura[:, :-1] != dept_sin_costura[:, 1:]) & \
                es_valido[:, :-1] & es_valido[:, 1:]
    dif_vert  = (dept_sin_costura[:-1, :] != dept_sin_costura[1:, :]) & \
                es_valido[:-1, :] & es_valido[1:, :]
    borde[:, :-1] |= dif_horiz
    borde[:-1, :] |= dif_vert

    # Se escalan por separado el borde y la zona "válida" a la resolución
    # final de la imagen (NEAREST, sin interpolar).
    borde_img = Image.fromarray((borde * 255).astype(np.uint8), mode="L")
    borde_img = borde_img.resize(img_base.size, Image.NEAREST)
    borde_arr = np.array(borde_img) > 127

    valido_img = Image.fromarray((es_valido * 255).astype(np.uint8), mode="L")
    valido_img = valido_img.resize(img_base.size, Image.NEAREST)
    valido_arr = np.array(valido_img) > 127

    # Donde un límite interno llega justo al borde de la provincia, se
    # extiende 1 SOLO píxel (ya en la resolución final) hacia la zona
    # "fuera de mapa" para que se una visualmente con el contorno de la
    # imagen satelital. Se hace acá, después de escalar, para que la
    # extensión sea de exactamente 1 píxel y no se agrande junto con el
    # factor de escala (lo que antes hacía que se saliera del contorno).
    borde_dilatado = ndimage.binary_dilation(borde_arr, iterations=1)
    borde_arr = borde_arr | (borde_dilatado & ~valido_arr)

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

    if not MAT_PATH.exists():
        st.warning(
            "No se encontró **matriz de departamentos.xlsx**. "
            "Subila al repositorio para activar el cálculo."
        )
    else:
        try:
            calculo_bytes             = imagen_a_bytes(crop_geo)
            mat_mtime                 = MAT_PATH.stat().st_mtime
            dept_matrix, mascara_nube = calcular_mascara_nube(calculo_bytes, ts_key, diurno, mat_mtime)

            col_img, col_tabla = st.columns([1, 1])

            with col_img:
                img_con_bordes = generar_imagen_con_limites(crop_display, dept_matrix)
                st.image(img_con_bordes, use_container_width=True)
                st.download_button(
                    label="⬇️ Descargar imagen con límites",
                    data=imagen_a_bytes(img_con_bordes, fmt="PNG"),
                    file_name="tucuman_con_limites.png",
                    mime="image/png",
                    use_container_width=False
                )

            with col_tabla:
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

        except Exception as e:
            st.error(f"Error en el cálculo de nubosidad: {e}")

except Exception as e:
    st.error(f"⚠️ Error al cargar la imagen: {e}")
