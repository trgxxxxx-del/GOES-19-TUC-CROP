import streamlit as st
from PIL import Image
from datetime import datetime, timezone, timedelta
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="DEBUG RGB", layout="wide")
st.title("🔬 DEBUG RGB por departamento - Band 13")

URL_BAND13 = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/13/7200x4320.jpg"
CROP       = (2717, 1382, 2932, 1600)
MAT_PATH   = Path("matriz de departamentos.xlsx")

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

try:
    with st.spinner("Descargando Band 13..."):
        resp = requests.get(URL_BAND13, timeout=120)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        crop = img.crop(CROP)

    st.image(crop, caption="Crop Band 13 actual", width=400)

    df         = pd.read_excel(MAT_PATH, sheet_name=0, header=None)
    mat        = df.values.astype(int)
    mat_h, mat_w = mat.shape

    if crop.size != (mat_w, mat_h):
        crop = crop.resize((mat_w, mat_h), Image.LANCZOS)

    arr = np.array(crop)
    st.write(f"Crop size: {crop.size} | Matriz shape: {mat.shape}")

    rows_data = []
    for nombre, codigo in DEPARTAMENTOS.items():
        mask = mat == codigo
        total = int(np.sum(mask))
        if total == 0:
            rows_data.append({"Departamento": nombre, "Código": codigo, "Píxeles": 0,
                               "R": "-", "G": "-", "B": "-", "Color aprox.": "—"})
            continue
        p = arr[mask]
        r, g, b = float(p[:,0].mean()), float(p[:,1].mean()), float(p[:,2].mean())

        if   r > 180 and g < 70  and b < 50:                  color = "🔴 Tormenta severa"
        elif r > 180 and g >= 70 and g < 160 and b < 30:      color = "🟠 Tormenta fuerte"
        elif r > 180 and g >= 160 and b < 30:                  color = "🟡 Lluvia fuerte"
        elif g > 120 and r < 120 and b < 60:                   color = "🟢 Lluvia moderada"
        elif b > 70  and r < 50  and g < 110:                  color = "🔵 Lluvia leve"
        elif b > 150 and g > 120 and r < 120:                  color = "🩵 Nubosidad alta"
        else:                                                   color = "⬜ Sin lluvia"

        rows_data.append({
            "Departamento": nombre,
            "Píxeles": total,
            "R mean": round(r, 1),
            "G mean": round(g, 1),
            "B mean": round(b, 1),
            "Clasificación": color,
        })

    df_out = pd.DataFrame(rows_data)
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # Muestra de píxeles individuales de San Miguel
    st.subheader("Píxeles individuales - San Miguel de Tucumán (código 76)")
    mask_smt = mat == 76
    p_smt = arr[mask_smt]
    df_smt = pd.DataFrame(p_smt, columns=["R", "G", "B"])
    st.dataframe(df_smt, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")
