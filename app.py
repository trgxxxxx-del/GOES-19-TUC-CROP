"""
Calibración de umbrales Banda 13 – GOES-19
Ejecutar una vez para ver los valores reales de píxel por departamento.
"""

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
import requests

# ── Config ────────────────────────────────────────────────────────────────────
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

# Umbrales ACTUALES (para comparar)
LLUVIA_PIXELES = {
    "Tormenta fuerte": 40,
    "Lluvia fuerte":   70,
    "Lluvia moderada": 95,
    "Lluvia leve":     120,
    "Alta nubosidad":  150,
}

# ── Descarga ──────────────────────────────────────────────────────────────────
print("Descargando Banda 13...")
resp = requests.get(URL_BAND13, timeout=120)
resp.raise_for_status()
img  = Image.open(BytesIO(resp.content)).convert("L")
print(f"  Imagen original: {img.size[0]}×{img.size[1]} px, modo L")

crop_b13 = img.crop(CROP)
print(f"  Crop Tucumán:    {crop_b13.size[0]}×{crop_b13.size[1]} px")

# ── Matriz ────────────────────────────────────────────────────────────────────
df          = pd.read_excel(MAT_PATH, sheet_name=0, header=None)
dept_matrix = df.values.astype(int)
mat_h, mat_w = dept_matrix.shape

if crop_b13.size != (mat_w, mat_h):
    crop_b13 = crop_b13.resize((mat_w, mat_h), Image.LANCZOS)
    print(f"  Redimensionado a: {mat_w}×{mat_h}")

arr = np.array(crop_b13)

# ── Análisis global del crop ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"CROP COMPLETO  min={arr.min()}  max={arr.max()}  media={arr.mean():.1f}  std={arr.std():.1f}")
print(f"  Oscuro (< 80):  {(arr < 80).mean()*100:.1f}%  → nubes altas/lluvia")
print(f"  Medio (80-150): {((arr>=80)&(arr<150)).mean()*100:.1f}%  → nubosidad media")
print(f"  Claro (>= 150): {(arr>=150).mean()*100:.1f}%  → superficie/sin lluvia")

# ── Análisis por departamento ─────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"{'DEPARTAMENTO':<28} {'MIN':>5} {'MAX':>5} {'MEDIA':>6} {'STD':>5} | "
      f"{'<40':>5} {'40-70':>6} {'70-95':>6} {'95-120':>7} {'120-150':>8} {'>150':>5} | CATEGORÍA ACTUAL")
print(f"{'─'*28} {'─'*5} {'─'*5} {'─'*6} {'─'*5}   "
      f"{'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*8} {'─'*5}   {'─'*18}")

for nombre, codigo in DEPARTAMENTOS.items():
    mask  = dept_matrix == codigo
    total = int(np.sum(mask))
    if total == 0:
        continue

    px = arr[mask]

    mn   = int(px.min())
    mx   = int(px.max())
    med  = float(px.mean())
    std  = float(px.std())

    p = {
        "<40":     (px < 40).mean() * 100,
        "40-70":   ((px >= 40)  & (px < 70)).mean()  * 100,
        "70-95":   ((px >= 70)  & (px < 95)).mean()  * 100,
        "95-120":  ((px >= 95)  & (px < 120)).mean() * 100,
        "120-150": ((px >= 120) & (px < 150)).mean() * 100,
        ">150":    (px >= 150).mean() * 100,
    }

    # Categoría según umbrales actuales
    lim = LLUVIA_PIXELES
    cat = "Sin lluvia"
    if   p["<40"]     >= 2:  cat = "⛈️ Tormenta fuerte"
    elif p["40-70"]   >= 4:  cat = "🌧️ Lluvia fuerte"
    elif p["70-95"]   >= 6:  cat = "🌦️ Lluvia moderada"
    elif p["95-120"]  >= 15: cat = "🌂 Lluvia leve"
    elif p["120-150"] >= 10: cat = "☁️ Alta nubosidad"
    else:                    cat = "🌤️ Sin lluvia"

    print(f"{nombre:<28} {mn:>5} {mx:>5} {med:>6.1f} {std:>5.1f} | "
          f"{p['<40']:>5.1f} {p['40-70']:>6.1f} {p['70-95']:>6.1f} "
          f"{p['95-120']:>7.1f} {p['120-150']:>8.1f} {p['>150']:>5.1f} | {cat}")

# ── Sugerencia de calibración ────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("SUGERENCIA DE CALIBRACIÓN:")
print("  Si un departamento tiene llovizna confirmada pero aparece como 'Sin lluvia',")
print("  fijate en qué columna tiene más píxeles y bajá el umbral de esa categoría.")
print("  Ejemplo: si SMT tiene media ~160 y cae en >150 (Sin lluvia),")
print("           bajá Alta nubosidad de 150 → 170 y Lluvia leve de 120 → 145.")
print(f"{'─'*60}")
