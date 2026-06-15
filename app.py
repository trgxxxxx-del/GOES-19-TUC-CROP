from PIL import Image
import numpy as np

img = Image.open("ruta/a/tu/crop_b13.png").convert("L")  # o usá crop_b13 directo
arr = np.array(img)

# Código de SMT es 76 en tu matriz
# Cargá la matriz
import pandas as pd
df = pd.read_excel("matriz de departamentos.xlsx", sheet_name=0, header=None)
dept = df.values.astype(int)

mask_smt = dept == 76
pixeles_smt = arr[mask_smt]
print("SMT - min:", pixeles_smt.min(), "max:", pixeles_smt.max(), "media:", pixeles_smt.mean().round(1))
print("Histograma:", np.histogram(pixeles_smt, bins=8, range=(0,255)))
