import io
import requests
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path

URL  = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/ssa/GEOCOLOR/7200x4320.jpg"
CROP = (2679, 1344, 2985, 1639)
OUT  = Path("tucuman.png")
META = Path("last_update.txt")

print("Descargando GOES-19...")
resp = requests.get(URL, timeout=120)
resp.raise_for_status()

img  = Image.open(io.BytesIO(resp.content))
crop = img.crop(CROP)
crop.save(OUT, format="PNG")

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
META.write_text(ts)
print(f"Listo: {ts}")
