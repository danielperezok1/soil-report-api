# Soil Report API (Córdoba - IDECOR)

API FastAPI que:
- recibe límite de campo (ZIP shapefile o KML/KMZ/GeoJSON)
- consulta WFS de IDECOR por bbox + paginación + tiles
- recorta (clip) al campo
- calcula hectáreas y porcentajes por unidad/serie
- genera mapas PNG y un informe DOCX

## Requisitos
Python 3.10+ recomendado.

## Instalación
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
