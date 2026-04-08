# Proyecto (Análisis Numérico): Volumen por Dataset

Este proyecto calcula el volumen de un sólido (típicamente un sólido de revolución) a partir de un dataset de mediciones.

## Idea

Si se mide el radio `r(z)` (o el área `A(z)`) a distintas alturas `z`, el volumen se aproxima por:

V ≈ ∫ A(z) dz

con A(z) = π r(z)^2 (si el dataset trae radios).

## Dataset

Formato CSV recomendado:

### Opción A: `z_cm` y `r_cm`

Columnas:
- `z_cm`
- `r_cm`

### Opción B: `z_cm` y `A_cm2`

Columnas:
- `z_cm`
- `A_cm2`

Ejemplos en `datasets/example_r.csv` y `datasets/example_A.csv`.

## Ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --dataset datasets/example_r.csv --columns z_cm r_cm --area-from-radius --method simpson
python main.py --dataset datasets/example_A.csv --columns z_cm A_cm2 --method trapezoidal

# Front local (Flask)
python webapp.py
```

## Métodos

- Integración:
  - `trapezoidal`
  - `simpson`
- Interpolación (opcional para evaluar A(z) en nuevos puntos):
  - `newton`
  - `lagrange`
