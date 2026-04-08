# Proyecto (Análisis Numérico): Volumen por Dataset

Este proyecto calcula el volumen de un sólido (aprox. sólido de revolución) a partir de mediciones `r(z)`.

## Idea

Si se mide el radio `r(z)` a distintas alturas `z`, el volumen se aproxima por:

V ≈ ∫ A(z) dz

con A(z) = π r(z)^2.

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

Además, el sistema puede generar/actualizar un dataset acumulado:

- `datasets/collected.csv`

## Ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Front local (Flask)
python webapp.py
```

Abre en el navegador:

- http://127.0.0.1:8000

## Flujo principal (recomendado): Imagen → collected.csv → Volumen

1. Toma una foto del **objeto de perfil** junto a una **regla de 30 cm** (la regla debe verse completa).
2. En la web, sube la imagen.
3. El sistema:
   - estima la escala usando la regla (pixeles → cm)
   - extrae un perfil `z_cm,r_cm`
   - hace append a `datasets/collected.csv`
   - calcula el volumen
4. Puedes descargar `collected.csv` desde el link en la página.

## API Key (Gemini) (opcional)

El proyecto está listo para integrar Gemini como fallback, pero **si no configuras la key**, el sistema funciona solo con el pipeline local.

Cuando tengas la key:

```bash
export GEMINI_API_KEY="TU_KEY"
```

## Modo CLI (por dataset)

```bash
python main.py --dataset datasets/example_r.csv --columns z_cm r_cm --area-from-radius --method simpson --resample-step 1
python main.py --dataset datasets/example_A.csv --columns z_cm A_cm2 --method trapezoidal
```

## Métodos

- Integración:
  - `trapezoidal`
  - `simpson`
- Interpolación (opcional para evaluar A(z) en nuevos puntos):
  - `newton`
  - `lagrange`
