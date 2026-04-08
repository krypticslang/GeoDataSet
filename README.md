# Image-to-Volume (Análisis Numérico)

Aplicación local (Flask) para estimar el volumen de un sólido **aproximado como sólido de revolución** a partir de un perfil `r(z)`.

Puedes alimentar el sistema de dos formas:

- **Modo Imagen (recomendado)**: subes una foto (objeto + regla de 30 cm), se extrae el perfil `z_cm,r_cm`, se guarda en `datasets/collected.csv` y se calcula el volumen.
- **Modo CSV**: subes un dataset `z` y `r` (o `A`), y se integra directamente.

---

## Fundamento matemático

La aproximación usada es:

\[
V \approx \int_{z_{min}}^{z_{max}} A(z)\, dz
\]

donde:

- Si mides radio: \(A(z) = \pi r(z)^2\)
- Si ya tienes área: \(A(z)\) es tu dataset

Unidades:

- Si `z` está en **cm** y `A` en **cm²**, entonces `V` sale en **cm³**.

---

## Instalación y ejecución

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python webapp.py
```

Abrir en el navegador:

- http://127.0.0.1:8002

---

## Interfaz: Tabs + Wizard (orden de uso)

La UI está organizada en **tabs** para evitar ver todo al mismo tiempo:

- **Imagen**: flujo principal tipo wizard (subir → seleccionar objeto si aplica → ver resultado).
- **Vista**: inspección visual (imagen, overlay, máscara, escala estimada).
- **Integración**: muestra exactamente qué se integró (tabla de `z` y `A(z)` ya preparada, incluye remuestreo).
- **Datos**: tabla del dataset usado (`z`, `r`, `A`).
- **CSV**: modo alternativo para integrar un dataset directamente.

### Wizard (tab Imagen)

#### Paso 1: Subir imagen

Requisitos recomendados:

- Objeto de perfil.
- Regla de **30 cm completa** visible.
- Regla y objeto en el mismo plano.
- Buena iluminación y fondo contrastante.

Parámetros:

- `Paso de muestreo (cm)`: define el espaciado de `z` al extraer `r(z)`.
- `Método`: `trapezoidal` o `simpson`.
- `Remuestrear`: si activas Simpson, se recomienda remuestrear a malla uniforme.
- `Paso Δz`: tamaño del paso al remuestrear.

#### Paso 2: Seleccionar objeto (si aplica)

Si la segmentación detecta múltiples componentes, la app te muestra un selector con **outlines**.

Tú eliges el componente correcto y el sistema integra ese objeto.

#### Paso 3: Resultado

Se muestra:

- Volumen
- `z_min`, `z_max`
- cantidad de puntos
- método usado

---

## ¿Cómo funciona el pipeline de imagen?

En `image_to_profile.py`:

1) **Decodificar imagen**

2) **Estimación de escala `px_per_cm` usando la regla**

- Método local: detección de líneas con Canny + HoughLinesP y selección de la línea más larga
- Si falla el método local y está permitido, usa Gemini como fallback

3) **Segmentación (máscara) del objeto**

- Se usa un umbral automático (Otsu) sobre canal L (LAB) con realce por CLAHE
- Se limpia con morfología

4) **Mejora clave de robustez**

- Se elimina de la máscara la banda alrededor de la línea de la regla (para evitar que la regla domine la segmentación).

5) **Extracción de perfil `r(z)`**

- Se recorre `z` en pasos `step_cm`.
- Para cada `z`, se toma una fila horizontal del mask y se estima el ancho del objeto (en pixeles).
- Se convierte a cm con `px_per_cm` y se calcula el radio.

6) (Opcional) **Artefactos de debug**

Si activas debug, se guardan imágenes y metadata para depurar.

---

## Panel “Vista” (qué se midió)

Si activas **Guardar artefactos de debug**, el tab **Vista** muestra:

- Imagen original
- Overlay (contorno + regla)
- Máscara
- `px_per_cm` y fuente (`local` o `gemini`)

Esto sirve para validar rápidamente:

- Si la regla fue detectada correctamente
- Si el objeto fue segmentado correctamente

---

## Debug artifacts (bug fixing)

Cada corrida de debug crea:

- `debug_runs/<run_id>/meta.json`
- `debug_runs/<run_id>/input.jpg`
- `debug_runs/<run_id>/mask.png`
- `debug_runs/<run_id>/overlay.png`
- `debug_runs/<run_id>/profile.csv`

Y si hubo selección de objetos:

- `component_<id>.png` (máscara del componente)
- `preview_<id>.png` (overlay con contorno)

Puedes descargar todo como `.zip` desde el link en Resultado.

---

## Dataset CSV

Ejemplos incluidos:

- `datasets/example_r.csv`
- `datasets/example_A.csv`

Y dataset acumulado:

- `datasets/collected.csv`

Formatos:

### Opción A: `z_cm` y `r_cm`

- `z_cm`: altura
- `r_cm`: radio

### Opción B: `z_cm` y `A_cm2`

- `z_cm`: altura
- `A_cm2`: área

---

## Métodos numéricos implementados

En `webapp.py`:

### Integración

- **Trapezoidal compuesto** (`trapezoidal`)
- **Simpson compuesto** (`simpson`)
  - Requiere **espaciado uniforme**; por eso existe el remuestreo.

### Remuestreo

Si activas `Remuestrear a malla uniforme`, se interpola linealmente a un `Δz` fijo.

### Interpolación (modo CSV)

En el flujo CSV existen implementaciones de:

- Lagrange
- Newton (diferencias divididas)

Estas se usan para evaluar `A(z)` en un conjunto de puntos nuevo si habilitas esa opción.

---

## Gemini 1.5 (opcional) y seguridad anti-créditos

Gemini se usa **solo** como fallback para obtener los extremos de la regla en pixeles.

Por seguridad:

- Aunque tengas `GEMINI_API_KEY`, **no se llama** la API a menos que:
  - `GEMINI_ENABLE_CALLS=1`
  - y actives el switch “Usar Gemini como fallback” en el front

### Configuración

```bash
export GEMINI_API_KEY="TU_KEY"
export GEMINI_ENABLE_CALLS=1
```

---

## Endpoints (Flask)

- `GET /` UI
- `POST /compute/image` flujo imagen
- `POST /compute/csv` flujo CSV
- `GET /collected.csv` descargar dataset acumulado
- `GET /examples/<name>` descargar ejemplos
- `GET /debug/<run_id>.zip` descargar artefactos
- `GET /debug/<run_id>/<name>` servir assets de debug (png/jpg)

---

## Troubleshooting

### La máscara agarra la regla

- Activa “Guardar artefactos de debug”
- Ve a tab **Vista** y revisa `mask` y `overlay`
- Si hay múltiples componentes, usa el selector

### Simpson falla

Simpson requiere paso uniforme. Solución:

- Activa `Remuestrear a malla uniforme`

### Volumen demasiado grande

Revisa el tab **Integración**:

- Si `A(z)` es enorme, el perfil `r(z)` probablemente está sobreestimado (por perspectiva o segmentación).

