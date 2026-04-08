from __future__ import annotations

import io
import json
from pathlib import Path
import uuid
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template_string, request, url_for

import cv2

from image_to_profile import connected_components, profile_from_image_bytes, profile_from_image_bytes_with_debug, profile_from_mask


@dataclass(frozen=True)
class ComputeResult:
    volume: float
    z_min: float
    z_max: float
    n: int
    method: str
    z_used: list[float] | None = None
    A_used: list[float] | None = None


@dataclass(frozen=True)
class TableRow:
    z: float
    r: float | None
    A: float


@dataclass(frozen=True)
class ProfileRow:
    z: float
    r: float


def composite_trapezoidal(x: np.ndarray, f: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = getattr(np, "trapz")
    return float(trapezoid(f, x))


def composite_simpson_uniform(x: np.ndarray, f: np.ndarray) -> float:
    if x.size < 3:
        raise ValueError("Simpson requires at least 3 points")

    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-6, atol=1e-9):
        raise ValueError("Simpson requires uniform spacing. Enable resampling.")

    n = int(x.size)
    if (n - 1) % 2 != 0:
        x = x[:-1]
        f = f[:-1]
        n -= 1

    h0 = float(h[0])
    s = f[0] + f[-1]
    s += 4.0 * float(np.sum(f[1:-1:2]))
    s += 2.0 * float(np.sum(f[2:-2:2]))
    return float((h0 / 3.0) * s)


def lagrange_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    y = y.astype(float)
    xq = xq.astype(float)
    n = x.size
    out = np.zeros_like(xq, dtype=float)

    for i in range(n):
        li = np.ones_like(xq, dtype=float)
        for j in range(n):
            if i == j:
                continue
            li *= (xq - x[j]) / (x[i] - x[j])
        out += y[i] * li

    return out


def newton_divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    coef = y.astype(float).copy()
    n = x.size
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1 : n - 1]) / (x[j:n] - x[0 : n - j])
    return coef


def newton_evaluate(coef: np.ndarray, x_data: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xq = xq.astype(float)
    n = coef.size
    p = np.full_like(xq, coef[n - 1], dtype=float)
    for k in range(n - 2, -1, -1):
        p = p * (xq - x_data[k]) + coef[k]
    return p


def resample_uniform(x: np.ndarray, y: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("step must be > 0")
    x_new = np.arange(float(x[0]), float(x[-1]) + 0.5 * step, float(step), dtype=float)
    y_new = np.interp(x_new, x, y).astype(float)
    return x_new, y_new


def compute_from_df(
    df: pd.DataFrame,
    *,
    col_z: str,
    col_y: str,
    y_is_radius: bool,
    method: str,
    resample: bool,
    step: float,
    interpolation: str,
    interp_n: int,
) -> ComputeResult:
    z = df[col_z].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)
    order = np.argsort(z)
    z = z[order]
    y = y[order]

    if y_is_radius:
        A = np.pi * (y**2)
    else:
        A = y

    if resample:
        z, A = resample_uniform(z, A, float(step))

    if interpolation != "none" and int(interp_n) > 1:
        zq = np.linspace(float(z[0]), float(z[-1]), int(interp_n), dtype=float)
        if interpolation == "lagrange":
            Aq = lagrange_interpolate(z, A, zq)
        else:
            coef = newton_divided_differences(z, A)
            Aq = newton_evaluate(coef, z, zq)
        z, A = zq, Aq

    if method == "trapezoidal":
        V = composite_trapezoidal(z, A)
    elif method == "simpson":
        V = composite_simpson_uniform(z, A)
    else:
        raise ValueError("Unknown method")

    return ComputeResult(
        volume=float(V),
        z_min=float(z[0]),
        z_max=float(z[-1]),
        n=int(z.size),
        method=method,
        z_used=[float(v) for v in z.tolist()],
        A_used=[float(v) for v in A.tolist()],
    )


def volume_from_profile(z_cm: np.ndarray, r_cm: np.ndarray, *, method: str, resample: bool, step: float) -> ComputeResult:
    z = z_cm.astype(float)
    r = r_cm.astype(float)
    order = np.argsort(z)
    z = z[order]
    r = r[order]

    A = np.pi * (r**2)
    if resample:
        z, A = resample_uniform(z, A, step=float(step))

    if method == "trapezoidal":
        V = composite_trapezoidal(z, A)
    else:
        V = composite_simpson_uniform(z, A)

    return ComputeResult(
        volume=float(V),
        z_min=float(z[0]),
        z_max=float(z[-1]),
        n=int(z.size),
        method=method,
        z_used=[float(v) for v in z.tolist()],
        A_used=[float(v) for v in A.tolist()],
    )


def table_rows_from_df(
    df: pd.DataFrame,
    *,
    col_z: str,
    col_y: str,
    y_is_radius: bool,
) -> list[TableRow]:
    z = df[col_z].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)
    order = np.argsort(z)
    z = z[order]
    y = y[order]

    if y_is_radius:
        r = y
        A = np.pi * (r**2)
        rows = [TableRow(z=float(zi), r=float(ri), A=float(ai)) for zi, ri, ai in zip(z, r, A)]
    else:
        A = y
        rows = [TableRow(z=float(zi), r=None, A=float(ai)) for zi, ai in zip(z, A)]
    return rows


HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image-to-Volume</title>
  <style>
    *{box-sizing:border-box;}
    :root{
      --bg:#0b1020;
      --panel:rgba(255,255,255,.06);
      --panel2:rgba(255,255,255,.04);
      --fg:#eef2ff;
      --muted:rgba(238,242,255,.72);
      --muted2:rgba(238,242,255,.55);
      --accent:#7c3aed;
      --accent2:#22c55e;
      --border:rgba(255,255,255,.12);
      --shadow: 0 18px 50px rgba(0,0,0,.35);
    }
    body{margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background:
        radial-gradient(900px 500px at 20% 10%, rgba(124,58,237,.28), transparent 55%),
        radial-gradient(700px 450px at 85% 25%, rgba(34,197,94,.22), transparent 60%),
        var(--bg);
      color:var(--fg);
    }
    .wrap{max-width:1180px; margin:0 auto; padding:26px 18px 72px;}
    .top{
      display:flex; align-items:flex-start; justify-content:space-between; gap:14px; flex-wrap:wrap;
      margin-bottom:14px;
    }
    h1{font-size:22px; margin:0; letter-spacing:-0.2px;}
    .sub{color:var(--muted); margin:4px 0 0; font-size:13px; line-height:1.35; max-width:820px;}
    .links{display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
    .chip{display:inline-flex; align-items:center; gap:8px; padding:8px 10px; border-radius:999px; border:1px solid var(--border);
      background:rgba(255,255,255,.04); color:var(--fg); font-size:12px; text-decoration:none;
    }
    .chip:hover{filter:brightness(1.05);}
    .grid{display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap:14px; align-items:start;}
    .card{background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:16px; overflow:hidden; backdrop-filter: blur(8px); box-shadow:var(--shadow);}
    .section-title{margin:0 0 10px; font-size:12px; color:var(--muted2); letter-spacing:0.2px; text-transform:uppercase;}
    label{display:block; font-size:12px; color:var(--muted); margin:10px 0 6px;}
    input, select{width:100%; padding:11px 12px; border-radius:12px; border:1px solid var(--border);
      background:rgba(0,0,0,.18); color:var(--fg); outline:none; appearance:none;
    }
    input:focus, select:focus{border-color:rgba(124,58,237,.55); box-shadow:0 0 0 4px rgba(124,58,237,.16);}
    .row{display:grid; grid-template-columns: 1fr 1fr; gap:10px;}
    .btn{margin-top:12px; width:100%; padding:12px; border-radius:14px; border:1px solid rgba(124,58,237,.55);
      background:linear-gradient(180deg, rgba(124,58,237,.42), rgba(124,58,237,.20)); color:var(--fg); cursor:pointer; font-weight:750;
    }
    .btn:hover{filter:brightness(1.06);}
    .btn.secondary{border-color:rgba(255,255,255,.20); background:rgba(255,255,255,.06); font-weight:650;}
    .btn.secondary:hover{filter:brightness(1.08);}
    .kpi{display:grid; grid-template-columns: 1fr 1fr; gap:10px;}
    .k{background:var(--panel2); border:1px solid var(--border); border-radius:14px; padding:10px 12px;}
    .k .l{color:var(--muted2); font-size:12px;}
    .k .v{font-size:18px; font-weight:750; margin-top:4px; letter-spacing:-0.2px;}
    .err{border:1px solid rgba(255,90,90,.42); background:rgba(255,90,90,.10); padding:10px 12px; border-radius:12px;}
    .ok{border:1px solid rgba(34,197,94,.35); background:rgba(34,197,94,.10); padding:10px 12px; border-radius:12px;}
    .hint{color:var(--muted2); font-size:12px; margin-top:10px;}
    .small{color:var(--muted); font-size:12px;}
    a{color:rgba(167,139,250,.95);}

    .table-wrap{width:100%; overflow:auto; margin-top:10px; border-radius:14px; border:1px solid var(--border); background:rgba(0,0,0,.16);}
    table{width:100%; min-width:720px; border-collapse:separate; border-spacing:0;}
    thead th{position:sticky; top:0; font-size:12px; color:var(--muted); font-weight:800; text-align:left; padding:12px 10px;
      border-bottom:1px solid var(--border);
      background:rgba(15,18,32,.92);
      backdrop-filter: blur(6px);
    }
    tbody td{padding:12px 10px; border-bottom:1px solid rgba(255,255,255,.08);}
    tbody tr:hover{background:rgba(124,58,237,.08);}
    .mono{font-variant-numeric: tabular-nums;}
    .pill{display:inline-block; padding:3px 10px; border-radius:999px; border:1px solid var(--border);
      background:rgba(255,255,255,.05); color:var(--fg); font-size:12px;}

    .switch-row{display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px;}
    .switch-row .txt{display:flex; flex-direction:column; gap:2px;}
    .switch-row .txt .t{color:var(--muted); font-size:12px;}
    .switch-row .txt .d{color:rgba(215,255,224,.85); font-size:12px;}

    .switch{position:relative; display:inline-block; width:44px; height:26px; flex:0 0 auto;}
    .switch input{opacity:0; width:0; height:0;}
    .slider{position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0;
      background:rgba(255,255,255,.10); border:1px solid var(--border); transition:.2s; border-radius:999px;
    }
    .slider:before{position:absolute; content:""; height:20px; width:20px; left:3px; top:2px;
      background:rgba(215,255,224,.9); transition:.2s; border-radius:999px;
    }
    .switch input:checked + .slider{background:rgba(34,197,94,.22);}
    .switch input:checked + .slider:before{transform:translateX(18px); background:rgba(34,197,94,.95);}

    details{border:1px solid var(--border); border-radius:14px; background:rgba(255,255,255,.03); padding:10px 12px; margin-top:12px;}
    summary{cursor:pointer; color:var(--muted); font-size:12px; font-weight:800; list-style:none;}
    summary::-webkit-details-marker{display:none;}
    .divider{height:1px; background:rgba(255,255,255,.10); margin:14px 0;}
    .preview{
      border:1px solid var(--border); border-radius:14px; background:rgba(255,255,255,.03); overflow:hidden;
    }
    .preview .t{padding:10px 10px 0;}
    .preview img{display:block; width:100%; height:auto;}
    .muted{color:var(--muted2);}

    .tabs{display:flex; gap:10px; flex-wrap:wrap; margin:14px 0;}
    .tab{
      border:1px solid var(--border);
      background:rgba(255,255,255,.04);
      color:var(--fg);
      border-radius:999px;
      padding:9px 12px;
      font-size:12px;
      font-weight:800;
      cursor:pointer;
    }
    .tab[aria-selected="true"]{
      border-color:rgba(124,58,237,.60);
      background:rgba(124,58,237,.22);
    }
    .panel{display:none;}
    .panel.active{display:block;}
    .stepper{display:flex; gap:10px; flex-wrap:wrap; margin:0 0 12px;}
    .step{
      display:flex; align-items:center; gap:8px;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      font-size:12px;
      color:var(--muted);
      font-weight:800;
    }
    .step .n{
      width:20px; height:20px; border-radius:999px;
      display:inline-flex; align-items:center; justify-content:center;
      background:rgba(255,255,255,.10);
      color:var(--fg);
      border:1px solid var(--border);
      font-size:12px;
      flex:0 0 auto;
    }
    .step.done{color:rgba(238,242,255,.92); border-color:rgba(34,197,94,.35); background:rgba(34,197,94,.10);}
    .step.done .n{background:rgba(34,197,94,.20); border-color:rgba(34,197,94,.40);}

    @media (max-width: 980px){
      .grid{grid-template-columns: 1fr;}
      table{min-width: 640px;}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1>Image-to-Volume</h1>
        <p class="sub">Sube una imagen con el objeto y una regla de 30 cm. El sistema detecta la regla para escalar (px → cm), segmenta el objeto, extrae `r(z)` y calcula el volumen por integración numérica.</p>
      </div>
      <div class="links">
        <a class="chip" href="{{ url_for('download_collected') }}">Descargar collected.csv</a>
        <a class="chip" href="{{ url_for('download_example', name='example_r.csv') }}">Ejemplo r(z)</a>
        <a class="chip" href="{{ url_for('download_example', name='example_A.csv') }}">Ejemplo A(z)</a>
      </div>
    </div>

    {% if message %}
      <div class="{{ 'err' if is_error else 'ok' }}">{{ message }}</div>
    {% endif %}

    <div class="tabs" role="tablist" aria-label="Navegación">
      <button class="tab" type="button" role="tab" data-tab="image" aria-selected="true">Imagen</button>
      <button class="tab" type="button" role="tab" data-tab="view" aria-selected="false">Vista</button>
      <button class="tab" type="button" role="tab" data-tab="integrand" aria-selected="false">Integración</button>
      <button class="tab" type="button" role="tab" data-tab="data" aria-selected="false">Datos</button>
      <button class="tab" type="button" role="tab" data-tab="csv" aria-selected="false">CSV</button>
    </div>

    <div class="panel" id="panel-image" role="tabpanel">
      <div class="stepper" aria-label="Wizard">
        <div class="step done"><span class="n">1</span><span>Sube imagen</span></div>
        <div class="step {% if component_choices %}done{% endif %}"><span class="n">2</span><span>Selecciona objeto</span></div>
        <div class="step {% if result %}done{% endif %}"><span class="n">3</span><span>Resultado</span></div>
      </div>

      <div class="grid">
        <div class="card">
          <form method="post" action="{{ url_for('compute_image') }}" enctype="multipart/form-data">
            <div class="section-title">Paso 1 · Imagen</div>
            <label>Imagen (objeto + regla de 30 cm)</label>
            <input id="imageInput" type="file" name="image" accept="image/*" required />

            <div id="clientPreview" class="preview" style="margin-top:12px; display:none;">
              <div class="small t">Previsualización</div>
              <img id="clientPreviewImg" alt="preview" />
            </div>

            <div class="row" style="margin-top:10px;">
              <div>
                <label>Paso de muestreo (cm)</label>
                <input name="step_cm" type="number" step="0.1" value="1.0" />
              </div>
              <div>
                <label>Método</label>
                <select name="method">
                  <option value="trapezoidal">trapezoidal</option>
                  <option value="simpson">simpson</option>
                </select>
              </div>
            </div>

            <details open>
              <summary>Ajustes</summary>
              <div class="switch-row">
                <div class="txt">
                  <div class="t">Remuestrear a malla uniforme</div>
                  <div class="d">Útil para Simpson si z no es uniforme.</div>
                </div>
                <label class="switch" aria-label="remuestrear">
                  <input type="checkbox" name="resample" value="yes" />
                  <span class="slider"></span>
                </label>
              </div>

              <label>Paso Δz (si remuestreas)</label>
              <input name="step" type="number" step="0.01" value="1.0" />

              <div class="switch-row">
                <div class="txt">
                  <div class="t">Usar Gemini como fallback</div>
                  <div class="d">Solo si falla la detección local. Requiere `GEMINI_API_KEY` y `GEMINI_ENABLE_CALLS=1`.</div>
                </div>
                <label class="switch" aria-label="gemini-fallback">
                  <input type="checkbox" name="gemini_fallback" value="yes" />
                  <span class="slider"></span>
                </label>
              </div>

              <div class="switch-row">
                <div class="txt">
                  <div class="t">Guardar artefactos de debug</div>
                  <div class="d">Guarda imagen, máscara y overlay para bug fixing.</div>
                </div>
                <label class="switch" aria-label="save-debug">
                  <input type="checkbox" name="save_debug" value="yes" />
                  <span class="slider"></span>
                </label>
              </div>
            </details>

            <button class="btn" type="submit">Extraer y calcular</button>
            <div class="hint">Tip: iluminación uniforme + regla completa en el mismo plano que el objeto.</div>
          </form>
        </div>

        <div class="card">
          <div class="section-title">Paso 3 · Resultado</div>
          {% if result %}
            <div class="kpi">
              <div class="k"><div class="l">Volumen estimado</div><div class="v">{{ '%.6f'|format(result.volume) }}</div></div>
              <div class="k"><div class="l">Método</div><div class="v">{{ result.method }}</div></div>
              <div class="k"><div class="l">z min</div><div class="v">{{ '%.6f'|format(result.z_min) }}</div></div>
              <div class="k"><div class="l">z max</div><div class="v">{{ '%.6f'|format(result.z_max) }}</div></div>
              <div class="k"><div class="l">n puntos</div><div class="v">{{ result.n }}</div></div>
            </div>
            <p class="small" style="margin-top:10px;">Unidades: cm³ si `z` está en cm y `A` en cm².</p>
            {% if debug_zip_url %}
              <p class="small" style="margin-top:10px;">Debug: <a href="{{ debug_zip_url }}">descargar artefactos (.zip)</a></p>
            {% endif %}
          {% else %}
            <p class="small">Aún no hay resultados. Sube una imagen y ejecuta el cálculo.</p>
          {% endif %}
        </div>
      </div>

      {% if component_choices %}
        <div class="card" style="margin-top:14px;">
          <div class="section-title">Paso 2 · Selección de objeto</div>
          <div class="small">Detecté múltiples objetos. Elige cuál integrar (verás el contorno).</div>
          <form method="post" action="{{ url_for('compute_image') }}" style="margin-top:12px;">
            <input type="hidden" name="run_id" value="{{ run_id }}" />
            <input type="hidden" name="step_cm" value="{{ step_cm }}" />
            <input type="hidden" name="method" value="{{ method }}" />
            <input type="hidden" name="resample" value="{{ 'yes' if resample else '' }}" />
            <input type="hidden" name="step" value="{{ step }}" />
            <input type="hidden" name="gemini_fallback" value="{{ 'yes' if gemini_fallback else '' }}" />
            <input type="hidden" name="save_debug" value="{{ 'yes' if save_debug else '' }}" />

            <div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:10px;">
              {% for c in component_choices %}
                <label style="display:block; border:1px solid var(--border); background:rgba(255,255,255,.03); border-radius:14px; padding:10px; cursor:pointer;">
                  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
                    <div class="small">Componente {{ c.component_id }} · área {{ c.area_px }} px</div>
                    <input type="radio" name="component_id" value="{{ c.component_id }}" {% if loop.index0 == 0 %}checked{% endif %} />
                  </div>
                  <div style="margin-top:10px; overflow:hidden; border-radius:12px; border:1px solid var(--border);">
                    <img alt="preview" src="{{ c.preview_url }}" style="display:block; width:100%; height:auto;" />
                  </div>
                </label>
              {% endfor %}
            </div>
            <button class="btn" type="submit" style="margin-top:12px;">Usar objeto seleccionado</button>
          </form>
        </div>
      {% endif %}
    </div>

    <div class="panel" id="panel-view" role="tabpanel">
      <div class="card">
        <div class="section-title">Vista</div>
        {% if view_assets %}
          <div class="small">Verifica lo que se midió: regla detectada, máscara y contorno.</div>
          <div class="kpi" style="margin-top:12px;">
            <div class="k"><div class="l">px por cm</div><div class="v">{{ '%.3f'|format(view_assets.px_per_cm) }}</div></div>
            <div class="k"><div class="l">fuente escala</div><div class="v">{{ view_assets.px_per_cm_source }}</div></div>
          </div>
          <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:12px; margin-top:12px;">
            <div class="preview"><div class="small t">Imagen</div><img alt="input" src="{{ view_assets.input_url }}" /></div>
            <div class="preview"><div class="small t">Overlay</div><img alt="overlay" src="{{ view_assets.overlay_url }}" /></div>
            <div class="preview"><div class="small t">Máscara</div><img alt="mask" src="{{ view_assets.mask_url }}" /></div>
          </div>
          {% if debug_zip_url %}
            <p class="small" style="margin-top:10px;">Debug: <a href="{{ debug_zip_url }}">descargar artefactos (.zip)</a></p>
          {% endif %}
        {% else %}
          <div class="small">Activa “Guardar artefactos de debug” y ejecuta un cálculo para ver aquí la imagen, overlay y máscara.</div>
        {% endif %}
      </div>
    </div>

    <div class="panel" id="panel-integrand" role="tabpanel">
      <div class="card">
        <div class="section-title">Integración</div>
        {% if integrand_rows %}
          <div class="small">Mostrando z y A(z) ya preparados (incluye remuestreo si aplicó).</div>
          <div class="table-wrap" style="margin-top:10px;">
            <table class="mono" style="min-width:560px;">
              <thead>
                <tr>
                  <th>z (cm)</th>
                  <th>A(z) (cm²)</th>
                </tr>
              </thead>
              <tbody>
                {% for row in integrand_rows %}
                  <tr>
                    <td>{{ '%.6f'|format(row[0]) }}</td>
                    <td>{{ '%.6f'|format(row[1]) }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div class="small">Aún no hay datos de integración. Ejecuta un cálculo primero.</div>
        {% endif %}
      </div>
    </div>

    <div class="panel" id="panel-data" role="tabpanel">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
          <div>
            <div class="section-title">Datos</div>
            <div class="small">Vista previa del dataset y el cálculo de área.</div>
          </div>
          {% if y_mode %}
            <span class="pill">y = {{ y_mode }}</span>
          {% endif %}
        </div>

        {% if rows %}
          <div class="table-wrap">
            <table class="mono">
              <thead>
                <tr>
                  <th>Altura (h en cm)</th>
                  <th>Radio (r en cm)</th>
                  <th>Área (A = π · r²)</th>
                </tr>
              </thead>
              <tbody>
                {% for row in rows %}
                  <tr>
                    <td>{{ '%.6f'|format(row.z) }}</td>
                    <td>
                      {% if row.r is none %}
                        —
                      {% else %}
                        {{ '%.6f'|format(row.r) }}
                      {% endif %}
                    </td>
                    <td>{{ '%.6f'|format(row.A) }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div class="small" style="margin-top:10px;">Aún no hay datos. Ejecuta un cálculo para ver la tabla.</div>
        {% endif %}
      </div>
    </div>

    <div class="panel" id="panel-csv" role="tabpanel">
      <div class="card">
        <div class="section-title">CSV</div>
        <div class="small">Modo CSV para cuando ya tienes un dataset `z` + `r` o `A`.</div>
        <form method="post" action="{{ url_for('compute_csv') }}" enctype="multipart/form-data" style="margin-top:10px;">
          <label>Dataset CSV</label>
          <input type="file" name="file" accept=".csv" required />

          <div class="row">
            <div>
              <label>Columna z</label>
              <input name="col_z" placeholder="z_cm" value="z_cm" />
            </div>
            <div>
              <label>Columna y</label>
              <input name="col_y" placeholder="A_cm2 o r_cm" value="r_cm" />
            </div>
          </div>

          <label>Qué representa y</label>
          <select name="y_mode">
            <option value="radius">radio r(z) (A=πr²)</option>
            <option value="area">área A(z)</option>
          </select>

          <label>Método de integración</label>
          <select name="method">
            <option value="trapezoidal">trapezoidal</option>
            <option value="simpson">simpson</option>
          </select>

          <div class="switch-row">
            <div class="txt">
              <div class="t">Remuestrear a malla uniforme</div>
              <div class="d">Recomendado para Simpson si tus z no están uniformes.</div>
            </div>
            <label class="switch" aria-label="remuestrear">
              <input type="checkbox" name="resample" value="yes" />
              <span class="slider"></span>
            </label>
          </div>

          <label>Paso Δz (si remuestreas)</label>
          <input name="step" type="number" step="0.01" value="1.0" />

          <button class="btn secondary" type="submit">Calcular con CSV</button>
        </form>
      </div>
    </div>

    <div class="divider"></div>
    <p class="small muted">Sugerencia práctica: si el objeto y la regla aparecen como un solo bloque en la máscara, activa “Guardar artefactos de debug” para revisar el overlay y la máscara.</p>
  </div>

  <script>
    (function(){
      const input = document.getElementById('imageInput');
      const wrap = document.getElementById('clientPreview');
      const img = document.getElementById('clientPreviewImg');
      if(!input || !wrap || !img) return;
      input.addEventListener('change', () => {
        const f = input.files && input.files[0];
        if(!f){ wrap.style.display='none'; return; }
        const url = URL.createObjectURL(f);
        img.src = url;
        wrap.style.display='block';
      });
    })();

    (function(){
      const tabs = Array.from(document.querySelectorAll('.tab'));
      const panels = {
        image: document.getElementById('panel-image'),
        view: document.getElementById('panel-view'),
        integrand: document.getElementById('panel-integrand'),
        data: document.getElementById('panel-data'),
        csv: document.getElementById('panel-csv'),
      };

      function pickDefault(){
        const hasView = {{ 'true' if view_assets else 'false' }};
        const hasIntegrand = {{ 'true' if integrand_rows else 'false' }};
        const hasRows = {{ 'true' if rows else 'false' }};
        const hasResult = {{ 'true' if result else 'false' }};
        const hasComponents = {{ 'true' if component_choices else 'false' }};
        if(hasComponents) return 'image';
        if(hasView) return 'view';
        if(hasResult) return 'image';
        if(hasIntegrand) return 'integrand';
        if(hasRows) return 'data';
        return 'image';
      }

      function setTab(id){
        if(!panels[id]) id = 'image';
        tabs.forEach(t => t.setAttribute('aria-selected', (t.dataset.tab === id) ? 'true' : 'false'));
        Object.entries(panels).forEach(([k, el]) => {
          if(!el) return;
          el.classList.toggle('active', k === id);
        });
        try{ localStorage.setItem('itv_tab', id); } catch(e){}
        if(location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
      }

      tabs.forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
      const fromHash = (location.hash || '').replace('#','');
      let initial = fromHash;
      if(!initial){
        try{ initial = localStorage.getItem('itv_tab') || ''; } catch(e){}
      }
      if(!initial) initial = pickDefault();
      setTab(initial);
    })();
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = uuid.uuid4().hex

    datasets_dir = Path(__file__).resolve().parent / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    collected_path = datasets_dir / "collected.csv"

    debug_dir = Path(__file__).resolve().parent / "debug_runs"
    debug_dir.mkdir(exist_ok=True)

    def _zip_dir(path: Path) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in path.rglob("*"):
                if p.is_dir():
                    continue
                zf.write(p, arcname=str(p.relative_to(path)))
        return buf.getvalue()

    @app.get("/")
    def index() -> str:
        return render_template_string(
            HTML,
            result=None,
            rows=None,
            y_mode=None,
            message=None,
            is_error=False,
            debug_zip_url=None,
            integrand_rows=None,
            component_choices=None,
            run_id=None,
            step_cm=None,
            method=None,
            resample=None,
            step=None,
            gemini_fallback=None,
            save_debug=None,
            view_assets=None,
        )

    @app.get("/debug/<run_id>.zip")
    def download_debug(run_id: str) -> Response:
        safe = "".join([c for c in run_id if c.isalnum() or c in ("-", "_")])
        path = debug_dir / safe
        if not path.exists() or not path.is_dir():
            return Response("Not found", status=404)
        data = _zip_dir(path)
        return Response(
            data,
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename=debug_{safe}.zip"},
        )

    @app.get("/debug/<run_id>/<name>")
    def debug_asset(run_id: str, name: str) -> Response:
        safe = "".join([c for c in run_id if c.isalnum() or c in ("-", "_")])
        safe_name = "".join([c for c in name if c.isalnum() or c in ("-", "_", ".")])
        path = debug_dir / safe / safe_name
        if not path.exists() or not path.is_file():
            return Response("Not found", status=404)
        if safe_name.lower().endswith(".png"):
            mt = "image/png"
        elif safe_name.lower().endswith(".jpg") or safe_name.lower().endswith(".jpeg"):
            mt = "image/jpeg"
        else:
            mt = "application/octet-stream"
        return Response(path.read_bytes(), mimetype=mt)

    @app.post("/compute/csv")
    def compute_csv() -> str:
        if "file" not in request.files:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode=None,
                message="Falta archivo.",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        f = request.files["file"]
        if not f.filename:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode=None,
                message="Archivo inválido.",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        try:
            df = pd.read_csv(io.BytesIO(f.read()))
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode=None,
                message=f"No se pudo leer CSV: {e}",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        col_z = request.form.get("col_z", "z_cm")
        col_y = request.form.get("col_y", "A_cm2")
        if col_z not in df.columns or col_y not in df.columns:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode=None,
                message=f"Columnas no encontradas. Disponibles: {', '.join(df.columns)}",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        y_mode = request.form.get("y_mode", "area")
        y_is_radius = y_mode == "radius"

        method = request.form.get("method", "trapezoidal")
        resample = request.form.get("resample") == "yes"
        step = float(request.form.get("step", "1.0"))
        interpolation = request.form.get("interpolation", "none")
        interp_n = int(request.form.get("interp_n", "0") or 0)

        try:
            rows = table_rows_from_df(df, col_z=col_z, col_y=col_y, y_is_radius=y_is_radius)
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode=y_mode,
                message=f"No se pudo preparar la tabla: {e}",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        try:
            result = compute_from_df(
                df,
                col_z=col_z,
                col_y=col_y,
                y_is_radius=y_is_radius,
                method=method,
                resample=resample,
                step=step,
                interpolation=interpolation,
                interp_n=interp_n,
            )
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=rows,
                y_mode=y_mode,
                message=str(e),
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        return render_template_string(
            HTML,
            result=result,
            rows=rows,
            y_mode=y_mode,
            message="Cálculo completado.",
            is_error=False,
            debug_zip_url=None,
            integrand_rows=(list(zip(result.z_used, result.A_used))[:200] if result.z_used and result.A_used else None),
            component_choices=None,
            run_id=None,
            step_cm=None,
            method=None,
            resample=None,
            step=None,
            gemini_fallback=None,
            save_debug=None,
            view_assets=None,
        )

    @app.post("/compute/image")
    def compute_image() -> str:
        run_id_in = request.form.get("run_id", "").strip()
        component_id_in = request.form.get("component_id", "").strip()

        if run_id_in and component_id_in:
            safe = "".join([c for c in run_id_in if c.isalnum() or c in ("-", "_")])
            run_path = debug_dir / safe
            if not run_path.exists():
                return render_template_string(
                    HTML,
                    result=None,
                    rows=None,
                    y_mode="radius",
                    message="Sesión de selección inválida (debug run no existe).",
                    is_error=True,
                    debug_zip_url=None,
                    integrand_rows=None,
                    component_choices=None,
                    run_id=None,
                    step_cm=None,
                    method=None,
                    resample=None,
                    step=None,
                    gemini_fallback=None,
                    save_debug=None,
                )

            try:
                meta = json.loads((run_path / "meta.json").read_text(encoding="utf-8"))
                px_per_cm = float(meta["px_per_cm"])
                step_cm = float(meta["step_cm"])
                method = str(meta["method"])
                resample = bool(meta["resample"])
                step = float(meta["step"])
                comp_id = int(component_id_in)
                mask_path = run_path / f"component_{comp_id}.png"
                if not mask_path.exists():
                    raise ValueError("Máscara de componente no encontrada")
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError("No se pudo leer máscara")
                mask01 = (mask > 0).astype(np.uint8)
                z_cm, r_cm = profile_from_mask(mask01, px_per_cm=px_per_cm, step_cm=step_cm)
                pr_z, pr_r = z_cm, r_cm
            except Exception as e:
                return render_template_string(
                    HTML,
                    result=None,
                    rows=None,
                    y_mode="radius",
                    message=f"Error usando selección: {e}",
                    is_error=True,
                    debug_zip_url=None,
                    integrand_rows=None,
                    component_choices=None,
                    run_id=None,
                    step_cm=None,
                    method=None,
                    resample=None,
                    step=None,
                    gemini_fallback=None,
                    save_debug=None,
                )

            df_new = pd.DataFrame({"z_cm": pr_z.astype(float), "r_cm": pr_r.astype(float)})
            df_new = df_new.sort_values("z_cm").reset_index(drop=True)

            if collected_path.exists():
                df_all = pd.read_csv(collected_path)
                df_all = pd.concat([df_all, df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(collected_path, index=False)

            rows_profile = [ProfileRow(z=float(z), r=float(r)) for z, r in zip(df_new["z_cm"], df_new["r_cm"])]
            rows_table = [TableRow(z=row.z, r=row.r, A=float(np.pi * (row.r**2))) for row in rows_profile]

            try:
                result = volume_from_profile(df_new["z_cm"].to_numpy(), df_new["r_cm"].to_numpy(), method=method, resample=resample, step=step)
            except Exception as e:
                return render_template_string(
                    HTML,
                    result=None,
                    rows=rows_table,
                    y_mode="radius",
                    message=str(e),
                    is_error=True,
                    debug_zip_url=url_for("download_debug", run_id=safe),
                    integrand_rows=None,
                    component_choices=None,
                    run_id=None,
                    step_cm=None,
                    method=None,
                    resample=None,
                    step=None,
                    gemini_fallback=None,
                    save_debug=None,
                )

            return render_template_string(
                HTML,
                result=result,
                rows=rows_table,
                y_mode="radius",
                message=f"Perfil extraído (selección) y guardado en {collected_path.name}.",
                is_error=False,
                debug_zip_url=url_for("download_debug", run_id=safe),
                integrand_rows=(list(zip(result.z_used, result.A_used))[:200] if result.z_used and result.A_used else None),
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
            )

        if "image" not in request.files:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message="Falta imagen.",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        f = request.files["image"]
        if not f.filename:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message="Imagen inválida.",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        try:
            step_cm = float(request.form.get("step_cm", "1.0"))
        except Exception:
            step_cm = 1.0

        method = request.form.get("method", "trapezoidal")
        resample = request.form.get("resample") == "yes"
        step = float(request.form.get("step", "1.0"))
        allow_gemini_fallback = request.form.get("gemini_fallback") == "yes"
        save_debug = request.form.get("save_debug") == "yes"

        image_bytes = f.read()
        if not image_bytes:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message="La imagen se recibió vacía. Vuelve a seleccionarla y reintenta.",
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )
        run_id = uuid.uuid4().hex[:12]
        run_path = debug_dir / run_id

        try:
            if save_debug:
                pr, dbg = profile_from_image_bytes_with_debug(image_bytes, step_cm=step_cm, allow_gemini_fallback=allow_gemini_fallback)
            else:
                pr = profile_from_image_bytes(image_bytes, step_cm=step_cm, allow_gemini_fallback=allow_gemini_fallback)
                dbg = None
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message=str(e),
                is_error=True,
                debug_zip_url=None,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        debug_zip_url = None
        if save_debug and dbg is not None:
            run_path.mkdir(parents=True, exist_ok=True)
            (run_path / "meta.json").write_text(
                json.dumps(
                    {
                        "step_cm": float(step_cm),
                        "method": str(method),
                        "resample": bool(resample),
                        "step": float(step),
                        "allow_gemini_fallback": bool(allow_gemini_fallback),
                        "px_per_cm": float(pr.px_per_cm),
                        "px_per_cm_source": str(dbg.px_per_cm_source),
                        "ruler_line": list(dbg.ruler_line) if dbg.ruler_line is not None else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_path / "input.jpg").write_bytes(image_bytes)
            cv2.imwrite(str(run_path / "mask.png"), (dbg.mask * 255).astype(np.uint8))
            cv2.imwrite(str(run_path / "overlay.png"), dbg.overlay_bgr)
            pd.DataFrame({"z_cm": pr.z_cm.astype(float), "r_cm": pr.r_cm.astype(float)}).to_csv(run_path / "profile.csv", index=False)
            debug_zip_url = url_for("download_debug", run_id=run_id)

            view_assets = {
                "input_url": url_for("debug_asset", run_id=run_id, name="input.jpg"),
                "overlay_url": url_for("debug_asset", run_id=run_id, name="overlay.png"),
                "mask_url": url_for("debug_asset", run_id=run_id, name="mask.png"),
                "px_per_cm": float(pr.px_per_cm),
                "px_per_cm_source": str(dbg.px_per_cm_source),
            }

            infos, masks = connected_components(dbg.mask, min_area_px=2500, max_components=6)
            if len(infos) >= 2:
                component_choices = []
                for info, cmask in zip(infos, masks):
                    overlay = dbg.overlay_bgr.copy()
                    contours, _ = cv2.findContours((cmask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
                    cv2.imwrite(str(run_path / f"preview_{info.component_id}.png"), overlay)
                    cv2.imwrite(str(run_path / f"component_{info.component_id}.png"), (cmask * 255).astype(np.uint8))
                    component_choices.append(
                        {
                            "component_id": info.component_id,
                            "area_px": info.area_px,
                            "preview_url": url_for("debug_asset", run_id=run_id, name=f"preview_{info.component_id}.png"),
                        }
                    )

                return render_template_string(
                    HTML,
                    result=None,
                    rows=None,
                    y_mode="radius",
                    message="Selecciona el objeto a integrar.",
                    is_error=False,
                    debug_zip_url=debug_zip_url,
                    integrand_rows=None,
                    component_choices=component_choices,
                    run_id=run_id,
                    step_cm=step_cm,
                    method=method,
                    resample=resample,
                    step=step,
                    gemini_fallback=allow_gemini_fallback,
                    save_debug=save_debug,
                    view_assets=view_assets,
                )

        df_new = pd.DataFrame({"z_cm": pr.z_cm.astype(float), "r_cm": pr.r_cm.astype(float)})
        df_new = df_new.sort_values("z_cm").reset_index(drop=True)

        if collected_path.exists():
            df_all = pd.read_csv(collected_path)
            df_all = pd.concat([df_all, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(collected_path, index=False)

        rows_profile = [ProfileRow(z=float(z), r=float(r)) for z, r in zip(df_new["z_cm"], df_new["r_cm"])]
        rows_table = [TableRow(z=row.z, r=row.r, A=float(np.pi * (row.r**2))) for row in rows_profile]

        try:
            result = volume_from_profile(df_new["z_cm"].to_numpy(), df_new["r_cm"].to_numpy(), method=method, resample=resample, step=step)
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=rows_table,
                y_mode="radius",
                message=str(e),
                is_error=True,
                debug_zip_url=debug_zip_url,
                integrand_rows=None,
                component_choices=None,
                run_id=None,
                step_cm=None,
                method=None,
                resample=None,
                step=None,
                gemini_fallback=None,
                save_debug=None,
                view_assets=None,
            )

        return render_template_string(
            HTML,
            result=result,
            rows=rows_table,
            y_mode="radius",
            message=f"Perfil extraído y guardado en {collected_path.name}.",
            is_error=False,
            debug_zip_url=debug_zip_url,
            integrand_rows=(list(zip(result.z_used, result.A_used))[:200] if result.z_used and result.A_used else None),
            component_choices=None,
            run_id=None,
            step_cm=None,
            method=None,
            resample=None,
            step=None,
            gemini_fallback=None,
            save_debug=None,
            view_assets=(view_assets if (save_debug and dbg is not None) else None),
        )

    @app.get("/examples/<name>")
    def download_example(name: str) -> Response:
        allowed = {"example_A.csv", "example_r.csv"}
        if name not in allowed:
            return Response("Not found", status=404)

        path = f"datasets/{name}"
        with open(path, "rb") as fh:
            data = fh.read()

        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={name}"},
        )

    @app.get("/collected.csv")
    def download_collected() -> Response:
        if not collected_path.exists():
            data = b"z_cm,r_cm\n"
        else:
            data = collected_path.read_bytes()
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=collected.csv"},
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8002, debug=True)
