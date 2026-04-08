from __future__ import annotations

import io
from pathlib import Path
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template_string, request, url_for

from image_to_profile import profile_from_image_bytes


@dataclass(frozen=True)
class ComputeResult:
    volume: float
    z_min: float
    z_max: float
    n: int
    method: str


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

    return ComputeResult(volume=float(V), z_min=float(z[0]), z_max=float(z[-1]), n=int(z.size), method=method)


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

    return ComputeResult(volume=float(V), z_min=float(z[0]), z_max=float(z[-1]), n=int(z.size), method=method)


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
  <title>Volumen por Dataset (Análisis Numérico)</title>
  <style>
    *{box-sizing:border-box;}
    :root{
      --bg:#070a0f; --panel:#0d141f; --fg:#d7ffe0; --muted:#7fe0a2; --accent:#00ff66; --border:rgba(0,255,102,.22);
    }
    body{margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: radial-gradient(1200px 600px at 20% 10%, rgba(0,255,102,.10), transparent 55%),
                  radial-gradient(900px 500px at 80% 20%, rgba(0,200,83,.10), transparent 60%),
                  var(--bg);
      color:var(--fg);
    }
    .wrap{max-width:1100px; margin:0 auto; padding:28px 18px 60px;}
    h1{font-size:22px; margin:0 0 6px;}
    .sub{color:var(--muted); margin:0 0 18px; font-size:13px;}
    .grid{display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap:14px; align-items:start;}
    .card{background:rgba(255,255,255,.02); border:1px solid var(--border); border-radius:14px; padding:14px; overflow:hidden;}
    .section-title{margin:0 0 10px; font-size:13px; color:var(--muted); letter-spacing:0.2px;}
    label{display:block; font-size:12px; color:var(--muted); margin:10px 0 6px;}
    input, select{width:100%; padding:10px; border-radius:10px; border:1px solid rgba(0,255,102,.18);
      background:rgba(0,0,0,.25); color:var(--fg);
    }
    .row{display:grid; grid-template-columns: 1fr 1fr; gap:10px;}
    .btn{margin-top:12px; width:100%; padding:12px; border-radius:12px; border:1px solid rgba(0,255,102,.35);
      background:rgba(0,255,102,.10); color:var(--fg); cursor:pointer;
    }
    .btn:hover{background:rgba(0,255,102,.16); border-color:rgba(0,255,102,.55);}
    .kpi{display:grid; grid-template-columns: 1fr 1fr; gap:10px;}
    .k{background:rgba(0,255,102,.06); border:1px solid rgba(0,255,102,.20); border-radius:14px; padding:10px 12px;}
    .k .l{color:var(--muted); font-size:12px;}
    .k .v{font-size:18px; font-weight:700; margin-top:4px;}
    .err{border:1px solid rgba(255,90,90,.35); background:rgba(255,90,90,.08); padding:10px 12px; border-radius:12px;}
    .ok{border:1px solid rgba(0,255,102,.25); background:rgba(0,255,102,.06); padding:10px 12px; border-radius:12px;}
    .hint{color:var(--muted); font-size:12px; margin-top:10px;}
    .small{color:var(--muted); font-size:12px;}
    a{color:var(--accent);}

    .table-wrap{width:100%; overflow:auto; margin-top:10px; border-radius:14px; border:1px solid rgba(0,255,102,.12);}
    table{width:100%; min-width:720px; border-collapse:collapse;}
    thead th{font-size:12px; color:var(--muted); font-weight:700; text-align:left; padding:12px 10px;
      border-bottom:1px solid rgba(0,255,102,.18);
      background:rgba(255,255,255,.01);
    }
    tbody td{padding:12px 10px; border-bottom:1px solid rgba(0,255,102,.10);}
    tbody tr:hover{background:rgba(0,255,102,.05);}
    .mono{font-variant-numeric: tabular-nums;}
    .pill{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid rgba(0,255,102,.25);
      background:rgba(0,255,102,.06); color:var(--fg); font-size:12px;}

    .switch-row{display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px;}
    .switch-row .txt{display:flex; flex-direction:column; gap:2px;}
    .switch-row .txt .t{color:var(--muted); font-size:12px;}
    .switch-row .txt .d{color:rgba(215,255,224,.85); font-size:12px;}

    .switch{position:relative; display:inline-block; width:44px; height:26px; flex:0 0 auto;}
    .switch input{opacity:0; width:0; height:0;}
    .slider{position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0;
      background:rgba(255,255,255,.08); border:1px solid rgba(0,255,102,.25); transition:.2s; border-radius:999px;
    }
    .slider:before{position:absolute; content:""; height:20px; width:20px; left:3px; top:2px;
      background:rgba(215,255,224,.9); transition:.2s; border-radius:999px;
    }
    .switch input:checked + .slider{background:rgba(0,255,102,.18);}
    .switch input:checked + .slider:before{transform:translateX(18px); background:rgba(0,255,102,.95);}

    @media (max-width: 920px){
      .grid{grid-template-columns: 1fr;}
      table{min-width: 640px;}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Volumen por Dataset (Análisis Numérico)</h1>
    <p class="sub">Sube una imagen (objeto + regla de 30 cm) para generar `z_cm,r_cm`, guardar en `collected.csv` y calcular el volumen.</p>

    {% if message %}
      <div class="{{ 'err' if is_error else 'ok' }}">{{ message }}</div>
    {% endif %}

    <div class="grid">
      <div class="card">
        <form method="post" action="{{ url_for('compute_image') }}" enctype="multipart/form-data">
          <div class="section-title">Entrada</div>
          <label>Imagen (objeto + regla de 30 cm)</label>
          <input type="file" name="image" accept="image/*" required />

          <label>Paso de muestreo (cm)</label>
          <input name="step_cm" type="number" step="0.1" value="1.0" />

          <div class="section-title" style="margin-top:14px;">Ajustes numéricos</div>
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

          <button class="btn" type="submit">Extraer perfil → Guardar → Calcular</button>
          <div class="hint">Asegúrate que la regla de 30 cm se vea completa y esté en el mismo plano que el objeto.</div>
        </form>

        <div style="margin-top:14px; border-top:1px solid rgba(0,255,102,.12); padding-top:14px;">
          <div class="section-title">Alternativa</div>
          <div class="small">Si ya tienes un CSV, puedes usar el modo CSV.</div>
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

            <button class="btn" type="submit">Calcular con CSV</button>
          </form>
        </div>
      </div>

      <div class="card">
        <h2 style="font-size:14px; margin:0 0 10px; color:var(--muted);">Resultado</h2>
        {% if result %}
          <div class="kpi">
            <div class="k"><div class="l">Volumen estimado</div><div class="v">{{ '%.6f'|format(result.volume) }}</div></div>
            <div class="k"><div class="l">Método</div><div class="v">{{ result.method }}</div></div>
            <div class="k"><div class="l">z min</div><div class="v">{{ '%.6f'|format(result.z_min) }}</div></div>
            <div class="k"><div class="l">z max</div><div class="v">{{ '%.6f'|format(result.z_max) }}</div></div>
            <div class="k"><div class="l">n puntos</div><div class="v">{{ result.n }}</div></div>
          </div>
          <p class="small" style="margin-top:10px;">Unidades: cm³ si `z` está en cm y `A` en cm².</p>
        {% else %}
          <p class="small">Sube un CSV y calcula para ver resultados aquí.</p>
        {% endif %}
      </div>
    </div>

    <div class="card" style="margin-top:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
        <div>
          <h2 style="font-size:14px; margin:0; color:var(--muted);">Tabla de datos</h2>
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
        <div class="small" style="margin-top:10px;">Aún no hay datos. Calcula con un CSV para ver la tabla aquí.</div>
      {% endif %}
    </div>

    <p class="small" style="margin-top:16px;">
      Archivos de ejemplo:
      <a href="{{ url_for('download_example', name='example_A.csv') }}">example_A.csv</a>
      · <a href="{{ url_for('download_example', name='example_r.csv') }}">example_r.csv</a>
      · <a href="{{ url_for('download_collected') }}">collected.csv</a>
    </p>
  </div>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = uuid.uuid4().hex

    datasets_dir = Path(__file__).resolve().parent / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    collected_path = datasets_dir / "collected.csv"

    @app.get("/")
    def index() -> str:
        return render_template_string(
            HTML,
            result=None,
            rows=None,
            y_mode=None,
            message=None,
            is_error=False,
        )

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
            )

        return render_template_string(
            HTML,
            result=result,
            rows=rows,
            y_mode=y_mode,
            message="Cálculo completado.",
            is_error=False,
        )

    @app.post("/compute/image")
    def compute_image() -> str:
        if "image" not in request.files:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message="Falta imagen.",
                is_error=True,
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
            )

        try:
            step_cm = float(request.form.get("step_cm", "1.0"))
        except Exception:
            step_cm = 1.0

        method = request.form.get("method", "trapezoidal")
        resample = request.form.get("resample") == "yes"
        step = float(request.form.get("step", "1.0"))
        allow_gemini_fallback = request.form.get("gemini_fallback") == "yes"

        try:
            pr = profile_from_image_bytes(f.read(), step_cm=step_cm, allow_gemini_fallback=allow_gemini_fallback)
        except Exception as e:
            return render_template_string(
                HTML,
                result=None,
                rows=None,
                y_mode="radius",
                message=str(e),
                is_error=True,
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
            )

        return render_template_string(
            HTML,
            result=result,
            rows=rows_table,
            y_mode="radius",
            message=f"Perfil extraído y guardado en {collected_path.name}.",
            is_error=False,
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
    app.run(host="127.0.0.1", port=8000, debug=True)
