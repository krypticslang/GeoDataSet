from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    z: np.ndarray
    y: np.ndarray  # r(z) or A(z)


def load_csv_dataset(path: str | Path, col_z: str, col_y: str) -> Dataset:
    df = pd.read_csv(path)
    if col_z not in df.columns or col_y not in df.columns:
        raise ValueError(f"Dataset must include columns: {col_z}, {col_y}")
    z = df[col_z].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)
    if z.size < 2:
        raise ValueError("Dataset needs at least 2 rows")
    order = np.argsort(z)
    z = z[order]
    y = y[order]
    return Dataset(z=z, y=y)


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
        raise ValueError("Simpson requires uniform spacing. Use --resample-step to resample.")

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--columns", nargs=2, metavar=("Z_COL", "Y_COL"), required=True)

    p.add_argument("--area-from-radius", action="store_true")

    p.add_argument("--method", choices=["trapezoidal", "simpson"], default="trapezoidal")

    p.add_argument("--resample-step", type=float, default=None)

    p.add_argument("--interpolation", choices=["none", "newton", "lagrange"], default="none")
    p.add_argument("--interp-n", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    col_z, col_y = args.columns

    ds = load_csv_dataset(args.dataset, col_z=col_z, col_y=col_y)
    z = ds.z
    y = ds.y

    if args.area_from_radius:
        r = y
        A = np.pi * (r**2)
    else:
        A = y

    if args.resample_step is not None:
        z, A = resample_uniform(z, A, step=float(args.resample_step))

    if args.interpolation != "none" and args.interp_n and args.interp_n > 1:
        zq = np.linspace(float(z[0]), float(z[-1]), int(args.interp_n), dtype=float)
        if args.interpolation == "lagrange":
            Aq = lagrange_interpolate(z, A, zq)
        else:
            coef = newton_divided_differences(z, A)
            Aq = newton_evaluate(coef, z, zq)
        z, A = zq, Aq

    if args.method == "trapezoidal":
        V = composite_trapezoidal(z, A)
    else:
        V = composite_simpson_uniform(z, A)

    print(f"Dataset: {args.dataset}")
    print(f"z range: {float(z[0]):.6f} .. {float(z[-1]):.6f} (n={z.size})")
    print(f"Method: {args.method}")
    if args.area_from_radius:
        print("Area: A(z)=pi*r(z)^2")
    else:
        print("Area: using dataset column as A(z)")
    if args.resample_step is not None:
        print(f"Resample: step={float(args.resample_step):.6f}")
    if args.interpolation != "none" and args.interp_n and args.interp_n > 1:
        print(f"Interpolation: {args.interpolation} to n={int(args.interp_n)}")

    print(f"V ≈ {V:.6f} (units: cm^3 if z in cm and A in cm^2)")


if __name__ == "__main__":
    main()
