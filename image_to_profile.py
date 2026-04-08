from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProfileResult:
    z_cm: np.ndarray
    r_cm: np.ndarray
    px_per_cm: float


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    return (labels == i).astype(np.uint8)


def _estimate_px_per_cm_from_ruler(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(0.35 * max(h, w)), maxLineGap=15)
    if lines is None:
        raise ValueError("No se pudo detectar la regla (líneas). Asegúrate que la regla de 30cm se vea completa y nítida.")

    best_len = 0.0
    best = None
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    if best is None or best_len <= 0:
        raise ValueError("No se pudo estimar la longitud de la regla.")

    px_per_cm = best_len / 30.0
    if px_per_cm < 2.0:
        raise ValueError("Escala inválida: muy pocos pixeles por cm. Acerca más la cámara o usa mayor resolución.")

    return float(px_per_cm)


def _segment_object_mask(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    blur = cv2.GaussianBlur(l2, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th_inv = 255 - th
    c1 = _largest_connected_component(th)
    c2 = _largest_connected_component(th_inv)

    if int(np.sum(c1)) > int(np.sum(c2)):
        mask = c1
    else:
        mask = c2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return (mask > 0).astype(np.uint8)


def profile_from_image_bytes(
    image_bytes: bytes,
    *,
    step_cm: float = 1.0,
) -> ProfileResult:
    if step_cm <= 0:
        raise ValueError("step_cm debe ser > 0")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen")

    px_per_cm = _estimate_px_per_cm_from_ruler(img)
    mask = _segment_object_mask(img)

    ys, xs = np.where(mask > 0)
    if ys.size < 1000:
        raise ValueError("No se pudo segmentar el objeto (muy pocos pixeles). Usa fondo más uniforme y buena iluminación.")

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    z_max_cm = float((y_max - y_min) / px_per_cm)
    if z_max_cm <= 0:
        raise ValueError("Altura inválida detectada")

    z_cm = np.arange(0.0, z_max_cm + 0.5 * step_cm, float(step_cm), dtype=float)

    r_cm = np.zeros_like(z_cm, dtype=float)
    for i, z in enumerate(z_cm):
        y = int(round(y_max - z * px_per_cm))
        y = int(np.clip(y, 0, mask.shape[0] - 1))

        row = mask[y, :]
        idx = np.where(row > 0)[0]
        if idx.size < 2:
            r_cm[i] = np.nan
            continue
        width_px = float(idx[-1] - idx[0])
        r_cm[i] = 0.5 * (width_px / px_per_cm)

    good = np.isfinite(r_cm) & (r_cm > 0)
    if int(np.sum(good)) < 3:
        raise ValueError("No se pudo calcular r(z) suficiente. Intenta una foto más perpendicular y con fondo contrastante.")

    z_cm = z_cm[good]
    r_cm = r_cm[good]

    return ProfileResult(z_cm=z_cm, r_cm=r_cm, px_per_cm=float(px_per_cm))
