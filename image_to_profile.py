from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gemini_api import ruler_endpoints_from_image_bytes


@dataclass(frozen=True)
class ProfileResult:
    z_cm: np.ndarray
    r_cm: np.ndarray
    px_per_cm: float


@dataclass(frozen=True)
class ProfileDebug:
    mask: np.ndarray
    overlay_bgr: np.ndarray
    ruler_line: tuple[int, int, int, int] | None
    px_per_cm_source: str


@dataclass(frozen=True)
class ComponentInfo:
    component_id: int
    area_px: int
    bbox: tuple[int, int, int, int]


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

    ang_thr = float(np.deg2rad(20.0))
    best_len = 0.0
    best = None
    best_any_len = 0.0
    best_any = None
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length > best_any_len:
            best_any_len = length
            best_any = (x1, y1, x2, y2)

        ang = float(np.arctan2(dy, dx))
        ang = float((ang + np.pi) % (2.0 * np.pi) - np.pi)
        horiz_dev = float(min(abs(ang), abs(np.pi - abs(ang))))
        vert_dev = float(abs(abs(ang) - (0.5 * np.pi)))
        if min(horiz_dev, vert_dev) > ang_thr:
            continue
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    if best is None:
        best = best_any
        best_len = best_any_len

    if best is None or best_len <= 0:
        raise ValueError("No se pudo estimar la longitud de la regla.")

    px_per_cm = best_len / 30.0
    if px_per_cm < 2.0:
        raise ValueError("Escala inválida: muy pocos pixeles por cm. Acerca más la cámara o usa mayor resolución.")

    if (float(w) / float(px_per_cm)) > 120.0 or (float(h) / float(px_per_cm)) > 120.0:
        raise ValueError(
            "Escala inválida: la regla detectada es demasiado corta (px/cm muy pequeño). "
            "Asegúrate de que la regla de 30 cm se vea completa o habilita Gemini como fallback."
        )

    return float(px_per_cm)


def _estimate_px_per_cm_from_ruler_debug(img_bgr: np.ndarray) -> tuple[float, tuple[int, int, int, int]]:
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(0.35 * max(h, w)), maxLineGap=15)
    if lines is None:
        raise ValueError("No se pudo detectar la regla (líneas). Asegúrate que la regla de 30cm se vea completa y nítida.")

    ang_thr = float(np.deg2rad(20.0))
    best_len = 0.0
    best: tuple[int, int, int, int] | None = None
    best_any_len = 0.0
    best_any: tuple[int, int, int, int] | None = None
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length > best_any_len:
            best_any_len = length
            best_any = (int(x1), int(y1), int(x2), int(y2))

        ang = float(np.arctan2(dy, dx))
        ang = float((ang + np.pi) % (2.0 * np.pi) - np.pi)
        horiz_dev = float(min(abs(ang), abs(np.pi - abs(ang))))
        vert_dev = float(abs(abs(ang) - (0.5 * np.pi)))
        if min(horiz_dev, vert_dev) > ang_thr:
            continue
        if length > best_len:
            best_len = length
            best = (int(x1), int(y1), int(x2), int(y2))

    if best is None:
        best = best_any
        best_len = best_any_len

    if best is None or best_len <= 0:
        raise ValueError("No se pudo estimar la longitud de la regla.")

    px_per_cm = best_len / 30.0
    if px_per_cm < 2.0:
        raise ValueError("Escala inválida: muy pocos pixeles por cm. Acerca más la cámara o usa mayor resolución.")

    if (float(w) / float(px_per_cm)) > 120.0 or (float(h) / float(px_per_cm)) > 120.0:
        raise ValueError(
            "Escala inválida: la regla detectada es demasiado corta (px/cm muy pequeño). "
            "Asegúrate de que la regla de 30 cm se vea completa o habilita Gemini como fallback."
        )

    return float(px_per_cm), best


def _estimate_px_per_cm_from_gemini(image_bytes: bytes) -> float:
    resp = ruler_endpoints_from_image_bytes(image_bytes)
    if not resp.ok or not resp.data:
        raise ValueError(resp.message)

    try:
        p1 = resp.data["ruler"]["p1"]
        p2 = resp.data["ruler"]["p2"]
        x1, y1 = float(p1["x"]), float(p1["y"])
        x2, y2 = float(p2["x"]), float(p2["y"])
    except Exception:
        raise ValueError("Gemini devolvió JSON inválido para puntos de la regla")

    length_px = float(np.hypot(x2 - x1, y2 - y1))
    if length_px <= 0:
        raise ValueError("Gemini devolvió una longitud de regla inválida")
    px_per_cm = length_px / 30.0
    if px_per_cm < 2.0:
        raise ValueError("Escala inválida (Gemini): muy pocos pixeles por cm")
    return float(px_per_cm)


def _segment_object_mask(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    blur = cv2.GaussianBlur(l2, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th_inv = 255 - th

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s_blur = cv2.GaussianBlur(s, (7, 7), 0)
    _, ths = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ths_inv = 255 - ths

    h, w = th.shape[:2]
    total = float(th.size)

    def border_contact_frac_component(lbl: np.ndarray, cid: int) -> float:
        top = int(np.sum(lbl[0, :] == cid))
        bot = int(np.sum(lbl[h - 1, :] == cid))
        left = int(np.sum(lbl[:, 0] == cid))
        right = int(np.sum(lbl[:, w - 1] == cid))
        return float(top + bot + left + right) / float(2 * w + 2 * h)

    def best_component(binary: np.ndarray) -> tuple[np.ndarray, float] | tuple[None, float]:
        b = (binary > 0).astype(np.uint8)
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if num <= 1:
            return None, 0.0

        best_score = -1.0
        best_mask = None
        for cid in range(1, num):
            area = float(stats[cid, cv2.CC_STAT_AREA])
            if area < 1200.0:
                continue
            x = float(stats[cid, cv2.CC_STAT_LEFT])
            y = float(stats[cid, cv2.CC_STAT_TOP])
            bw = float(stats[cid, cv2.CC_STAT_WIDTH])
            bh = float(stats[cid, cv2.CC_STAT_HEIGHT])
            width_frac = bw / float(w)
            height_frac = bh / float(h)
            area_frac = area / total
            bc = border_contact_frac_component(lbl, cid)

            if width_frac >= 0.97 or height_frac >= 0.97:
                continue
            if area_frac >= 0.70:
                continue

            score = area * (1.0 - min(0.95, bc))
            score *= (1.0 - max(0.0, width_frac - 0.85))
            if score > best_score:
                best_score = score
                best_mask = (lbl == cid).astype(np.uint8)

        if best_mask is None:
            return None, 0.0
        return best_mask, float(best_score)

    candidates = [th, th_inv, ths, ths_inv]
    best = None
    best_score = -1.0
    for c in candidates:
        cm, sc = best_component(c)
        if cm is None:
            continue
        if sc > best_score:
            best_score = sc
            best = cm

    if best is None:
        best = _largest_connected_component(th)

    mask = best

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return (mask > 0).astype(np.uint8)


def connected_components(mask01: np.ndarray, *, min_area_px: int = 1500, max_components: int = 8) -> tuple[list[ComponentInfo], list[np.ndarray]]:
    m = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    infos: list[ComponentInfo] = []
    masks: list[np.ndarray] = []
    if num_labels <= 1:
        return infos, masks

    items: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for cid in range(1, num_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        items.append((cid, area, (x, y, w, h)))

    items.sort(key=lambda t: t[1], reverse=True)
    items = items[: int(max_components)]

    for idx, (cid, area, bbox) in enumerate(items):
        comp_mask = (labels == cid).astype(np.uint8)
        infos.append(ComponentInfo(component_id=idx, area_px=area, bbox=bbox))
        masks.append(comp_mask)

    return infos, masks


def profile_from_mask(mask01: np.ndarray, *, px_per_cm: float, step_cm: float) -> tuple[np.ndarray, np.ndarray]:
    if step_cm <= 0:
        raise ValueError("step_cm debe ser > 0")
    if px_per_cm <= 0:
        raise ValueError("px_per_cm inválido")

    mask = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise ValueError("No se pudo segmentar el objeto (sin componentes).")

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = int(np.argmax(areas)) + 1

    bw = float(stats[main_label, cv2.CC_STAT_WIDTH])
    bh = float(stats[main_label, cv2.CC_STAT_HEIGHT])
    if bh > 0 and (bw / bh) >= 3.0:
        raise ValueError(
            "La silueta detectada es muy horizontal (ancho >> alto). "
            "Este método asume un sólido de revolución con el eje vertical (z). "
            "Toma la foto en perfil con el objeto orientado verticalmente (o rota la imagen) y reintenta."
        )

    ys, xs = np.where(labels == main_label)
    if ys.size < 1000:
        raise ValueError("No se pudo segmentar el objeto (muy pocos pixeles).")

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    z_max_cm = float((y_max - y_min) / float(px_per_cm))
    if z_max_cm <= 0:
        raise ValueError("Altura inválida detectada")

    z_cm = np.arange(0.0, z_max_cm + 0.5 * step_cm, float(step_cm), dtype=float)
    r_cm = np.zeros_like(z_cm, dtype=float)
    for i, z in enumerate(z_cm):
        y = int(round(y_max - z * float(px_per_cm)))
        y = int(np.clip(y, 0, mask.shape[0] - 1))

        row = labels[y, :]
        idx = np.where(row == main_label)[0]
        if idx.size < 2:
            r_cm[i] = np.nan
            continue
        width_px = float(idx[-1] - idx[0])
        r_cm[i] = 0.5 * (width_px / float(px_per_cm))

    good = np.isfinite(r_cm) & (r_cm > 0)
    if int(np.sum(good)) < 3:
        raise ValueError("No se pudo calcular r(z) suficiente.")

    return z_cm[good], r_cm[good]


def profile_from_image_bytes(
    image_bytes: bytes,
    *,
    step_cm: float = 1.0,
    allow_gemini_fallback: bool = False,
) -> ProfileResult:
    pr, _dbg = profile_from_image_bytes_with_debug(image_bytes, step_cm=step_cm, allow_gemini_fallback=allow_gemini_fallback)
    return pr


def profile_from_image_bytes_with_debug(
    image_bytes: bytes,
    *,
    step_cm: float = 1.0,
    allow_gemini_fallback: bool = False,
) -> tuple[ProfileResult, ProfileDebug]:
    if step_cm <= 0:
        raise ValueError("step_cm debe ser > 0")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen")

    ruler_line: tuple[int, int, int, int] | None = None
    px_per_cm_source = "local"
    try:
        px_per_cm, ruler_line = _estimate_px_per_cm_from_ruler_debug(img)
    except Exception as e:
        if not allow_gemini_fallback:
            raise e
        px_per_cm_source = "gemini"
        resp = ruler_endpoints_from_image_bytes(image_bytes)
        if not resp.ok or not resp.data:
            raise ValueError(resp.message)
        try:
            p1 = resp.data["ruler"]["p1"]
            p2 = resp.data["ruler"]["p2"]
            x1, y1 = int(p1["x"]), int(p1["y"])
            x2, y2 = int(p2["x"]), int(p2["y"])
            ruler_line = (x1, y1, x2, y2)
        except Exception:
            raise ValueError("Gemini devolvió JSON inválido para puntos de la regla")
        length_px = float(np.hypot(float(ruler_line[2] - ruler_line[0]), float(ruler_line[3] - ruler_line[1])))
        if length_px <= 0:
            raise ValueError("Gemini devolvió una longitud de regla inválida")
        px_per_cm = float(length_px / 30.0)
        if px_per_cm < 2.0:
            raise ValueError("Escala inválida (Gemini): muy pocos pixeles por cm")

    mask = _segment_object_mask(img)
    if ruler_line is not None:
        h, w = mask.shape[:2]
        x1, y1, x2, y2 = ruler_line

        y_r = int(round(0.5 * (float(y1) + float(y2))))
        y_r = int(np.clip(y_r, 0, h - 1))

        thickness = int(max(12, round(0.120 * float(h))))
        rm = np.zeros((h, w), dtype=np.uint8)
        cv2.line(rm, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness)

        mask = (mask > 0).astype(np.uint8)
        mask[rm > 0] = 0

        margin = int(max(12, round(0.5 * float(thickness))))
        y_top = int(max(0, y_r - margin))
        y_bot = int(min(h, y_r + margin))

        top_mask = mask.copy()
        top_mask[y_top:, :] = 0
        bot_mask = mask.copy()
        bot_mask[:y_bot, :] = 0

        def _best_component(m: np.ndarray) -> tuple[np.ndarray, float] | tuple[None, float]:
            m = (m > 0).astype(np.uint8)
            num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num <= 1:
                return None, 0.0
            best_score = -1.0
            best_comp = None
            for cid in range(1, num):
                area = float(stats[cid, cv2.CC_STAT_AREA])
                x = float(stats[cid, cv2.CC_STAT_LEFT])
                bw = float(stats[cid, cv2.CC_STAT_WIDTH])
                if area < 1500.0:
                    continue
                width_frac = bw / float(w)
                if width_frac >= 0.90:
                    continue
                score = area
                if score > best_score:
                    best_score = score
                    best_comp = (lbl == cid).astype(np.uint8)
            if best_comp is None:
                return None, 0.0
            return best_comp, float(best_score)

        top_comp, top_score = _best_component(top_mask)
        bot_comp, bot_score = _best_component(bot_mask)

        if top_score >= bot_score and top_comp is not None:
            mask = top_comp
        elif bot_comp is not None:
            mask = bot_comp
        else:
            if int(np.sum(top_mask)) >= int(np.sum(bot_mask)):
                mask = top_mask
            else:
                mask = bot_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = _largest_connected_component(mask)
    z_cm, r_cm = profile_from_mask(mask, px_per_cm=float(px_per_cm), step_cm=float(step_cm))

    overlay = img.copy()
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    if ruler_line is not None:
        x1, y1, x2, y2 = ruler_line
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    pr = ProfileResult(z_cm=z_cm, r_cm=r_cm, px_per_cm=float(px_per_cm))
    dbg = ProfileDebug(mask=mask.astype(np.uint8), overlay_bgr=overlay, ruler_line=ruler_line, px_per_cm_source=px_per_cm_source)
    return pr, dbg
