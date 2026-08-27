"""Operator overlays drawn on a copy of the canvas, never on the canvas itself.

Drawing into the mosaic would poison it: the next composite would treat overlay pixels as
imagery and they would persist forever. Everything here works on the encoder's copy.

The overlay answers the questions an operator actually has: which UAV is contributing what,
is anyone stale, how did each ground plane get its height, and how far is a metre.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .canvas import Canvas
from .coords import CanvasGeometry

__all__ = ["UAV_COLOURS", "draw_hud"]

#: One BGR colour per UAV, reused for its footprint outline and its stats row.
UAV_COLOURS = {
    0: (80, 220, 255),   # amber
    1: (120, 255, 120),  # green
    2: (255, 170, 90),   # blue
    3: (200, 130, 255),  # violet
}
_STALE = (110, 110, 110)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _colour(uav_id: int) -> tuple[int, int, int]:
    return UAV_COLOURS.get(uav_id, (200, 200, 200))


def draw_hud(
    frame: np.ndarray,
    canvas: Canvas,
    geom: CanvasGeometry,
    footprints: dict,
    stale: dict[int, bool] | None = None,
    stats_lines: tuple[str, ...] = (),
    scale: float = 1.0,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw footprints, markers, a scale bar and a stats block onto ``frame`` in place.

    ``scale`` and ``offset`` map canvas pixels to frame pixels, so the overlay lands
    correctly on a downscaled and letterboxed output frame.
    """
    stale = stale or {}
    h, w = frame.shape[:2]
    off = np.asarray(offset, dtype=np.float64)

    for uav_id, fp in sorted(footprints.items()):
        is_stale = stale.get(uav_id, False)
        col = _STALE if is_stale else _colour(uav_id)
        pts = (fp.corners_px * scale + off).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, col, 2, cv2.LINE_AA)

        # a tick on the top edge shows which way is "up" in the source image
        tl, tr = (fp.corners_px[0] * scale + off), (fp.corners_px[1] * scale + off)
        mid = ((tl + tr) / 2).astype(int)
        cv2.line(frame, tuple(tl.astype(int)), tuple(tr.astype(int)), col, 3, cv2.LINE_AA)

        cx, cy = (fp.corners_px.mean(axis=0) * scale + off).astype(int)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), 5, col, -1, cv2.LINE_AA)
            label = f"UAV{uav_id}" + (" STALE" if is_stale else "")
            cv2.putText(frame, label, (cx + 9, cy - 7), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (cx + 9, cy - 7), _FONT, 0.5, col, 1, cv2.LINE_AA)
            sub = f"{fp.plane}  {np.degrees(np.arccos(np.clip(fp.worst_cos_theta, -1, 1))):.0f}deg"
            cv2.putText(frame, sub, (cx + 9, cy + 9), _FONT, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, sub, (cx + 9, cy + 9), _FONT, 0.4, col, 1, cv2.LINE_AA)
        if not is_stale:
            cv2.drawMarker(frame, (cx, cy), col, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    _scale_bar(frame, geom, scale)
    _north(frame)
    _stats(frame, canvas, footprints, stale, stats_lines)
    return frame


def _scale_bar(frame: np.ndarray, geom: CanvasGeometry, scale: float) -> None:
    """A bar whose length is a round number of metres at the current output scale."""
    h, w = frame.shape[:2]
    m_per_px = geom.gsd / max(scale, 1e-9)
    for metres in (50, 100, 200, 500, 1000, 2000, 5000):
        px = metres / m_per_px
        if px >= w * 0.12:
            break
    px = int(px)
    x0, y0 = 18, h - 26
    cv2.line(frame, (x0, y0), (x0 + px, y0), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(frame, (x0, y0), (x0 + px, y0), (255, 255, 255), 2, cv2.LINE_AA)
    for x in (x0, x0 + px):
        cv2.line(frame, (x, y0 - 5), (x, y0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
    txt = f"{metres} m"
    cv2.putText(frame, txt, (x0, y0 - 10), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (x0, y0 - 10), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _north(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    x, y = w - 34, 46
    cv2.arrowedLine(frame, (x, y + 22), (x, y - 16), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(frame, (x, y + 22), (x, y - 16), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(frame, "N", (x - 7, y - 20), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "N", (x - 7, y - 20), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _stats(frame, canvas, footprints, stale, extra) -> None:
    counts = canvas.owner_counts()
    total = max(sum(counts.values()), 1)
    lines = [f"coverage {canvas.coverage_fraction() * 100:5.1f}%"]
    for uav_id in sorted(set(list(counts) + list(footprints))):
        share = counts.get(uav_id, 0) / total * 100
        flag = "STALE" if stale.get(uav_id) else "live "
        lines.append(f"UAV{uav_id} {flag} {share:5.1f}% of mosaic")
    lines.extend(extra)
    lines.append(time.strftime("%H:%M:%S"))

    pad, lh = 8, 17
    box_w = int(max(cv2.getTextSize(s, _FONT, 0.45, 1)[0][0] for s in lines) + pad * 2)
    box_h = lh * len(lines) + pad
    panel = frame[6 : 6 + box_h, 6 : 6 + box_w]
    if panel.size:
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.0, 0, dst=panel)

    for i, s in enumerate(lines):
        y = 6 + pad + lh * i + 4
        col = (255, 255, 255)
        if s.startswith("UAV"):
            uid = int(s[3])
            col = _STALE if stale.get(uid) else _colour(uid)
        cv2.putText(frame, s, (6 + pad, y), _FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (6 + pad, y), _FONT, 0.45, col, 1, cv2.LINE_AA)
