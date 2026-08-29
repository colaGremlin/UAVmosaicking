"""Operator overlays, drawn on a copy of the canvas and never on the canvas itself.

Drawing into the mosaic would poison it: the next composite would treat overlay pixels as
imagery and they would persist for the rest of the sortie. Everything here works on the
encoder's copy.

Three levels, because an operator and an engineer want different screens
------------------------------------------------------------------------
Real ground-control software does not put diagnostics on the operator's map. The map shows
the picture and the few marks needed to act on it; anything an engineer needs to debug the
sensor chain lives somewhere the operator can call up, not somewhere it competes with the
imagery. So:

``off``      imagery only -- nothing drawn. Cleanest possible feed for recording.
``minimal``  the default. Where each aircraft is looking, which way is north, how far is a
             metre. Nothing else.
``full``     adds the engineering layer: viewing angle, which source set the ground plane,
             per-aircraft share of the mosaic, and loop timing. For bring-up and fault-finding.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .canvas import Canvas
from .coords import CanvasGeometry

__all__ = ["UAV_COLOURS", "HUD_LEVELS", "draw_hud"]

HUD_LEVELS = ("off", "minimal", "full")

#: One BGR colour per UAV, reused for its footprint outline and its stats row.
UAV_COLOURS = {
    0: (80, 220, 255),   # amber
    1: (120, 255, 120),  # green
    2: (255, 170, 90),   # blue
    3: (200, 130, 255),  # violet
}
_STALE = (120, 120, 120)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _colour(uav_id: int) -> tuple[int, int, int]:
    return UAV_COLOURS.get(uav_id, (200, 200, 200))


def _label(frame, text, org, colour, scale=0.45):
    """Text with a dark outline, so it stays readable over any imagery underneath."""
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, colour, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    canvas: Canvas,
    geom: CanvasGeometry,
    footprints: dict,
    stale: dict[int, bool] | None = None,
    stats_lines: tuple[str, ...] = (),
    scale: float = 1.0,
    offset: tuple[int, int] = (0, 0),
    level: str = "minimal",
    map_rect: tuple[int, int, int, int] | None = None,
    view_origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Draw the overlay onto ``frame`` in place and return it.

    ``scale`` and ``offset`` map canvas pixels to frame pixels, so the overlay lands
    correctly on a downscaled and letterboxed output frame.

    ``map_rect`` is the ``(x, y, w, h)`` region of ``frame`` that actually holds map. When a
    square area of interest is letterboxed into a wider frame, footprints are drawn into that
    sub-view so OpenCV clips them for us. Two reasons that matters: the letterbox bars are not
    map, so an outline out there is a lie -- no imagery can ever exist beyond the area of
    interest -- and a footprint that pokes past the frame would otherwise be drawn over bars
    the resize never repaints, accumulating into solid colour over a few hundred frames.
    """
    if level == "off":
        return frame

    full = level == "full"
    stale = stale or {}
    h, w = frame.shape[:2]

    if map_rect is not None:
        mx, my, mw, mh = map_rect
        canvas_area = frame[my : my + mh, mx : mx + mw]
        # Inside the sub-view, canvas pixel (0,0) sits at -view_origin: the view may be panned
        # to follow the imagery rather than showing the whole area of interest.
        off = -np.asarray(view_origin, dtype=np.float64)
        h, w = mh, mw
    else:
        canvas_area = frame
        off = np.asarray(offset, dtype=np.float64)

    for uav_id, fp in sorted(footprints.items()):
        is_stale = stale.get(uav_id, False)
        col = _STALE if is_stale else _colour(uav_id)
        pts = (fp.corners_px * scale + off).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas_area, [pts], True, col, 1 if not full else 2, cv2.LINE_AA)

        cx, cy = (fp.corners_px.mean(axis=0) * scale + off).astype(int)
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        if full:
            # engineering layer: which way the frame was oriented, plus sensor detail
            tl = fp.corners_px[0] * scale + off
            tr = fp.corners_px[1] * scale + off
            cv2.line(canvas_area, tuple(tl.astype(int)), tuple(tr.astype(int)), col, 3, cv2.LINE_AA)
            cv2.circle(canvas_area, (cx, cy), 5, col, -1, cv2.LINE_AA)
            _label(canvas_area, f"UAV{uav_id}" + (" STALE" if is_stale else ""), (cx + 9, cy - 7), col, 0.5)
            deg = np.degrees(np.arccos(np.clip(fp.worst_cos_theta, -1, 1)))
            _label(canvas_area, f"{fp.plane}  {deg:.0f}deg", (cx + 9, cy + 9), col, 0.4)
        else:
            # operator layer: only which aircraft is looking where
            cv2.drawMarker(canvas_area, (cx, cy), col, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
            _label(canvas_area, str(uav_id), (cx + 9, cy + 4), col, 0.44)

    _scale_bar(frame, geom, scale)
    _north(frame)

    if full:
        _stats(frame, canvas, footprints, stale, stats_lines)
    return frame


def _scale_bar(frame: np.ndarray, geom: CanvasGeometry, scale: float) -> None:
    """A bar whose length is a round number of metres at the current output scale."""
    h, w = frame.shape[:2]
    m_per_px = geom.gsd / max(scale, 1e-9)
    metres, px = 100, 0
    for m in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        metres, px = m, m / m_per_px
        if px >= w * 0.12:
            break
    px = int(px)
    x0, y0 = 18, h - 24
    cv2.line(frame, (x0, y0), (x0 + px, y0), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(frame, (x0, y0), (x0 + px, y0), (255, 255, 255), 2, cv2.LINE_AA)
    for x in (x0, x0 + px):
        cv2.line(frame, (x, y0 - 5), (x, y0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
    _label(frame, f"{metres} m", (x0, y0 - 9), (255, 255, 255))


def _north(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    x, y = w - 32, 42
    for col, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.arrowedLine(frame, (x, y + 20), (x, y - 14), col, thick, cv2.LINE_AA, tipLength=0.4)
    _label(frame, "N", (x - 6, y - 18), (255, 255, 255))


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
        col = (255, 255, 255)
        if s.startswith("UAV"):
            uid = int(s[3])
            col = _STALE if stale.get(uid) else _colour(uid)
        _label(frame, s, (6 + pad, 6 + pad + lh * i + 4), col)
