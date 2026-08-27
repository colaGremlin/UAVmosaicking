"""Per-pixel compositing weights -- the adaptive part Map2DFusion describes but ships only
partially.

The Map2DFusion paper states the weight is computed "considering the height, view angle and
pixel localization", but ``MultiBandMap2DCPU.cpp`` implements only the radial (pixel
localization) term. The other two are written here::

    w(u, v) = w_radial(u, v) * w_incidence(u, v) * w_gsd

+--------------+-----------------------------+------------------------------------------+
| term         | form                        | why                                      |
+==============+=============================+==========================================+
| w_radial     | (1 - r/r_max)^p             | lens edges carry the worst residual      |
|              |                             | distortion and vignetting (Map2DFusion)  |
| w_incidence  | max(0, cos(theta))^p        | Hinzmann's "closest to nadir wins", made |
|              |                             | continuous instead of a hard argmax      |
| w_gsd        | (gsd_ref / gsd_frame)^p     | a lower or zoomed-in UAV resolves more   |
|              |                             | detail and should win the pixel          |
+--------------+-----------------------------+------------------------------------------+

Weights live in **source-image space** and are warped by the same homography as the pixels,
which is what Map2DFusion does and what makes the warped weight automatically consistent
with the warped colour.

Incidence shortcut
------------------
``cos(theta) = -d_w.z / |d_w|`` where ``d_w = R @ d_c``. Two observations collapse this to
almost nothing:

1. R is orthogonal, so ``|d_w| == |d_c|``, and ``d_w.z`` is just ``R[2, :] @ d_c``. The map
   therefore depends only on the **third row of R** -- no 3x3 multiply per pixel.
2. ``|d_c| = sqrt(nx^2 + ny^2 + 1)`` depends only on the **intrinsics**, not on the pose. So
   its reciprocal is computed once per sensor and cached forever.

What remains per frame is a broadcast multiply-add over two 1-D arrays, a multiply by the
cached reciprocal, a clip and a square. Measured at 1.00 ms for 1280x720 -- against 9.56 ms
for the naive form, and indistinguishable in cost from a 1/16-scale approximation that
carried 6.6e-3 of error. Exactness here is free, so the approximation was dropped.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .camera import Intrinsics

__all__ = [
    "WEIGHT_FLOOR",
    "radial_weight_map",
    "incidence_weight_map",
    "gsd_weight",
    "frame_weight_map",
]

#: Never let a weight reach zero: zero is the canvas's "nothing here yet" sentinel, and a
#: valid-but-terrible pixel must still outrank empty canvas. Map2DFusion floors at 1e-5.
WEIGHT_FLOOR = 1e-5


@lru_cache(maxsize=16)
def _ray_grid(width, height, fx, fy, cx, cy):
    """Cached, pose-independent ray geometry for one set of intrinsics.

    Returns ``(nx, ny, inv_norm)`` where ``nx`` is ``(1, W)``, ``ny`` is ``(H, 1)`` and
    ``inv_norm`` is the full ``(H, W)`` reciprocal ray length. Only ``inv_norm`` is large,
    and it is built once per sensor rather than once per frame.
    """
    nx = ((np.arange(width, dtype=np.float32) - np.float32(cx)) / np.float32(fx))[None, :]
    ny = ((np.arange(height, dtype=np.float32) - np.float32(cy)) / np.float32(fy))[:, None]
    inv_norm = (1.0 / np.sqrt(nx * nx + ny * ny + np.float32(1.0))).astype(np.float32)
    for a in (nx, ny, inv_norm):
        a.setflags(write=False)
    return nx, ny, inv_norm


@lru_cache(maxsize=16)
def radial_weight_map(
    width: int,
    height: int,
    cx: float | None = None,
    cy: float | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """``(1 - r/r_max)^power``: 1.0 at the optical axis, falling to the floor at the corners.

    ``r`` is measured from the **principal point**, not the geometric centre of the array.
    Vignetting and residual distortion are radial about the optical axis, so for a camera
    whose principal point is offset the two differ -- and the optical axis is the physically
    meaningful one. ``r_max`` is the distance to the furthest corner, so the weight reaches
    the floor exactly once.

    Cached: depends only on the sensor geometry, so it is built once and reused for every
    frame forever. Returned read-only, which makes that sharing safe.
    """
    px = (width - 1) / 2.0 if cx is None else float(cx)
    py = (height - 1) / 2.0 if cy is None else float(cy)

    ys = (np.arange(height, dtype=np.float32) - np.float32(py))[:, None]
    xs = (np.arange(width, dtype=np.float32) - np.float32(px))[None, :]
    r = np.sqrt(xs * xs + ys * ys)

    r_max = max(
        float(np.hypot(px - c[0], py - c[1]))
        for c in ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0))
    )
    w = np.clip(1.0 - r / max(r_max, 1e-9), 0.0, 1.0) ** np.float32(power)
    out = np.maximum(w, WEIGHT_FLOOR).astype(np.float32)
    out.setflags(write=False)
    return out


def incidence_weight_map(
    intr: Intrinsics, R_enu_cam, power: float = 2.0
) -> np.ndarray:
    """``max(0, cos(theta))^power`` per source pixel; theta is the angle off **nadir**.

    Encodes Hinzmann's best-viewing-angle rule as a continuous weight rather than a hard
    per-cell argmax, so the seam between two UAVs falls where their view quality actually
    crosses over instead of on an arbitrary boundary.

    Note the convention: a gimbal dipped ``d`` degrees below the **horizon** looks
    ``90 - d`` degrees off nadir, so the centre weight is ``sin(d) ** power``, not
    ``cos(d) ** power``.
    """
    R = np.asarray(R_enu_cam, dtype=np.float64)
    nx, ny, inv_norm = _ray_grid(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)

    # cos(theta) = -(R[2, :] @ d_c) * inv_norm.
    # The first expression must materialise (H, W) in one go: nx is (1, W) and ny is (H, 1),
    # so an in-place chain starting from nx would try to broadcast (H, 1) into a (1, W)
    # output and fail. After this, everything is in place.
    out = (nx * np.float32(-R[2, 0])) - (np.float32(R[2, 1]) * ny) - np.float32(R[2, 2])
    out *= inv_norm
    np.clip(out, 0.0, 1.0, out=out)
    if power == 2.0:
        out *= out  # cheaper and exact for the default
    else:
        out **= np.float32(power)
    np.maximum(out, WEIGHT_FLOOR, out=out)
    return out


def gsd_weight(gsd_frame_m_per_px: float, gsd_ref_m_per_px: float, power: float = 1.0) -> float:
    """Scalar favouring the sharper source. >1 when this frame out-resolves the canvas.

    ``gsd_frame`` is ground metres per *source* pixel; ``gsd_ref`` is the canvas GSD. A UAV
    flying lower, or zoomed in, produces a smaller number and therefore a larger weight.
    """
    if gsd_frame_m_per_px <= 0.0:
        return WEIGHT_FLOOR
    return float(max(gsd_ref_m_per_px / gsd_frame_m_per_px, WEIGHT_FLOOR) ** power)


def frame_weight_map(
    intr: Intrinsics,
    R_enu_cam,
    gsd_frame_m_per_px: float,
    gsd_ref_m_per_px: float,
    radial_power: float = 2.0,
    incidence_power: float = 2.0,
    gsd_power: float = 1.0,
) -> np.ndarray:
    """The full ``w_radial * w_incidence * w_gsd`` map in source-image space, float32.

    Warp this with the frame's homography and hand the result to
    :meth:`uavmosaic.canvas.Canvas.composite`.
    """
    # incidence_weight_map returns a fresh array, so everything after it is in place --
    # a full-resolution temporary per term would cost more than the maths does.
    w = incidence_weight_map(intr, R_enu_cam, incidence_power)
    w *= radial_weight_map(intr.width, intr.height, intr.cx, intr.cy, radial_power)
    w *= np.float32(gsd_weight(gsd_frame_m_per_px, gsd_ref_m_per_px, gsd_power))
    np.maximum(w, WEIGHT_FLOOR, out=w)
    return w
