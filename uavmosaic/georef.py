"""Direct georeferencing: pixel rays -> ground plane -> canvas homography.

No feature matching anywhere. Every transform here is arithmetic on the pose, the
intrinsics and the LRF range, which is what lets four non-overlapping footprints land in
correct relative position.

The core operation is a ray-plane intersection, stated identically by three independent
sources::

    Hinzmann Alg. 2 / ortho-forward-homography.cc:91
        lam = -(z_GC - h_ground) / (R_GC @ t_CL).z ;  p' = t_GC + lam * R_GC @ t_CL
    Correia Eq. 60-61
        z_C = (z_ENU - t_z) / (z'_ENU - t_z) ;  P_ENU = z_C*P'_ENU - z_C*T + T
    SkyPin Eq. 4
        lam_i = (z - C_p.z) / d_w.z ;  d_i = C_p + lam_i * d_w

Correia's form is algebraically the same -- take the z-component of ``P = z_C(P' - T) + T``
and solve. Two codebases implement it. This module uses the SkyPin form, being the most
direct.

Why a single homography per frame is exact here
-----------------------------------------------
GROMS partitions an orthophoto into blocks, measures DEM standard deviation per block, and
applies *one remap* to flat blocks, reserving per-pixel correction for rugged ones. The LRF
gives us one plane per UAV per frame, so every block is flat by construction and the whole
frame collapses onto GROMS's fast path. This is not an approximation of a better method; it
is the exact solution of the model we are in.

And because ``cv2.warpPerspective`` samples *backward* internally (it inverts H unless
WARP_INVERSE_MAP is set), we get dense hole-free backward-projection quality at
forward-projection cost. Hinzmann's 25-100x penalty for backward mapping exists only
because his backward path ray-traces a DSM; on a plane the backward map *is* the inverse
homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics
from .coords import CanvasGeometry

__all__ = [
    "GeorefError",
    "RayGeometryError",
    "GroundPlane",
    "Footprint",
    "solve_ground_plane",
    "project_rays_to_plane",
    "compute_footprint",
    "world_to_pixel",
    "DEFAULT_MAX_INCIDENCE_DEG",
    "DEFAULT_CLAMP_FACTOR",
]

#: Beyond this the flat-plane assumption smears badly and resampling degenerates.
#: Map2DFusion gates at ``axis.dot(downLook) < 0.4`` (~66 deg); SkyPin reports that even
#: state-of-the-art matchers struggle at shallow viewing angles.
DEFAULT_MAX_INCIDENCE_DEG = 65.0

#: Reject a footprint wider than this multiple of the nadir footprint diagonal.
#: Straight from Argus ``_compute_ground_footprint``.
DEFAULT_CLAMP_FACTOR = 20.0

#: The boresight must dip at least this far below horizontal for an LRF slant range to be a
#: trustworthy source of plane height.
DEFAULT_MIN_BORESIGHT_DIP_DEG = 10.0

_EPS_DESCENT = 1e-9


class GeorefError(Exception):
    """Frame cannot be georeferenced and must be dropped."""


class RayGeometryError(GeorefError):
    """A ray does not descend to the ground plane (points at or above the horizon)."""


# --------------------------------------------------------------------------------------
# Ground plane
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundPlane:
    """The z = const plane this frame projects onto, and where that number came from."""

    z: float
    tier: str  #: 'lrf_slant' | 'agl' | 'default'

    def __str__(self) -> str:
        return f"z={self.z:.2f}m({self.tier})"


def solve_ground_plane(
    cam_enu,
    R_enu_cam,
    lrf_slant_m: float | None = None,
    agl_m: float | None = None,
    default_z: float = 0.0,
    min_dip_deg: float = DEFAULT_MIN_BORESIGHT_DIP_DEG,
) -> GroundPlane:
    """Three-tier cascade for the per-UAV ground plane height.

    1. ``lrf_slant`` -- intersect the boresight at the measured range and take the altitude
       of the point it actually hits. Correct for an oblique gimbal, where the scene centre
       is nowhere near directly below the aircraft.
    2. ``agl`` -- ``z_cam - agl``. Exact at nadir, degrades over sloped terrain.
    3. ``default`` -- the AOI elevation.

    Tier 1 is skipped when the boresight is too close to horizontal: a grazing ray turns a
    small range error into a large height error, and such returns are often spurious.
    """
    cam_enu = np.asarray(cam_enu, dtype=np.float64).reshape(3)
    R = np.asarray(R_enu_cam, dtype=np.float64).reshape(3, 3)

    if lrf_slant_m is not None and np.isfinite(lrf_slant_m) and lrf_slant_m > 0.0:
        boresight = R @ np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(boresight))
        if n > 1e-12 and (-boresight[2] / n) >= math.sin(math.radians(min_dip_deg)):
            hit = cam_enu + float(lrf_slant_m) * (boresight / n)
            return GroundPlane(z=float(hit[2]), tier="lrf_slant")

    if agl_m is not None and np.isfinite(agl_m) and agl_m > 0.0:
        return GroundPlane(z=float(cam_enu[2] - agl_m), tier="agl")

    return GroundPlane(z=float(default_z), tier="default")


# --------------------------------------------------------------------------------------
# Ray -> plane
# --------------------------------------------------------------------------------------


def project_rays_to_plane(rays_cam, R_enu_cam, cam_enu, plane_z: float):
    """Camera-frame rays ``(N,3)`` -> ``(ground_enu (N,3), lam (N,), cos_theta (N,))``.

    ``cos_theta`` is the cosine of the angle between the ray and straight down: 1.0 at
    nadir, 0.0 at the horizon.

    Raises :class:`RayGeometryError` if any ray fails to descend. A frame containing the
    horizon has no finite footprint, and projecting it anyway smears a handful of pixels
    across kilometres of canvas.
    """
    rays_cam = np.asarray(rays_cam, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_enu_cam, dtype=np.float64).reshape(3, 3)
    C = np.asarray(cam_enu, dtype=np.float64).reshape(3)

    dirs = rays_cam @ R.T  # (N,3) in ENU
    dz = dirs[:, 2]
    if np.any(dz >= -_EPS_DESCENT):
        n_bad = int(np.count_nonzero(dz >= -_EPS_DESCENT))
        raise RayGeometryError(f"{n_bad}/{len(dirs)} ray(s) do not descend to the ground plane")

    lam = (plane_z - C[2]) / dz
    if np.any(lam <= 0.0):
        raise RayGeometryError("negative range: camera is at or below the ground plane")

    ground = C[None, :] + lam[:, None] * dirs
    cos_theta = -dz / np.linalg.norm(dirs, axis=1)
    return ground, lam, cos_theta


def world_to_pixel(pts_enu, intr: Intrinsics, R_enu_cam, cam_enu):
    """ENU points ``(N,3)`` -> image pixels ``(N,2)``; the exact inverse of the projection.

    Backs the round-trip invariant in the tests and the target read-out. Points behind the
    camera come back as NaN rather than silently wrapping to a plausible-looking pixel.
    """
    pts = np.asarray(pts_enu, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_enu_cam, dtype=np.float64).reshape(3, 3)
    C = np.asarray(cam_enu, dtype=np.float64).reshape(3)

    cam = (pts - C[None, :]) @ R  # ENU -> camera; (R.T @ v) written row-wise
    z = cam[:, 2]
    out = np.full((len(pts), 2), np.nan)
    ok = z > 1e-9
    if np.any(ok):
        proj = cam[ok] / z[ok, None]
        out[ok, 0] = intr.fx * proj[:, 0] + intr.cx
        out[ok, 1] = intr.fy * proj[:, 1] + intr.cy
    return out


# --------------------------------------------------------------------------------------
# Footprint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Where one frame lands on the ground and on the canvas."""

    corners_enu: np.ndarray  #: (4,2) E,N -- TL,TR,BR,BL of the source quad
    corners_px: np.ndarray  #: (4,2) canvas pixels, same order
    src_px: np.ndarray  #: (4,2) source-image quad actually used
    lam: np.ndarray  #: (4,) slant range to each corner, metres
    cos_theta: np.ndarray  #: (4,) incidence cosine at each corner
    plane: GroundPlane
    gsd_m_per_px: float  #: ground metres per *source* pixel, at the footprint centre
    used_lower_half: bool

    @property
    def center_enu(self) -> np.ndarray:
        return self.corners_enu.mean(axis=0)

    @property
    def worst_cos_theta(self) -> float:
        return float(self.cos_theta.min())

    def canvas_roi(self, canvas_shape) -> tuple[int, int, int, int] | None:
        """Integer ``(x0, y0, x1, y1)`` bounding box clipped to the canvas, or None.

        None means the footprint lies entirely outside the AOI. Warping into this ROI
        rather than the full canvas is what keeps the loop cheap; Argus does the same.
        """
        h, w = int(canvas_shape[0]), int(canvas_shape[1])
        x0 = int(math.floor(self.corners_px[:, 0].min()))
        y0 = int(math.floor(self.corners_px[:, 1].min()))
        x1 = int(math.ceil(self.corners_px[:, 0].max()))
        y1 = int(math.ceil(self.corners_px[:, 1].max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def homography_to_roi(self, roi) -> np.ndarray:
        """3x3 mapping source-image pixels -> ROI-local canvas pixels."""
        x0, y0, _, _ = roi
        dst = (self.corners_px - np.array([x0, y0], dtype=np.float64)).astype(np.float32)
        return cv2.getPerspectiveTransform(self.src_px.astype(np.float32), dst)


def _footprint_extent(corners_enu, cam_enu) -> float:
    return float(np.abs(corners_enu - np.asarray(cam_enu)[:2][None, :]).max())


def compute_footprint(
    intr: Intrinsics,
    R_enu_cam,
    cam_enu,
    plane: GroundPlane,
    geom: CanvasGeometry,
    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG,
    clamp_factor: float = DEFAULT_CLAMP_FACTOR,
    allow_lower_half: bool = True,
) -> Footprint:
    """Project the frame corners to the ground and thence to canvas pixels.

    Three gates apply in order, each present because a reference implementation was bitten
    by its absence:

    * **descent** -- every corner ray must reach the plane (Map2DFusion ``downLook`` test).
    * **incidence** -- the worst corner must lie within ``max_incidence_deg`` of nadir.
    * **extent** -- the footprint must stay within ``clamp_factor`` nadir-diagonals of the
      aircraft (Argus sanity clamp).

    On failure with ``allow_lower_half``, retries using only the lower half of the image --
    the ground-facing portion under a tilted gimbal, exactly the Argus fallback. If that
    also fails, raises :class:`GeorefError` and the caller drops the frame.
    """
    cam_enu = np.asarray(cam_enu, dtype=np.float64).reshape(3)
    cos_limit = math.cos(math.radians(max_incidence_deg))

    nadir_diag = 2.0 * math.tan(math.radians(intr.diagonal_fov_deg) / 2.0) * max(
        cam_enu[2] - plane.z, 1e-6
    )
    max_extent = clamp_factor * nadir_diag

    w, h = float(intr.width), float(intr.height)
    attempts: list[tuple[np.ndarray, bool]] = [(intr.corners(), False)]
    if allow_lower_half:
        half = h / 2.0
        attempts.append(
            (np.array([[0.0, half], [w, half], [w, h], [0.0, h]], dtype=np.float32), True)
        )

    failures: list[str] = []
    for src_px, lower in attempts:
        tag = "lower-half" if lower else "full"
        try:
            ground, lam, cos_theta = project_rays_to_plane(
                intr.rays(src_px), R_enu_cam, cam_enu, plane.z
            )
        except RayGeometryError as exc:
            failures.append(f"{tag}: {exc}")
            continue

        if cos_theta.min() < cos_limit:
            worst = math.degrees(math.acos(min(1.0, max(-1.0, float(cos_theta.min())))))
            failures.append(f"{tag}: incidence {worst:.1f}deg > {max_incidence_deg:.1f}deg")
            continue

        enu2 = ground[:, :2]
        extent = _footprint_extent(enu2, cam_enu)
        if extent > max_extent:
            failures.append(f"{tag}: extent {extent:.0f}m > {max_extent:.0f}m clamp")
            continue

        px_x, px_y = geom.enu_to_px(enu2[:, 0], enu2[:, 1])
        return Footprint(
            corners_enu=enu2,
            corners_px=np.stack([px_x, px_y], axis=1),
            src_px=np.asarray(src_px, dtype=np.float64),
            lam=lam,
            cos_theta=cos_theta,
            plane=plane,
            gsd_m_per_px=float(lam.mean() / intr.fx),
            used_lower_half=lower,
        )

    raise GeorefError("footprint rejected -- " + "; ".join(failures))
