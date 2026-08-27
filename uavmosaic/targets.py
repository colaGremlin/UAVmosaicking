"""Target read-out: canvas or source pixel -> exact world coordinates.

This is what the mosaic is *for*. Two entry points:

``from_canvas_px``
    An operator clicks the fused canvas. The canvas affine inverts exactly, and the owner
    and stamp buffers say which UAV produced that pixel and when -- so the fix carries its
    own provenance rather than being an anonymous coordinate.

``from_source_px``
    A detector fires on a raw UAV frame. The pixel is back-projected through that frame's own
    pose directly onto the ground plane. This is **exact**, not interpolated.

Why not interpolate across the footprint corners
------------------------------------------------
Argus (``spatial.interpolate_detection_gps``) bilinearly interpolates a detection's position
between the four corner coordinates. For a perspective quadrilateral that is simply the wrong
function: perspective projection is a *projective* map, and bilinear interpolation is not.
The error vanishes at the corners and peaks near the centre, growing with obliquity --
exactly where a tilted gimbal puts most of its detections.
:func:`bilinear_error_metres` quantifies the gap for a given geometry, and the tests show it
reaching tens of metres at moderate tilt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Intrinsics
from .canvas import UNOWNED, Canvas
from .coords import CanvasGeometry, GeodeticAnchor
from .georef import Footprint, project_rays_to_plane, world_to_pixel

__all__ = ["TargetFix", "TargetResolver", "bilinear_error_metres"]


@dataclass(frozen=True)
class TargetFix:
    """A located target, with everything needed to judge how much to trust it."""

    enu: tuple[float, float, float]
    canvas_px: tuple[float, float]
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None

    uav_id: int | None = None
    sensor_id: int | None = None
    t_capture_us: int | None = None
    source_px: tuple[float, float] | None = None

    plane_tier: str | None = None  #: 'lrf_slant' | 'agl' | 'default'
    plane_z: float | None = None
    cos_theta: float | None = None  #: 1.0 nadir, 0.0 horizon
    age_s: float | None = None

    @property
    def incidence_deg(self) -> float | None:
        if self.cos_theta is None:
            return None
        return float(np.degrees(np.arccos(np.clip(self.cos_theta, -1.0, 1.0))))

    @property
    def quality(self) -> str:
        """A blunt confidence label, driven by the two things that actually degrade a fix."""
        if self.plane_tier == "default":
            return "low (no range measurement, assumed plane)"
        inc = self.incidence_deg
        if inc is None:
            return "unknown"
        if inc > 45.0:
            return f"low (oblique, {inc:.0f} deg off nadir)"
        if inc > 25.0 or self.plane_tier == "agl":
            return f"medium ({inc:.0f} deg off nadir, {self.plane_tier})"
        return f"good ({inc:.0f} deg off nadir, {self.plane_tier})"

    def describe(self) -> str:
        e, n, u = self.enu
        s = f"E {e:9.2f}  N {n:9.2f}  U {u:7.2f} m"
        if self.lat_deg is not None:
            s += f"   {self.lat_deg:.7f}, {self.lon_deg:.7f}"
        if self.uav_id is not None:
            s += f"   [UAV{self.uav_id}]"
        return s + f"   {self.quality}"


class TargetResolver:
    """Converts pixels to world coordinates, in whichever direction is needed."""

    def __init__(self, geom: CanvasGeometry, anchor: GeodeticAnchor | None = None) -> None:
        self.geom = geom
        self.anchor = anchor

    # -- helpers -----------------------------------------------------------------------

    def _geodetic(self, e: float, n: float, u: float):
        if self.anchor is None:
            return None, None, None
        return self.anchor.enu_to_geodetic(e, n, u)

    def enu_to_canvas_px(self, e: float, n: float) -> tuple[float, float]:
        x, y = self.geom.enu_to_px(e, n)
        return float(x), float(y)

    # -- entry points ------------------------------------------------------------------

    def from_canvas_px(
        self,
        x: float,
        y: float,
        canvas: Canvas | None = None,
        plane_z: float = 0.0,
        t_now: float | None = None,
    ) -> TargetFix:
        """Operator clicked ``(x, y)`` on the mosaic.

        With a canvas supplied, the fix also reports which UAV owns that pixel and how old it
        is -- a coordinate read off a 30-second-old pixel deserves less trust than a live one,
        and nothing else in the system would tell you.
        """
        e, n = self.geom.px_to_enu(x, y)
        e, n = float(e), float(n)
        lat, lon, alt = self._geodetic(e, n, plane_z)

        uav_id = None
        age = None
        if canvas is not None:
            xi = int(np.clip(round(x), 0, self.geom.width - 1))
            yi = int(np.clip(round(y), 0, self.geom.height - 1))
            owner = int(canvas.owner[yi, xi])
            if owner != UNOWNED:
                uav_id = owner
                if t_now is not None:
                    age = float(t_now - canvas.stamp[yi, xi])

        return TargetFix(
            enu=(e, n, plane_z), canvas_px=(float(x), float(y)),
            lat_deg=lat, lon_deg=lon, alt_m=alt, uav_id=uav_id, age_s=age,
        )

    def from_source_px(
        self,
        u: float,
        v: float,
        intr: Intrinsics,
        R_enu_cam,
        cam_enu,
        footprint: Footprint,
        uav_id: int | None = None,
        sensor_id: int | None = None,
        t_capture_us: int | None = None,
    ) -> TargetFix:
        """A detection at ``(u, v)`` in a raw frame -> exact ground coordinates.

        Back-projects the single ray and intersects the plane. No interpolation, no
        homography round trip, so the answer is exact for the model.
        """
        ground, _lam, cos_t = project_rays_to_plane(
            intr.rays([[u, v]]), R_enu_cam, cam_enu, footprint.plane.z
        )
        e, n, z = (float(c) for c in ground[0])
        lat, lon, alt = self._geodetic(e, n, z)
        x, y = self.enu_to_canvas_px(e, n)

        return TargetFix(
            enu=(e, n, z), canvas_px=(x, y),
            lat_deg=lat, lon_deg=lon, alt_m=alt,
            uav_id=uav_id, sensor_id=sensor_id, t_capture_us=t_capture_us,
            source_px=(float(u), float(v)),
            plane_tier=footprint.plane.tier, plane_z=footprint.plane.z,
            cos_theta=float(cos_t[0]),
        )

    def to_source_px(self, e: float, n: float, z: float, intr: Intrinsics, R_enu_cam, cam_enu):
        """Where does a known world point appear in this frame? NaN if behind the camera."""
        px = world_to_pixel([[e, n, z]], intr, R_enu_cam, cam_enu)[0]
        return float(px[0]), float(px[1])


def bilinear_error_metres(footprint: Footprint, intr: Intrinsics, R_enu_cam, cam_enu,
                          samples: int = 21) -> float:
    """Worst-case error, in metres, of corner-bilinear interpolation vs exact projection.

    Quantifies what the Argus-style shortcut costs for this specific geometry. Near nadir it
    is small; under a tilted gimbal it grows quickly, because perspective compresses the far
    half of the image and bilinear interpolation does not know that.
    """
    tl, tr, br, bl = (footprint.corners_enu[i] for i in range(4))
    src = footprint.src_px
    u0, v0 = src[0]
    u1, v1 = src[2]

    us = np.linspace(u0, u1, samples)
    vs = np.linspace(v0, v1, samples)
    U, V = np.meshgrid(us, vs)
    pts = np.stack([U.ravel(), V.ravel()], axis=1)

    exact, _, _ = project_rays_to_plane(
        intr.rays(pts), R_enu_cam, cam_enu, footprint.plane.z
    )
    exact = exact[:, :2]

    a = ((pts[:, 0] - u0) / (u1 - u0))[:, None]
    b = ((pts[:, 1] - v0) / (v1 - v0))[:, None]
    top = tl + a * (tr - tl)
    bot = bl + a * (br - bl)
    approx = top + b * (bot - top)

    return float(np.linalg.norm(exact - approx, axis=1).max())
