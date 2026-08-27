"""Camera intrinsics, rebuilt every frame because the optical zoom is variable.

Correia et al. Eq. 30-33 give the conversion that matters here::

    f_x = f[mm] * ImageWidth[px]  / SensorWidth[mm]
    f_y = f[mm] * ImageHeight[px] / SensorHeight[mm]

Unity is more likely to report a field of view than a focal length, so both routes are
supported and :meth:`Intrinsics.disagreement` cross-checks one against the other -- a
mismatch means the sensor size and the FOV in the Unity camera do not describe the same
optics, which is a configuration bug worth catching at frame 1 rather than by staring at a
misaligned mosaic.

EO and IR carry independent intrinsics. They are different sensors with different sensor
sizes, resolutions and fields of view, so each gets its own K and its own homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Intrinsics", "hfov_from_focal", "focal_px_from_hfov"]


def hfov_from_focal(f_mm: float, sensor_w_mm: float) -> float:
    """Horizontal field of view in degrees. Correia Eq. 66."""
    return math.degrees(2.0 * math.atan(sensor_w_mm / (2.0 * f_mm)))


def focal_px_from_hfov(hfov_deg: float, img_w: int) -> float:
    """Focal length in pixels from a horizontal FOV."""
    half = math.radians(hfov_deg) / 2.0
    if not (1e-6 < half < math.pi / 2 - 1e-6):
        raise ValueError(f"hfov_deg out of range: {hfov_deg}")
    return (img_w / 2.0) / math.tan(half)


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for one sensor at one instant (one zoom setting).

    Distortion is an identity hook: Unity renders an ideal pinhole, so there is nothing to
    undo, but the seam exists so real optics can be dropped in without touching the
    georeferencing code (Correia Eq. 65 gives the radial model).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...] | None = None  #: (k1,k2,p1,p2,k3) or None for an ideal pinhole

    _K: np.ndarray = field(init=False, repr=False, compare=False)
    _K_inv: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (self.fx > 0 and self.fy > 0):
            raise ValueError(f"focal lengths must be positive, got fx={self.fx} fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"bad image size {self.width}x{self.height}")
        k = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )
        # frozen dataclass: bypass the setattr guard for the cached derivatives
        object.__setattr__(self, "_K", k)
        object.__setattr__(self, "_K_inv", np.linalg.inv(k))

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_focal_sensor(
        cls,
        f_mm: float,
        sensor_w_mm: float,
        sensor_h_mm: float,
        img_w: int,
        img_h: int,
        cx: float | None = None,
        cy: float | None = None,
        dist: tuple[float, ...] | None = None,
    ) -> "Intrinsics":
        """Correia Eq. 32-33. Principal point defaults to the image centre."""
        if f_mm <= 0 or sensor_w_mm <= 0 or sensor_h_mm <= 0:
            raise ValueError("focal length and sensor dimensions must be positive")
        return cls(
            fx=f_mm * img_w / sensor_w_mm,
            fy=f_mm * img_h / sensor_h_mm,
            cx=img_w / 2.0 if cx is None else cx,
            cy=img_h / 2.0 if cy is None else cy,
            width=img_w,
            height=img_h,
            dist=dist,
        )

    @classmethod
    def from_hfov(
        cls,
        hfov_deg: float,
        img_w: int,
        img_h: int,
        square_pixels: bool = True,
        dist: tuple[float, ...] | None = None,
    ) -> "Intrinsics":
        """From a horizontal FOV. With ``square_pixels`` (the Unity case), ``fy == fx``."""
        fx = focal_px_from_hfov(hfov_deg, img_w)
        return cls(
            fx=fx,
            fy=fx if square_pixels else focal_px_from_hfov(hfov_deg, img_h),
            cx=img_w / 2.0,
            cy=img_h / 2.0,
            width=img_w,
            height=img_h,
            dist=dist,
        )

    @classmethod
    def from_matrix(cls, K, img_w: int, img_h: int, dist=None) -> "Intrinsics":
        K = np.asarray(K, dtype=np.float64)
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=img_w, height=img_h, dist=dist)

    # -- access ------------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def K_inv(self) -> np.ndarray:
        return self._K_inv

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    @property
    def diagonal_fov_deg(self) -> float:
        return math.degrees(
            2.0 * math.atan(math.hypot(self.width / 2.0, self.height / 2.0) / self.fx)
        )

    def corners(self) -> np.ndarray:
        """Image corners TL, TR, BR, BL as float32 ``(4, 2)`` -- the warp source points.

        Uses ``width``/``height`` rather than ``width-1``/``height-1`` so the quad spans the
        full pixel extent, matching how ``cv2.warpPerspective`` samples.
        """
        w, h = float(self.width), float(self.height)
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)

    def rays(self, pts_px) -> np.ndarray:
        """Pixels ``(N, 2)`` -> unnormalised camera-frame ray directions ``(N, 3)``.

        ``d = K^-1 [u, v, 1]^T`` -- SkyPin Eq. 2, Argus ``_compute_ground_footprint``.
        Left unnormalised: the ray-plane solve divides by ``d.z`` so scale cancels.
        """
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        return homog @ self._K_inv.T

    def disagreement(self, reported_hfov_deg: float | None) -> float | None:
        """Absolute degrees between our ``hfov_deg`` and what the telemetry claims.

        Returns ``None`` when nothing was reported. Callers should warn above ~0.5 deg: it
        means the sensor size and the Unity camera FOV describe different optics.
        """
        if reported_hfov_deg is None:
            return None
        return abs(self.hfov_deg - float(reported_hfov_deg))

    def undistort(self, image):
        """Identity for an ideal pinhole; real radial/tangential undistortion otherwise."""
        if not self.dist:
            return image
        import cv2  # local import: keeps this module importable without OpenCV

        return cv2.undistort(image, self._K, np.asarray(self.dist, dtype=np.float64))

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height} fx={self.fx:.2f} fy={self.fy:.2f} "
            f"c=({self.cx:.1f},{self.cy:.1f}) hfov={self.hfov_deg:.2f}deg"
        )
