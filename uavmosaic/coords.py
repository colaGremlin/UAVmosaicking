"""Coordinate frames and conversions -- the single source of truth for all handedness changes.

Every left-handed <-> right-handed operation in this project lives in this module. Nothing
else -- not the Unity sender, not the georeferencing, not the canvas -- is permitted to
transpose an axis or flip a sign. Correia et al. (Sensors 2022, 22, 604) spend five pages on
frame chains precisely because this is where projects silently break.

Frames
------
``W_u``  Unity world      X right, Y up,   Z forward   LEFT-handed
``C_u``  Unity camera     X right, Y up,   Z forward   LEFT-handed   (optical axis = +Z)
``C``    CV camera        X right, Y down, Z forward   right-handed  (optical axis = +Z)
``E``    Local ENU        X East,  Y North, Z Up       right-handed

Two constant matrices bridge them, ``S`` and ``F``. Both are involutions (S@S = F@F = I) and
both have det = -1, which is exactly what converts handedness::

            [1 0 0]                      [1  0  0]
    S   =   [0 0 1]              F   =   [0 -1  0]
            [0 1 0]                      [0  0  1]
    ENU <- Unity world           Unity cam <- CV cam
    E = x_u, N = z_u, U = y_u    y_cv = -y_unitycam

The composite rotation, verified by hand in the Phase-1 blueprint and by
``tests/test_coords.py``::

    R_ENU_CV = S @ R_unity @ F          det = (-1)(+1)(-1) = +1  (proper rotation)

References
----------
Correia et al. 2022, Sensors 22(4):604 -- Eq. 39 (camera->gimbal), Eq. 49 (NED->ENU),
    Eq. 19 (ZYX DCM), Eq. 60-61 (z_C determination).
Argus ``advanced_mapping._build_camera_rotation`` -- ENU world with X-right/Y-down/Z-fwd camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "S_ENU_UNITY",
    "F_UNITYCAM_CVCAM",
    "R_CAM_TO_GIMBAL",
    "R_NED_TO_ENU",
    "quat_to_matrix",
    "unity_quat_to_R_enu_cam",
    "unity_pos_to_enu",
    "dcm_zyx",
    "euler_chain_R_enu_cam",
    "CanvasGeometry",
    "GeodeticAnchor",
    "WGS84_A",
    "WGS84_E2",
]

# --------------------------------------------------------------------------------------
# Constant frame bridges
# --------------------------------------------------------------------------------------

#: ENU <- Unity world.  E = x_unity, N = z_unity, U = y_unity.  det = -1 (LH -> RH).
S_ENU_UNITY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)

#: Unity camera <- CV camera.  Flips the image-Y axis (CV y-down vs Unity y-up).
#: Self-inverse, so it also serves as CV camera <- Unity camera.  det = -1 (LH -> RH).
F_UNITYCAM_CVCAM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

#: Camera (X-right, Y-down, Z-fwd) -> Gimbal (X-fwd, Y-right, Z-down).  Correia Eq. 39.
#: Only used by the Euler-chain path; the quaternion path does not need it.
R_CAM_TO_GIMBAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

#: NED -> ENU.  Swap north/east, negate down.  Correia Eq. 49.
R_NED_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


# --------------------------------------------------------------------------------------
# Quaternion path (the primary path -- what Unity sends)
# --------------------------------------------------------------------------------------


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from a quaternion in ``(x, y, z, w)`` order.

    This is the standard formula, and it is deliberately *not* modified for Unity's
    left-handedness. Unity's own ``Quaternion * Vector3`` operator implements exactly this
    algebra, so applying it to Unity's ``(x, y, z, w)`` reproduces Unity's rotation of
    vectors *within Unity's coordinate system*. The handedness change is carried entirely
    by :data:`S_ENU_UNITY` and :data:`F_UNITYCAM_CVCAM`, never by the quaternion.

    The quaternion is normalised first; a zero quaternion raises.
    """
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("degenerate quaternion (norm ~ 0)")
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def unity_quat_to_R_enu_cam(x: float, y: float, z: float, w: float) -> np.ndarray:
    """``R_ENU_CV`` from Unity's world-space camera quaternion.

    Maps a direction expressed in the **CV camera frame** (X right, Y down, Z along the
    optical axis) into the **local ENU frame**.

        R_ENU_CV = S @ R_unity @ F

    For a nadir camera whose image-up points to Unity +Z this returns
    ``diag(1, -1, -1)``: image-right -> East, image-down -> South, boresight -> Down.
    """
    return S_ENU_UNITY @ quat_to_matrix(x, y, z, w) @ F_UNITYCAM_CVCAM


def unity_pos_to_enu(pos_unity, origin_enu=None) -> np.ndarray:
    """Unity world position (metres) -> local ENU position (metres).

    ``origin_enu`` offsets the local ENU origin from the Unity origin; by default they
    coincide.
    """
    p = S_ENU_UNITY @ np.asarray(pos_unity, dtype=np.float64).reshape(3)
    if origin_enu is not None:
        p = p - np.asarray(origin_enu, dtype=np.float64).reshape(3)
    return p


# --------------------------------------------------------------------------------------
# Euler path (secondary -- real autopilots and the Correia validation case)
# --------------------------------------------------------------------------------------


def dcm_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) direction cosine matrix, **outer frame -> body**.

    Correia Eq. 19. Angles in radians. This matches MATLAB ``angle2dcm(y, p, r)`` and
    NavPy ``angle2dcm`` with the default ZYX sequence, both of which return NED->body.
    Transpose it for body->NED.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, sy * cp, -sp],
            [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
            [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr],
        ]
    )


def euler_chain_R_enu_cam(
    gimbal_ypr,
    uas_ypr=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """``R_ENU_CV`` via the full Correia Euler chain (Eq. 53).

        R_C->ENU = R_NED->ENU @ R_UAS->NED @ R_G->UAS @ R_C->G

    Used for real autopilot telemetry (which reports Euler angles) and to replicate the
    published Correia validation case. The Unity path uses
    :func:`unity_quat_to_R_enu_cam` instead.

    .. warning::
       If the gimbal reports **world-referenced** angles (a stabilised EO/IR ball), leave
       ``uas_ypr`` at identity -- otherwise the airframe rotation is applied twice. This is
       exactly what Correia's Table 1 does, and it is the single most common error in this
       chain.
    """
    R_g_uas = dcm_zyx(*gimbal_ypr).T
    R_uas_ned = dcm_zyx(*uas_ypr).T
    return R_NED_TO_ENU @ R_uas_ned @ R_g_uas @ R_CAM_TO_GIMBAL


# --------------------------------------------------------------------------------------
# ENU <-> canvas pixels
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanvasGeometry:
    """Fixed world-anchored canvas: the affine tying ENU metres to canvas pixels.

    Immutable for the lifetime of a run, which is what lets the fusion loop avoid every
    allocation and lets the H.264 encoder keep a constant frame size.

    Canvas pixel y grows **southward** (north is up), matching image convention.
    """

    e_min: float
    n_min: float
    e_max: float
    n_max: float
    gsd: float  #: ground sample distance, metres per pixel
    elevation: float = 0.0  #: default plane elevation when the LRF gives nothing

    def __post_init__(self) -> None:
        if self.e_max <= self.e_min or self.n_max <= self.n_min:
            raise ValueError(f"degenerate AOI: E[{self.e_min},{self.e_max}] N[{self.n_min},{self.n_max}]")
        if self.gsd <= 0:
            raise ValueError(f"gsd must be positive, got {self.gsd}")

    @property
    def width(self) -> int:
        return int(math.ceil((self.e_max - self.e_min) / self.gsd))

    @property
    def height(self) -> int:
        return int(math.ceil((self.n_max - self.n_min) / self.gsd))

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` -- numpy order."""
        return (self.height, self.width)

    def enu_to_px(self, e, n):
        """ENU east/north (metres, scalar or array) -> canvas ``(x, y)`` float pixels."""
        e = np.asarray(e, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        return (e - self.e_min) / self.gsd, (self.n_max - n) / self.gsd

    def px_to_enu(self, x, y):
        """Canvas ``(x, y)`` pixels -> ENU east/north metres."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return self.e_min + x * self.gsd, self.n_max - y * self.gsd

    def matrix(self) -> np.ndarray:
        """3x3 homogeneous ENU(E,N,1) -> canvas(x,y,1). Handy for composing with a homography."""
        g = self.gsd
        return np.array(
            [
                [1.0 / g, 0.0, -self.e_min / g],
                [0.0, -1.0 / g, self.n_max / g],
                [0.0, 0.0, 1.0],
            ]
        )

    def describe(self) -> str:
        mb = self.height * self.width * 3 / 1e6
        return (
            f"AOI E[{self.e_min:.1f},{self.e_max:.1f}] N[{self.n_min:.1f},{self.n_max:.1f}] m "
            f"@ {self.gsd:.3f} m/px -> {self.width}x{self.height} px ({mb:.1f} MB BGR)"
        )


# --------------------------------------------------------------------------------------
# ENU <-> WGS-84  (exact, via ECEF -- no pyproj dependency)
# --------------------------------------------------------------------------------------

WGS84_A = 6378137.0  #: semi-major axis, metres
WGS84_F = 1.0 / 298.257223563  #: flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  #: first eccentricity squared
WGS84_B = WGS84_A * (1.0 - WGS84_F)  #: semi-minor axis


def _geodetic_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array([(n + h) * cl * co, (n + h) * cl * so, (n * (1.0 - WGS84_E2) + h) * sl])


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Ferrari/Bowring closed form, refined by Newton. Sub-millimetre."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:  # on the polar axis
        lat = math.copysign(math.pi / 2.0, z)
        return lat, lon, abs(z) - WGS84_B
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        z + ep2 * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    for _ in range(3):  # converges immediately; cheap insurance for high altitudes
        sl = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    sl = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    h = p / math.cos(lat) - n
    return lat, lon, h


@dataclass(frozen=True)
class GeodeticAnchor:
    """Ties the local ENU origin to WGS-84, so targets read out as lat/lon.

    Exact both ways (ENU <-> ECEF <-> geodetic), with no tangent-plane approximation and no
    external dependency.
    """

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    def _basis(self):
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        sl, cl = math.sin(lat), math.cos(lat)
        so, co = math.sin(lon), math.cos(lon)
        # rows: East, North, Up -- expressed in ECEF
        r_ecef_to_enu = np.array(
            [
                [-so, co, 0.0],
                [-sl * co, -sl * so, cl],
                [cl * co, cl * so, sl],
            ]
        )
        return r_ecef_to_enu, _geodetic_to_ecef(lat, lon, self.alt_m)

    def enu_to_geodetic(self, e: float, n: float, u: float = 0.0) -> tuple[float, float, float]:
        """Local ENU metres -> ``(lat_deg, lon_deg, alt_m)``."""
        r, ecef0 = self._basis()
        x, y, z = r.T @ np.array([e, n, u], dtype=np.float64) + ecef0
        lat, lon, h = _ecef_to_geodetic(x, y, z)
        return math.degrees(lat), math.degrees(lon), h

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        """``(lat_deg, lon_deg, alt_m)`` -> local ENU metres."""
        r, ecef0 = self._basis()
        ecef = _geodetic_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m)
        return r @ (ecef - ecef0)
