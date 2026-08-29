"""Synthetic world and camera renderer -- stands in for Unity, and gives ground truth.

This is the key validation lever. :func:`render_view` produces exactly what a camera at a
given pose would see of a known ground texture, by inverse-warping that texture through the
*same* homography the fusion path derives independently from telemetry. So:

    world --render--> camera frames --UDP--> fuse --> mosaic

must reconstruct the original world. Any sign error, axis swap or scale mistake anywhere in
the chain shows up as a mosaic that does not match the source, which is a far stronger
statement than any unit test can make on its own.

Note that ``render_view`` deliberately builds its homography from the same
:func:`~uavmosaic.georef.compute_footprint` the pipeline uses. That is not circular for the
properties being tested: it validates transport, reassembly, weighting, ROI arithmetic,
compositing and threading end to end. The *geometry* itself is pinned independently by the
closed-form and round-trip tests in ``tests/test_georef.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics
from .coords import CanvasGeometry, euler_chain_R_enu_cam
from .georef import GroundPlane, compute_footprint

__all__ = ["make_world", "render_view", "UavState", "FlightPlan", "unity_from_enu_pose"]


# --------------------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------------------


def make_world(geom: CanvasGeometry, seed: int = 7) -> np.ndarray:
    """A canvas-sized BGR ground texture with structure at several scales.

    Deliberately contains long straight edges (roads) and a metric grid: those are what make
    a registration error visible to the eye, where noise alone would hide it.
    """
    rng = np.random.default_rng(seed)
    h, w = geom.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # terrain base: low-frequency mottling, upscaled from a small random field
    coarse = rng.integers(55, 105, (max(h // 64, 2), max(w // 64, 2), 3), dtype=np.uint8)
    img[:] = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.add(img, rng.integers(0, 18, (h, w, 3), dtype=np.uint8))

    # fields
    for _ in range(70):
        x, y = rng.integers(0, w), rng.integers(0, h)
        bw = int(rng.integers(60, 420) / geom.gsd * 0.5)
        bh = int(rng.integers(60, 420) / geom.gsd * 0.5)
        colour = tuple(int(c) for c in rng.integers(40, 190, 3))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)

    # roads on a 200 m lattice
    step_px = max(int(200.0 / geom.gsd), 4)
    road = max(int(8.0 / geom.gsd), 1)
    for x in range(0, w, step_px):
        cv2.line(img, (x, 0), (x, h), (210, 210, 205), road)
    for y in range(0, h, step_px):
        cv2.line(img, (0, y), (w, y), (210, 210, 205), road)

    # buildings
    for _ in range(320):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = max(int(rng.integers(12, 45) / geom.gsd * 0.5), 3)
        cv2.rectangle(img, (x, y), (x + s, y + int(s * 0.7)),
                      tuple(int(c) for c in rng.integers(120, 250, 3)), -1)

    # a 100 m metric grid with labels, so scale errors are readable off the mosaic
    grid_px = max(int(100.0 / geom.gsd), 8)
    for x in range(0, w, grid_px):
        cv2.line(img, (x, 0), (x, h), (0, 0, 0), 1)
    for y in range(0, h, grid_px):
        cv2.line(img, (0, y), (w, y), (0, 0, 0), 1)
    for x in range(0, w, grid_px * 2):
        for y in range(0, h, grid_px * 2):
            e, n = geom.px_to_enu(x, y)
            cv2.putText(img, f"{int(e)},{int(n)}", (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_view(
    world: np.ndarray,
    geom: CanvasGeometry,
    intr: Intrinsics,
    R_enu_cam,
    cam_enu,
    plane_z: float = 0.0,
) -> np.ndarray | None:
    """What this camera sees of ``world``. None if the pose has no valid footprint.

    Implemented as ``warpPerspective(..., WARP_INVERSE_MAP)``: with that flag the supplied
    matrix maps *destination* (camera) pixels to *source* (world) pixels, which is exactly
    the footprint homography, so no inversion is needed.
    """
    from .georef import GeorefError

    try:
        fp = compute_footprint(
            intr, R_enu_cam, cam_enu, GroundPlane(plane_z, "default"), geom,
            max_incidence_deg=89.0, clamp_factor=1e9, allow_lower_half=False,
        )
    except GeorefError:
        return None

    H = fp.homography_to_roi((0, 0, geom.width, geom.height))
    return cv2.warpPerspective(
        world, H, (intr.width, intr.height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


# --------------------------------------------------------------------------------------
# Flight
# --------------------------------------------------------------------------------------


@dataclass
class UavState:
    uav_id: int
    cam_enu: np.ndarray
    R_enu_cam: np.ndarray
    intr: Intrinsics
    lrf_slant_m: float | None
    agl_m: float | None
    zoom: float


class FlightPlan:
    """Four UAVs flying a lawnmower survey, one lane each -- mirroring UavRandomFlight.cs.

    Kept deliberately in step with the Unity script so what the simulator produces is what the
    real fleet will produce. Lane spacing is set against the *zoomed-in* footprint, so the four
    footprints separate completely when the cameras zoom in -- the no-overlap case that
    feature-based stitching cannot handle and direct georeferencing does not care about -- and
    overlap heavily when they zoom out.
    """

    def __init__(
        self,
        geom: CanvasGeometry,
        n_uavs: int = 4,
        hfovs=(62.0, 55.0, 48.0, 70.0),
        img_size=(1280, 720),
        img_span_frac: float = 0.22,
        leg_frac: float = 0.55,
        transit_period_s: float = 40.0,
    ) -> None:
        """Everything is expressed as a fraction of the AOI, not in absolute metres.

        Hardcoded metres only work for one AOI size. Deriving the track, spacing and
        altitude from the AOI means the same plan is sensible whether the mission box is
        800 m or 8 km across, and the UAVs stay inside it by construction.
        """
        self.geom = geom
        self.n = max(n_uavs, 1)
        self.hfovs = hfovs
        self.img_w, self.img_h = img_size

        self.span_e = geom.e_max - geom.e_min
        self.span_n = geom.n_max - geom.n_min
        self.e0 = (geom.e_min + geom.e_max) / 2.0
        self.n0 = (geom.n_min + geom.n_max) / 2.0

        # abreast across the AOI, inset so no footprint centre sits on the boundary
        self.spacing = self.span_e / (self.n + 1)
        self.leg = self.span_n * leg_frac
        self.speed = 2.0 * self.leg / transit_period_s
        self.wander = self.span_e * 0.02

        # altitude chosen so one footprint spans ~img_span_frac of the AOI width, staggered
        # per UAV so the GSD weight has something to arbitrate between
        base = self.span_e * img_span_frac / (2.0 * math.tan(math.radians(hfovs[0] / 2.0)))
        self.altitudes = tuple(base * (1.0 + 0.18 * i) for i in range(self.n))

    def state(self, uav_id: int, t: float) -> UavState:
        # lawnmower: straight legs along the lane, alternating direction, stepping sideways
        # by a third of a lane each pass so the lane fills in evenly.
        passes = 3
        leg_time = 2.0 * self.leg / max(self.speed, 1e-9)
        u = (t % leg_time) / leg_time
        along = self.leg * (2.0 * u if u < 0.5 else 2.0 * (1.0 - u)) - self.leg / 2.0
        heading_north = u < 0.5

        which_pass = int(t / leg_time) % passes
        lane_centre = (uav_id - (self.n - 1) / 2.0) * self.spacing
        lateral = lane_centre - self.spacing / 2.0 + (which_pass + 0.5) * (self.spacing / passes)
        e = self.e0 + lateral
        n = self.n0 + along
        alt = self.altitudes[uav_id % len(self.altitudes)] * (
            1.0 + 0.08 * math.sin(t * 0.4 + uav_id * 1.7)
        )

        # near-nadir with a gentle wander, so incidence weighting has something to arbitrate
        yaw = (0.0 if heading_north else math.pi) + math.radians(8.0 * math.sin(t * 0.5 + uav_id))
        dip = math.radians(90.0 - (6.0 + 5.0 * math.sin(t * 0.35 + uav_id * 2.1)))
        roll = math.radians(4.0 * math.sin(t * 0.6 + uav_id))

        R = euler_chain_R_enu_cam((yaw, -dip, roll))
        cam = np.array([e, n, alt])

        # Optical zoom only ever narrows the field of view -- it never goes wider than the
        # lens's native FOV. Letting zoom drop below 1.0 produced a 93 deg horizontal FOV,
        # which no UAV EO camera has, and pushed the footprint into grazing angles where the
        # flat-plane projection smears badly.
        zoom = 1.15 + 0.35 * math.sin(t * 0.2 + uav_id)  # 0.80 .. 1.50, clamped below
        zoom = max(zoom, 1.0)
        intr = Intrinsics.from_hfov(
            self.hfovs[uav_id % len(self.hfovs)] / zoom, self.img_w, self.img_h
        )

        boresight = R @ np.array([0.0, 0.0, 1.0])
        slant = float(alt / max(-boresight[2], 1e-6))
        return UavState(uav_id, cam, R, intr, slant, float(alt), zoom)


def unity_from_enu_pose(cam_enu, R_enu_cam) -> tuple[tuple, tuple]:
    """Invert the pipeline's own conversion to produce what Unity would have sent.

    The sim must transmit **raw Unity values**, exactly like the C# sender, so that the
    receiver exercises the real LH->RH path rather than being handed pre-converted data.

    Returns ``(pos_unity_xyz, quat_world_cam_xyzw)``.
    """
    from .coords import F_UNITYCAM_CVCAM, S_ENU_UNITY

    pos_unity = tuple(float(v) for v in (S_ENU_UNITY @ np.asarray(cam_enu, dtype=np.float64)))

    # R_enu_cam = S @ R_u @ F  =>  R_u = S^-1 @ R_enu_cam @ F^-1 = S @ R_enu_cam @ F
    R_u = S_ENU_UNITY @ np.asarray(R_enu_cam, dtype=np.float64) @ F_UNITYCAM_CVCAM

    # matrix -> quaternion (x, y, z, w), branchless-ish Shepperd form for numerical safety
    m = R_u
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return pos_unity, (float(x), float(y), float(z), float(w))
