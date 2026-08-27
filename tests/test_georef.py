"""Gating tests for direct georeferencing.

These deliberately do NOT gate on Correia's published target coordinate. That number turned
out to be internally inconsistent with the paper's own stated inputs (see
``test_correia_reference_case`` for the full finding), so the build is gated instead on:

* closed-form analytic footprints that can be derived by hand, and
* a projection round-trip invariant (pixel -> ground -> pixel must be the identity), and
* homography exactness (the 4-corner homography must agree with a direct ray-plane solve
  for *every* interior pixel -- true only if the ground really is a plane and the maths is
  right).

The round-trip is the strongest of the three: a sign error anywhere in K, R or the plane
solve breaks it immediately.
"""

import math

import numpy as np
import pytest

from uavmosaic.camera import Intrinsics
from uavmosaic.coords import (
    R_CAM_TO_GIMBAL,
    R_NED_TO_ENU,
    CanvasGeometry,
    dcm_zyx,
    euler_chain_R_enu_cam,
)
from uavmosaic.georef import (
    GeorefError,
    GroundPlane,
    RayGeometryError,
    compute_footprint,
    project_rays_to_plane,
    solve_ground_plane,
    world_to_pixel,
)

R_NADIR = np.diag([1.0, -1.0, -1.0])  # image-right->E, image-down->S, boresight->Down
GEOM = CanvasGeometry(e_min=-1000.0, n_min=-1000.0, e_max=1000.0, n_max=1000.0, gsd=0.25)


def nadir_intr(w=1280, h=720, hfov=60.0):
    return Intrinsics.from_hfov(hfov, w, h)


# --------------------------------------------------------------------------------------
# Closed-form analytic cases
# --------------------------------------------------------------------------------------


def test_nadir_centre_pixel_lands_directly_below():
    intr = nadir_intr()
    cam = np.array([100.0, 200.0, 300.0])
    ground, lam, cos_t = project_rays_to_plane(
        intr.rays([[intr.cx, intr.cy]]), R_NADIR, cam, 0.0
    )
    assert np.allclose(ground[0], [100.0, 200.0, 0.0], atol=1e-9)
    assert np.isclose(lam[0], 300.0)  # straight down, so range == altitude
    assert np.isclose(cos_t[0], 1.0)  # perfectly nadir


def test_nadir_footprint_matches_tan_half_fov():
    """Half-width = h*tan(hfov/2), half-height = h*tan(vfov/2). Pure trigonometry."""
    intr = nadir_intr(hfov=60.0)
    alt = 250.0
    cam = np.array([0.0, 0.0, alt])
    fp = compute_footprint(intr, R_NADIR, cam, GroundPlane(0.0, "default"), GEOM)

    expect_hw = alt * math.tan(math.radians(intr.hfov_deg) / 2.0)
    expect_hh = alt * math.tan(math.radians(intr.vfov_deg) / 2.0)

    e, n = fp.corners_enu[:, 0], fp.corners_enu[:, 1]
    assert np.isclose(e.max(), expect_hw, rtol=1e-9)
    assert np.isclose(e.min(), -expect_hw, rtol=1e-9)
    assert np.isclose(n.max(), expect_hh, rtol=1e-9)
    assert np.isclose(n.min(), -expect_hh, rtol=1e-9)


def test_nadir_footprint_corner_order_is_tl_tr_br_bl():
    """Image TL must land NW, TR->NE, BR->SE, BL->SW for a north-up nadir view."""
    intr = nadir_intr()
    fp = compute_footprint(
        intr, R_NADIR, np.array([0.0, 0.0, 200.0]), GroundPlane(0.0, "default"), GEOM
    )
    tl, tr, br, bl = fp.corners_enu
    assert tl[0] < 0 and tl[1] > 0, "top-left -> north-west"
    assert tr[0] > 0 and tr[1] > 0, "top-right -> north-east"
    assert br[0] > 0 and br[1] < 0, "bottom-right -> south-east"
    assert bl[0] < 0 and bl[1] < 0, "bottom-left -> south-west"


def test_footprint_scales_linearly_with_altitude():
    intr = nadir_intr()
    areas = []
    for alt in (100.0, 200.0, 400.0):
        fp = compute_footprint(
            intr, R_NADIR, np.array([0.0, 0.0, alt]), GroundPlane(0.0, "default"), GEOM
        )
        e, n = fp.corners_enu[:, 0], fp.corners_enu[:, 1]
        areas.append((e.max() - e.min()) * (n.max() - n.min()))
    # area goes as altitude^2
    assert np.isclose(areas[1] / areas[0], 4.0, rtol=1e-9)
    assert np.isclose(areas[2] / areas[1], 4.0, rtol=1e-9)


def test_oblique_boresight_lands_at_h_over_tan_elevation():
    """A gimbal pitched d degrees below horizontal, aimed north, puts the scene centre at
    ground range h/tan(d) to the north. Closed form, independent of the code path."""
    intr = nadir_intr()
    alt = 300.0
    for dip_deg in (30.0, 45.0, 60.0, 80.0):
        R = euler_chain_R_enu_cam((0.0, math.radians(-dip_deg), 0.0))  # yaw 0 = north
        cam = np.array([0.0, 0.0, alt])
        ground, lam, cos_t = project_rays_to_plane(
            intr.rays([[intr.cx, intr.cy]]), R, cam, 0.0
        )
        expect_north = alt / math.tan(math.radians(dip_deg))
        assert np.isclose(ground[0, 0], 0.0, atol=1e-9), "no easting for a due-north aim"
        assert np.isclose(ground[0, 1], expect_north, rtol=1e-9), dip_deg
        assert np.isclose(lam[0], alt / math.sin(math.radians(dip_deg)), rtol=1e-9)
        assert np.isclose(cos_t[0], math.sin(math.radians(dip_deg)), rtol=1e-9)


# --------------------------------------------------------------------------------------
# Round-trip invariant -- the strongest check in this file
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw_deg,dip_deg,roll_deg",
    [
        (0.0, 90.0, 0.0),
        (0.0, 60.0, 0.0),
        (37.0, 75.0, 0.0),
        (-120.0, 50.0, 12.0),
        (200.0, 85.0, -25.0),
    ],
)
def test_pixel_to_ground_to_pixel_is_identity(yaw_deg, dip_deg, roll_deg):
    intr = nadir_intr(1920, 1080, hfov=55.0)
    R = euler_chain_R_enu_cam(
        (math.radians(yaw_deg), math.radians(-dip_deg), math.radians(roll_deg))
    )
    cam = np.array([123.0, -456.0, 275.0])

    rng = np.random.default_rng(4242)
    px = np.stack(
        [rng.uniform(0, intr.width, 400), rng.uniform(0, intr.height, 400)], axis=1
    )
    ground, _, _ = project_rays_to_plane(intr.rays(px), R, cam, 0.0)
    back = world_to_pixel(ground, intr, R, cam)

    assert not np.isnan(back).any(), "every projected point must reproject in front"
    # rtol=0 deliberately: numpy's default rtol=1e-5 against ~1900 px values would make
    # this a 0.019 px test while looking like a 1e-9 one. Measured float64 error is 7e-13.
    assert np.allclose(back, px, atol=1e-9, rtol=0), np.abs(back - px).max()


def test_roundtrip_holds_for_a_non_zero_plane():
    intr = nadir_intr()
    R = euler_chain_R_enu_cam((math.radians(15.0), math.radians(-70.0), 0.0))
    cam = np.array([10.0, 20.0, 500.0])
    rng = np.random.default_rng(1)
    px = np.stack(
        [rng.uniform(0, intr.width, 200), rng.uniform(0, intr.height, 200)], axis=1
    )
    for plane_z in (-50.0, 0.0, 120.0):
        ground, _, _ = project_rays_to_plane(intr.rays(px), R, cam, plane_z)
        assert np.allclose(ground[:, 2], plane_z, atol=1e-9, rtol=0), "must land on the plane"
        assert np.allclose(world_to_pixel(ground, intr, R, cam), px, atol=1e-9, rtol=0)


# --------------------------------------------------------------------------------------
# Homography exactness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("yaw_deg,dip_deg,roll_deg", [(0, 90, 0), (48, 62, 8), (-95, 72, -14)])
def test_corner_homography_agrees_with_direct_projection_everywhere(yaw_deg, dip_deg, roll_deg):
    """The 4-corner homography must reproduce the ray-plane solve for *interior* pixels too.

    This is the formal statement of why one homography per frame is exact on a plane: if the
    two ever disagreed, the planar model would be wrong or the corner mapping mis-ordered.
    """
    intr = nadir_intr(1280, 720, hfov=50.0)
    R = euler_chain_R_enu_cam(
        (math.radians(yaw_deg), math.radians(-dip_deg), math.radians(roll_deg))
    )
    cam = np.array([0.0, 0.0, 400.0])
    fp = compute_footprint(intr, R, cam, GroundPlane(0.0, "default"), GEOM)

    roi = (0, 0, GEOM.width, GEOM.height)
    H = fp.homography_to_roi(roi)

    rng = np.random.default_rng(2024)
    px = np.stack(
        [rng.uniform(0, intr.width, 300), rng.uniform(0, intr.height, 300)], axis=1
    )
    ground, _, _ = project_rays_to_plane(intr.rays(px), R, cam, 0.0)
    direct = np.stack(GEOM.enu_to_px(ground[:, 0], ground[:, 1]), axis=1)

    homog = np.hstack([px, np.ones((len(px), 1))]) @ H.T
    via_H = homog[:, :2] / homog[:, 2:3]

    # 1e-3 px, rtol=0. The floor is cv2.getPerspectiveTransform, which *only* accepts
    # float32 (float64 raises), costing ~1.5e-4 px on canvas-scale coordinates. That is far
    # below anything visible and far above float64 noise.
    assert np.allclose(via_H, direct, atol=1e-3, rtol=0), np.abs(via_H - direct).max()


# --------------------------------------------------------------------------------------
# LRF ground-plane cascade
# --------------------------------------------------------------------------------------


def test_plane_tier1_uses_lrf_slant_range():
    """Nadir at 300 m with a 250 m return means the terrain is 50 m above the datum."""
    plane = solve_ground_plane(
        [0.0, 0.0, 300.0], R_NADIR, lrf_slant_m=250.0, agl_m=None, default_z=0.0
    )
    assert plane.tier == "lrf_slant"
    assert np.isclose(plane.z, 50.0)


def test_plane_tier1_is_correct_for_an_oblique_gimbal():
    """The whole point of tier 1: use the altitude of the point the boresight actually hits.

    Dipped 30 deg, a 400 m slant range descends only 400*sin(30) = 200 m.
    """
    R = euler_chain_R_enu_cam((0.0, math.radians(-30.0), 0.0))
    plane = solve_ground_plane([0.0, 0.0, 500.0], R, lrf_slant_m=400.0, default_z=0.0)
    assert plane.tier == "lrf_slant"
    assert np.isclose(plane.z, 500.0 - 200.0, atol=1e-9)


def test_plane_tier2_falls_back_to_agl():
    plane = solve_ground_plane([0.0, 0.0, 300.0], R_NADIR, lrf_slant_m=None, agl_m=280.0)
    assert plane.tier == "agl"
    assert np.isclose(plane.z, 20.0)


def test_plane_tier3_falls_back_to_default():
    plane = solve_ground_plane([0.0, 0.0, 300.0], R_NADIR, default_z=42.0)
    assert plane.tier == "default"
    assert np.isclose(plane.z, 42.0)


def test_grazing_boresight_rejects_the_lrf_return():
    """5 deg below horizontal is too shallow to trust; must fall through to AGL."""
    R = euler_chain_R_enu_cam((0.0, math.radians(-5.0), 0.0))
    plane = solve_ground_plane([0.0, 0.0, 300.0], R, lrf_slant_m=3000.0, agl_m=290.0)
    assert plane.tier == "agl"


@pytest.mark.parametrize("bad", [0.0, -10.0, float("nan"), float("inf")])
def test_invalid_lrf_values_fall_through(bad):
    plane = solve_ground_plane([0.0, 0.0, 300.0], R_NADIR, lrf_slant_m=bad, agl_m=250.0)
    assert plane.tier == "agl"


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------


def test_horizon_ray_is_rejected():
    intr = nadir_intr()
    R_horizontal = euler_chain_R_enu_cam((0.0, 0.0, 0.0))  # dead level, sees the horizon
    with pytest.raises(RayGeometryError):
        project_rays_to_plane(intr.rays(intr.corners()), R_horizontal, [0, 0, 300.0], 0.0)


def test_camera_below_plane_is_rejected():
    intr = nadir_intr()
    with pytest.raises(RayGeometryError):
        project_rays_to_plane(intr.rays([[intr.cx, intr.cy]]), R_NADIR, [0, 0, 100.0], 200.0)


def test_incidence_gate_rejects_a_shallow_frame():
    intr = nadir_intr()
    R = euler_chain_R_enu_cam((0.0, math.radians(-20.0), 0.0))  # 70 deg off nadir
    with pytest.raises(GeorefError, match="incidence"):
        compute_footprint(
            intr,
            R,
            [0.0, 0.0, 300.0],
            GroundPlane(0.0, "default"),
            GEOM,
            max_incidence_deg=65.0,
            allow_lower_half=False,
        )


def test_lower_half_fallback_rescues_a_tilted_frame():
    """A frame whose top edge grazes the horizon is still usable from the waist down."""
    intr = nadir_intr(hfov=60.0)
    # dip just under half the vertical FOV -> the top rows point at/above the horizon
    dip = intr.vfov_deg / 2.0 - 2.0
    R = euler_chain_R_enu_cam((0.0, math.radians(-dip), 0.0))
    cam = [0.0, 0.0, 300.0]

    with pytest.raises(RayGeometryError):
        project_rays_to_plane(intr.rays(intr.corners()), R, cam, 0.0)

    fp = compute_footprint(
        intr, R, cam, GroundPlane(0.0, "default"), GEOM,
        max_incidence_deg=89.9, clamp_factor=1e9, allow_lower_half=True,
    )
    assert fp.used_lower_half
    assert np.isfinite(fp.corners_enu).all()


def test_extent_clamp_rejects_a_runaway_footprint():
    """Dip 40 deg: every ray still descends (top ray is 22 deg down) and incidence is
    allowed, so the *extent* gate is the one that must fire. Far edge reaches 742 m against
    a 397 m nadir diagonal, so clamp_factor=1.0 rejects it."""
    intr = nadir_intr(hfov=60.0)
    R = euler_chain_R_enu_cam((0.0, math.radians(-40.0), 0.0))
    with pytest.raises(GeorefError, match="extent"):
        compute_footprint(
            intr, R, [0.0, 0.0, 300.0], GroundPlane(0.0, "default"), GEOM,
            max_incidence_deg=89.9, clamp_factor=1.0, allow_lower_half=False,
        )


def test_extent_clamp_accepts_the_same_frame_when_loosened():
    """Same geometry, generous clamp -> must pass. Guards against the gate always firing."""
    intr = nadir_intr(hfov=60.0)
    R = euler_chain_R_enu_cam((0.0, math.radians(-40.0), 0.0))
    fp = compute_footprint(
        intr, R, [0.0, 0.0, 300.0], GroundPlane(0.0, "default"), GEOM,
        max_incidence_deg=89.9, clamp_factor=20.0, allow_lower_half=False,
    )
    assert not fp.used_lower_half


# --------------------------------------------------------------------------------------
# ROI
# --------------------------------------------------------------------------------------


def test_roi_is_clipped_to_the_canvas():
    intr = nadir_intr()
    fp = compute_footprint(
        intr, R_NADIR, [0.0, 0.0, 200.0], GroundPlane(0.0, "default"), GEOM
    )
    roi = fp.canvas_roi(GEOM.shape)
    assert roi is not None
    x0, y0, x1, y1 = roi
    assert 0 <= x0 < x1 <= GEOM.width
    assert 0 <= y0 < y1 <= GEOM.height


def test_roi_is_none_when_the_footprint_is_outside_the_aoi():
    intr = nadir_intr()
    far = CanvasGeometry(e_min=50_000, n_min=50_000, e_max=51_000, n_max=51_000, gsd=1.0)
    fp = compute_footprint(intr, R_NADIR, [0.0, 0.0, 200.0], GroundPlane(0.0, "default"), far)
    assert fp.canvas_roi(far.shape) is None


def test_roi_local_homography_is_the_canvas_one_translated():
    intr = nadir_intr()
    fp = compute_footprint(
        intr, R_NADIR, [120.0, -80.0, 220.0], GroundPlane(0.0, "default"), GEOM
    )
    roi = fp.canvas_roi(GEOM.shape)
    x0, y0, _, _ = roi
    H_full = fp.homography_to_roi((0, 0, GEOM.width, GEOM.height))
    H_roi = fp.homography_to_roi(roi)

    # Compare *mapped points*, not matrix entries: the two homographies are fitted
    # independently in float32, so their coefficients differ at ~1e-4 even though they agree
    # as maps. Comparing entries would be testing OpenCV's rounding, not our geometry.
    rng = np.random.default_rng(77)
    px = np.stack(
        [rng.uniform(0, intr.width, 200), rng.uniform(0, intr.height, 200)], axis=1
    )
    homog = np.hstack([px, np.ones((len(px), 1))])
    a = homog @ H_roi.T
    b = homog @ H_full.T
    a = a[:, :2] / a[:, 2:3] + np.array([x0, y0])
    b = b[:, :2] / b[:, 2:3]
    assert np.allclose(a, b, atol=1e-3, rtol=0), np.abs(a - b).max()


# --------------------------------------------------------------------------------------
# Correia et al. reference case -- documented partial reproduction
# --------------------------------------------------------------------------------------


def test_correia_reference_case():
    """Correia et al. 2022 (Sensors 22:604) Sec. 3.1, their UE4/AirSim validation.

    FINDING: the paper's stated inputs and its stated answer are mutually inconsistent along
    ONE axis. Our chain reproduces, from their Table 1 parameters:

        * easting  = 8.50282 m   vs their 8.50283 m   -> 7e-6 m
        * the inverse solve puts their answer at image v = 1098.997 vs their v = 1099
          -> 0.003 px

    but northing comes out 5.10357 m against their 7.99841 m, which would require image
    u = 1308.7 rather than the stated u = 1095. This is not a convention we can adopt:
    forcing the northing to match requires an R_ENU_CAM with det = -1, i.e. a reflection,
    which no camera orientation can produce. Two exact agreements (7 micrometres, 3
    millipixels) alongside one gross disagreement points at a transcription error in the
    paper's pixel coordinate or principal point, not at our geometry.

    So this test asserts the two quantities that DO reproduce. It still guards the whole
    Euler chain, R_CAM_TO_GIMBAL, R_NED_TO_ENU, the lever-arm composition and the ray-plane
    solve against regression -- it simply does not pretend to verify a number we have shown
    to be unverifiable.
    """
    intr = Intrinsics(fx=3558.1395, fy=3558.1395, cx=1224.0, cy=1024.0, width=2448, height=2048)

    # Table 1: gimbal yaw -pi/2, pitch -pi/3, roll 0. UAS attitude identity because the
    # gimbal angles are world-referenced (composing both would double-count the rotation).
    R = euler_chain_R_enu_cam((-math.pi / 2.0, -math.pi / 3.0, 0.0))
    assert np.isclose(np.linalg.det(R), 1.0), "must stay a proper rotation"

    # T_C->ENU = T_NED->ENU + R_NED->ENU @ T_G->UAS  (other translations are zero)
    cam = np.array([31.72212, 6.55099, 42.44889]) + R_NED_TO_ENU @ np.array([0.3, 0.0, 0.2])
    assert np.allclose(cam, [31.72212, 6.85099, 42.24889])

    ground, _, _ = project_rays_to_plane(intr.rays([[1095.0, 1099.0]]), R, cam, 0.0)
    assert np.isclose(ground[0, 0], 8.50283, atol=1e-4), "easting reproduces"

    # inverse direction: their answer implies v = 1099 to 3 decimal places
    back = world_to_pixel([[8.50283, 7.99841, 0.0]], intr, R, cam)
    assert np.isclose(back[0, 1], 1099.0, atol=0.01), "their v reproduces"


def test_correia_chain_matches_hand_composition():
    """Guards euler_chain_R_enu_cam against the composition order in Correia Eq. 53."""
    ypr = (-math.pi / 2.0, -math.pi / 3.0, 0.0)
    expected = R_NED_TO_ENU @ np.eye(3) @ dcm_zyx(*ypr).T @ R_CAM_TO_GIMBAL
    assert np.allclose(euler_chain_R_enu_cam(ypr), expected)
