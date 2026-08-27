"""Target read-out tests, including the round trip that makes a fix trustworthy."""

import math

import numpy as np
import pytest

from uavmosaic.camera import Intrinsics
from uavmosaic.canvas import Canvas
from uavmosaic.coords import CanvasGeometry, GeodeticAnchor, euler_chain_R_enu_cam
from uavmosaic.georef import GroundPlane, compute_footprint, solve_ground_plane
from uavmosaic.targets import TargetResolver, bilinear_error_metres

GEOM = CanvasGeometry(e_min=-500, n_min=-500, e_max=500, n_max=500, gsd=0.25)
ANCHOR = GeodeticAnchor(lat_deg=33.6844, lon_deg=73.0479, alt_m=520.0)


def scene(dip_deg=90.0, yaw_deg=0.0, roll_deg=0.0, alt=300.0, hfov=55.0):
    intr = Intrinsics.from_hfov(hfov, 1280, 720)
    R = euler_chain_R_enu_cam(
        (math.radians(yaw_deg), math.radians(-dip_deg), math.radians(roll_deg))
    )
    cam = np.array([0.0, 0.0, alt])
    fp = compute_footprint(
        intr, R, cam, GroundPlane(0.0, "lrf_slant"), GEOM,
        max_incidence_deg=89.0, clamp_factor=1e9, allow_lower_half=False,
    )
    return intr, R, cam, fp


# --------------------------------------------------------------------------------------
# Exactness
# --------------------------------------------------------------------------------------


def test_source_pixel_to_world_and_back_is_exact():
    intr, R, cam, fp = scene(dip_deg=62.0, yaw_deg=37.0, roll_deg=9.0)
    res = TargetResolver(GEOM, ANCHOR)
    rng = np.random.default_rng(3)
    for _ in range(200):
        u, v = rng.uniform(0, intr.width), rng.uniform(0, intr.height)
        fix = res.from_source_px(u, v, intr, R, cam, fp, uav_id=2)
        back = res.to_source_px(*fix.enu, intr, R, cam)
        assert np.allclose(back, (u, v), atol=1e-7, rtol=0), (u, v, back)


def test_centre_pixel_of_a_nadir_frame_is_directly_below():
    intr, R, cam, fp = scene(dip_deg=90.0, alt=250.0)
    fix = TargetResolver(GEOM).from_source_px(intr.cx, intr.cy, intr, R, cam, fp)
    assert np.allclose(fix.enu[:2], (0.0, 0.0), atol=1e-9)
    assert fix.cos_theta == pytest.approx(1.0)
    assert fix.incidence_deg == pytest.approx(0.0, abs=1e-4)


def test_canvas_pixel_round_trip():
    res = TargetResolver(GEOM)
    rng = np.random.default_rng(4)
    for _ in range(200):
        e, n = rng.uniform(-500, 500), rng.uniform(-500, 500)
        x, y = res.enu_to_canvas_px(e, n)
        fix = res.from_canvas_px(x, y)
        assert np.allclose(fix.enu[:2], (e, n), atol=1e-9)


def test_source_and_canvas_paths_agree():
    """A detection located from the raw frame must land on the canvas pixel it was drawn to."""
    intr, R, cam, fp = scene(dip_deg=70.0, yaw_deg=-25.0)
    res = TargetResolver(GEOM)
    fix = res.from_source_px(900.0, 300.0, intr, R, cam, fp)
    via_canvas = res.from_canvas_px(*fix.canvas_px)
    assert np.allclose(via_canvas.enu[:2], fix.enu[:2], atol=1e-6)


# --------------------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------------------


def test_geodetic_output_round_trips_through_the_anchor():
    intr, R, cam, fp = scene(dip_deg=75.0)
    res = TargetResolver(GEOM, ANCHOR)
    fix = res.from_source_px(400.0, 500.0, intr, R, cam, fp)
    assert fix.lat_deg is not None
    back = ANCHOR.geodetic_to_enu(fix.lat_deg, fix.lon_deg, fix.alt_m)
    assert np.allclose(back, fix.enu, atol=1e-6)


def test_no_anchor_means_no_latlon_rather_than_a_wrong_one():
    intr, R, cam, fp = scene()
    fix = TargetResolver(GEOM).from_source_px(100.0, 100.0, intr, R, cam, fp)
    assert fix.lat_deg is None and fix.lon_deg is None
    assert fix.enu is not None, "local metric coordinates are always available"


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def test_canvas_fix_reports_owner_and_age():
    canvas = Canvas(GEOM)
    canvas.composite(
        3, (100, 100, 200, 200),
        np.full((100, 100, 3), 200, np.uint8), np.ones((100, 100), np.float32), t_now=10.0,
    )
    fix = TargetResolver(GEOM).from_canvas_px(150.0, 150.0, canvas, t_now=13.5)
    assert fix.uav_id == 3
    assert fix.age_s == pytest.approx(3.5)


def test_unwritten_canvas_pixel_has_no_owner():
    canvas = Canvas(GEOM)
    fix = TargetResolver(GEOM).from_canvas_px(10.0, 10.0, canvas, t_now=1.0)
    assert fix.uav_id is None


@pytest.mark.parametrize(
    "dip,tier,expect",
    [
        (88.0, "lrf_slant", "good"),
        (60.0, "lrf_slant", "medium"),
        (30.0, "lrf_slant", "low"),
        (88.0, "default", "low"),
        (88.0, "agl", "medium"),
    ],
)
def test_quality_label_reflects_geometry_and_plane_source(dip, tier, expect):
    intr, R, cam, _ = scene(dip_deg=dip)
    fp = compute_footprint(
        intr, R, cam, GroundPlane(0.0, tier), GEOM,
        max_incidence_deg=89.0, clamp_factor=1e9, allow_lower_half=False,
    )
    fix = TargetResolver(GEOM).from_source_px(intr.cx, intr.cy, intr, R, cam, fp)
    assert fix.quality.startswith(expect), fix.quality


def test_plane_tier_is_carried_through_from_the_lrf_cascade():
    intr, R, cam, _ = scene(dip_deg=80.0)
    plane = solve_ground_plane(cam, R, lrf_slant_m=None, agl_m=280.0)
    fp = compute_footprint(intr, R, cam, plane, GEOM, max_incidence_deg=89.0)
    fix = TargetResolver(GEOM).from_source_px(640.0, 360.0, intr, R, cam, fp)
    assert fix.plane_tier == "agl"
    assert fix.plane_z == pytest.approx(20.0)


# --------------------------------------------------------------------------------------
# The bilinear shortcut we deliberately do not use
# --------------------------------------------------------------------------------------


def test_bilinear_interpolation_is_accurate_at_nadir():
    """Near nadir the projection is nearly affine, so bilinear is fine -- which is why the
    shortcut looks harmless if you only ever test it on nadir imagery."""
    intr, R, cam, fp = scene(dip_deg=90.0)
    assert bilinear_error_metres(fp, intr, R, cam) < 0.5


@pytest.mark.parametrize("dip", [70.0, 55.0, 40.0])
def test_bilinear_interpolation_degrades_badly_when_tilted(dip):
    """Under a tilted gimbal, corner-bilinear interpolation is metres wrong in the middle of
    the frame. This is the concrete reason from_source_px projects the ray instead."""
    intr, R, cam, fp = scene(dip_deg=dip)
    err = bilinear_error_metres(fp, intr, R, cam)
    assert err > 2.0, f"expected a large bilinear error at {dip} deg dip, got {err:.2f} m"


def test_bilinear_error_grows_monotonically_with_tilt():
    errs = []
    for dip in (90.0, 75.0, 60.0, 45.0):
        intr, R, cam, fp = scene(dip_deg=dip)
        errs.append(bilinear_error_metres(fp, intr, R, cam))
    assert errs == sorted(errs), f"error should grow as the view tilts: {errs}"
