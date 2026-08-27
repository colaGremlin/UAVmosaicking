"""Canvas compositing and weight-map tests.

The rule under test is the one that differs from every reference implementation: a UAV
always overwrites its own pixels, and max-weight arbitrates only between different UAVs.
``test_same_owner_always_overwrites`` and ``test_mosaic_does_not_freeze_under_a_moving_uav``
are the ones that matter -- they encode the failure the clause exists to prevent.
"""

import math

import numpy as np
import pytest

from uavmosaic.camera import Intrinsics
from uavmosaic.canvas import UNOWNED, Canvas
from uavmosaic.coords import CanvasGeometry, euler_chain_R_enu_cam
from uavmosaic.weights import (
    WEIGHT_FLOOR,
    frame_weight_map,
    gsd_weight,
    incidence_weight_map,
    radial_weight_map,
)

GEOM = CanvasGeometry(e_min=0, n_min=0, e_max=100, n_max=100, gsd=1.0)  # 100x100 px


def fresh():
    return Canvas(GEOM)


def block(v, shape, ch=3):
    return np.full((*shape, ch), v, dtype=np.uint8)


# --------------------------------------------------------------------------------------
# Canvas basics
# --------------------------------------------------------------------------------------


def test_starts_empty_and_unowned():
    c = fresh()
    assert c.shape == (100, 100)
    assert not c.color.any()
    assert not c.weight.any()
    assert (c.owner == UNOWNED).all()
    assert c.coverage_fraction() == 0.0
    assert c.owner_counts() == {}


def test_first_write_always_lands():
    """Any valid pixel must beat empty canvas, even at the weight floor."""
    c = fresh()
    roi = (10, 10, 30, 30)
    w = np.full((20, 20), WEIGHT_FLOOR, dtype=np.float32)
    res = c.composite(1, roi, block(200, (20, 20)), w, t_now=1.0)
    assert res.pixels_written == 400
    assert (c.color[10:30, 10:30] == 200).all()
    assert (c.owner[10:30, 10:30] == 1).all()
    assert c.coverage_fraction() == pytest.approx(400 / 10_000)


def test_zero_weight_is_the_validity_mask():
    """Warp fill is zero-weight and must never be written."""
    c = fresh()
    w = np.zeros((20, 20), dtype=np.float32)
    w[:10] = 1.0  # only the top half is real data
    res = c.composite(1, (0, 0, 20, 20), block(255, (20, 20)), w, t_now=1.0)
    assert res.pixels_written == 200
    assert (c.color[:10, :20] == 255).all()
    assert not c.color[10:20, :20].any()
    assert (c.owner[10:20, :20] == UNOWNED).all()


# --------------------------------------------------------------------------------------
# The arbitration rule
# --------------------------------------------------------------------------------------


def test_higher_weight_wins_between_different_uavs():
    c = fresh()
    roi = (0, 0, 10, 10)
    c.composite(1, roi, block(50, (10, 10)), np.full((10, 10), 0.5, np.float32), 1.0)
    c.composite(2, roi, block(90, (10, 10)), np.full((10, 10), 0.9, np.float32), 2.0)
    assert (c.color[:10, :10] == 90).all()
    assert (c.owner[:10, :10] == 2).all()


def test_lower_weight_loses_between_different_uavs():
    c = fresh()
    roi = (0, 0, 10, 10)
    c.composite(1, roi, block(50, (10, 10)), np.full((10, 10), 0.9, np.float32), 1.0)
    c.composite(2, roi, block(90, (10, 10)), np.full((10, 10), 0.2, np.float32), 2.0)
    assert (c.color[:10, :10] == 50).all(), "the better view must keep the pixel"
    assert (c.owner[:10, :10] == 1).all()


def test_same_owner_always_overwrites_even_at_lower_weight():
    """THE clause. A UAV's fresh frame must never be blocked by its own older one."""
    c = fresh()
    roi = (0, 0, 10, 10)
    c.composite(1, roi, block(50, (10, 10)), np.full((10, 10), 0.9, np.float32), 1.0)
    c.composite(1, roi, block(90, (10, 10)), np.full((10, 10), 0.1, np.float32), 2.0)
    assert (c.color[:10, :10] == 90).all(), "same UAV must overwrite itself"
    assert c.weight[:10, :10] == pytest.approx(0.1), "weight follows the new pixel"
    assert (c.stamp[:10, :10] == 2.0).all()


def test_mosaic_does_not_freeze_under_a_moving_uav():
    """Simulates the failure the same-owner clause prevents.

    A UAV banks progressively, so each new frame has a slightly worse weight than the last.
    Without the clause the imagery beneath it would stop updating after frame 1.
    """
    c = fresh()
    roi = (20, 20, 60, 60)
    n = (40, 40)
    for i in range(10):
        w = np.full(n, 0.9 - 0.08 * i, dtype=np.float32)  # monotonically worse
        c.composite(3, roi, block(10 * i + 10, n), w, t_now=float(i))
    assert (c.color[20:60, 20:60] == 100).all(), "must show the LATEST frame"
    assert (c.stamp[20:60, 20:60] == 9.0).all()


def test_a_second_uav_can_still_take_over_from_a_stale_owner():
    c = fresh()
    roi = (0, 0, 10, 10)
    c.composite(1, roi, block(50, (10, 10)), np.full((10, 10), 0.2, np.float32), 1.0)
    c.composite(2, roi, block(90, (10, 10)), np.full((10, 10), 0.5, np.float32), 2.0)
    assert (c.owner[:10, :10] == 2).all()
    # ...and 1 can take it back when its view improves
    c.composite(1, roi, block(70, (10, 10)), np.full((10, 10), 0.8, np.float32), 3.0)
    assert (c.owner[:10, :10] == 1).all()
    assert (c.color[:10, :10] == 70).all()


def test_partial_overlap_splits_ownership_by_weight():
    c = fresh()
    left = np.zeros((10, 10), np.float32)
    left[:, :5] = 0.9
    right = np.zeros((10, 10), np.float32)
    right[:, 3:] = 0.5
    c.composite(1, (0, 0, 10, 10), block(11, (10, 10)), left, 1.0)
    c.composite(2, (0, 0, 10, 10), block(22, (10, 10)), right, 2.0)
    assert (c.owner[:10, :5] == 1).all(), "UAV1 holds where it is stronger"
    assert (c.owner[:10, 5:10] == 2).all(), "UAV2 takes the rest"
    assert c.owner_counts(stride=1) == {1: 50, 2: 50}


def test_stats_subsampling_is_proportionate_but_not_exact():
    """HUD stats sample every 4th pixel. Proportions must survive; exact counts need
    stride=1. Documented because a caller expecting absolute pixel counts would be wrong."""
    c = fresh()
    c.composite(1, (0, 0, 100, 40), block(9, (40, 100)), np.ones((40, 100), np.float32), 1.0)
    c.composite(2, (0, 40, 100, 60), block(9, (20, 100)), np.ones((20, 100), np.float32), 1.0)
    exact = c.owner_counts(stride=1)
    assert exact == {1: 4000, 2: 2000}
    sampled = c.owner_counts()
    assert sampled[1] / sampled[2] == pytest.approx(2.0, rel=0.15)
    assert c.coverage_fraction(stride=1) == pytest.approx(0.6)
    assert c.coverage_fraction() == pytest.approx(0.6, rel=0.1)


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_shape_mismatch_is_rejected():
    c = fresh()
    with pytest.raises(ValueError, match="weight"):
        c.composite(1, (0, 0, 10, 10), block(1, (10, 10)), np.ones((9, 9), np.float32), 0.0)
    with pytest.raises(ValueError, match="color"):
        c.composite(1, (0, 0, 10, 10), block(1, (9, 9)), np.ones((10, 10), np.float32), 0.0)


def test_reserved_owner_id_is_rejected():
    c = fresh()
    with pytest.raises(ValueError, match="reserved"):
        c.composite(UNOWNED, (0, 0, 4, 4), block(1, (4, 4)), np.ones((4, 4), np.float32), 0.0)


def test_clear_resets_everything():
    c = fresh()
    c.composite(1, (0, 0, 10, 10), block(9, (10, 10)), np.ones((10, 10), np.float32), 1.0)
    c.clear()
    assert not c.weight.any() and (c.owner == UNOWNED).all() and not c.color.any()


def test_staleness_is_infinite_where_nothing_was_written():
    c = fresh()
    c.composite(1, (0, 0, 10, 10), block(9, (10, 10)), np.ones((10, 10), np.float32), 5.0)
    age = c.staleness_seconds(t_now=8.0)
    assert age[:10, :10] == pytest.approx(3.0)
    assert np.isinf(age[50, 50])


# --------------------------------------------------------------------------------------
# Weight maps
# --------------------------------------------------------------------------------------


def test_radial_peaks_at_centre_and_falls_to_the_floor():
    w = radial_weight_map(101, 51)  # odd dims -> an exact centre pixel exists
    assert w.shape == (51, 101)
    assert w[25, 50] == pytest.approx(1.0, abs=1e-6)
    assert w[0, 0] == pytest.approx(WEIGHT_FLOOR)
    assert w.min() >= WEIGHT_FLOOR


def test_radial_follows_the_principal_point_not_the_array_centre():
    """Vignetting is radial about the optical axis. With an offset principal point the peak
    must move with it -- otherwise a real camera is weighted about the wrong origin."""
    w = radial_weight_map(101, 51, cx=20.0, cy=10.0)
    assert np.unravel_index(np.argmax(w), w.shape) == (10, 20)
    assert w[10, 20] == pytest.approx(1.0, abs=1e-6)
    assert w[25, 50] < w[10, 20]


def test_radial_is_monotonic_outward():
    w = radial_weight_map(64, 64)
    mid = w[32, 32:]
    assert np.all(np.diff(mid) <= 1e-7), "must decrease monotonically toward the edge"


def test_radial_is_cached_and_read_only():
    a = radial_weight_map(128, 96)
    b = radial_weight_map(128, 96)
    assert a is b, "must be cached -- it is rebuilt for every frame otherwise"
    with pytest.raises(ValueError):
        a[0, 0] = 1.0


def test_incidence_is_unity_at_nadir():
    intr = Intrinsics.from_hfov(60.0, 320, 240)
    R = np.diag([1.0, -1.0, -1.0])
    w = incidence_weight_map(intr, R)
    assert w.shape == (240, 320)
    assert w[120, 160] == pytest.approx(1.0, abs=1e-4), "centre ray is exactly nadir"
    assert w[0, 0] < w[120, 160], "corners look further off-nadir than the centre"


@pytest.mark.parametrize("dip", [30.0, 45.0, 70.0])
def test_incidence_centre_is_sin_dip_squared(dip):
    """A gimbal dipped `dip` below the HORIZON is (90-dip) off NADIR, so the centre weight
    is sin(dip)^2. Tested at 30 and 70 as well as 45, because at 45 sin == cos and the
    convention would pass even if it were backwards."""
    intr = Intrinsics.from_hfov(60.0, 320, 240)
    R = euler_chain_R_enu_cam((0.0, math.radians(-dip), 0.0))
    w = incidence_weight_map(intr, R, power=2.0)
    assert w[120, 160] == pytest.approx(math.sin(math.radians(dip)) ** 2, abs=1e-5)


def test_incidence_gradient_points_toward_nadir():
    """Tilted north: the bottom of the image is nearer nadir, so it must weigh more."""
    intr = Intrinsics.from_hfov(60.0, 320, 240)
    R = euler_chain_R_enu_cam((0.0, math.radians(-50.0), 0.0))
    w = incidence_weight_map(intr, R)
    assert w[200, 160] > w[40, 160]


def test_incidence_matches_a_naive_per_pixel_reference():
    """The cached/fused implementation must equal the textbook form to float32 noise.

    The cached inv-norm trick is an optimisation, not an approximation -- if it ever drifts
    from the naive definition, that is a bug, not a tolerance to widen.
    """
    intr = Intrinsics.from_hfov(70.0, 640, 480)
    R = euler_chain_R_enu_cam((math.radians(30.0), math.radians(-55.0), math.radians(10.0)))

    u, v = np.meshgrid(
        np.arange(intr.width, dtype=np.float64), np.arange(intr.height, dtype=np.float64)
    )
    d = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=-1)
    dw = d @ np.asarray(R).T
    naive = np.clip(-dw[..., 2] / np.linalg.norm(d, axis=-1), 0.0, 1.0) ** 2
    naive = np.maximum(naive, WEIGHT_FLOOR)

    got = incidence_weight_map(intr, R)
    assert np.abs(got - naive).max() < 1e-6, np.abs(got - naive).max()


def test_gsd_weight_favours_the_sharper_source():
    assert gsd_weight(0.25, 0.5) == pytest.approx(2.0)  # frame out-resolves the canvas
    assert gsd_weight(1.0, 0.5) == pytest.approx(0.5)  # coarser than the canvas
    assert gsd_weight(0.0, 0.5) == WEIGHT_FLOOR  # degenerate input cannot win


def test_frame_weight_is_exactly_the_product_of_its_terms():
    intr = Intrinsics.from_hfov(60.0, 320, 240)
    R = np.diag([1.0, -1.0, -1.0])
    w = frame_weight_map(intr, R, gsd_frame_m_per_px=0.25, gsd_ref_m_per_px=0.5)

    assert w.dtype == np.float32
    assert w.shape == (240, 320)
    assert w.min() >= WEIGHT_FLOOR
    assert np.isfinite(w).all()

    expect = (
        radial_weight_map(intr.width, intr.height, intr.cx, intr.cy)
        * incidence_weight_map(intr, R)
        * gsd_weight(0.25, 0.5)
    )
    assert np.abs(w - expect).max() < 1e-6


def test_frame_weight_peaks_near_the_optical_axis_scaled_by_gsd():
    """Nadir + gsd 2x means the peak sits at ~2.0. It is only *approached*, not hit, on an
    even-sized sensor: with cx = 160.0 the optical axis falls between pixel columns, so the
    best any pixel can do is (1 - 0.5/r_max)^2 * 2. Asserting 2.0 exactly would be wrong."""
    intr = Intrinsics.from_hfov(60.0, 320, 240)
    w = frame_weight_map(intr, np.diag([1.0, -1.0, -1.0]), 0.25, 0.5)
    assert 1.98 < w.max() <= 2.0
