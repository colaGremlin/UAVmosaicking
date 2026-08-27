"""Gating tests for the LH->RH conversion. Nothing else gets built until these pass.

The blueprint calls this out as the single most likely place for a silent sign error, so
these tests check hand-derived golden cases and structural invariants rather than
round-tripping the implementation against itself.
"""

import math

import numpy as np
import pytest

from uavmosaic.coords import (
    F_UNITYCAM_CVCAM,
    R_CAM_TO_GIMBAL,
    R_NED_TO_ENU,
    S_ENU_UNITY,
    CanvasGeometry,
    GeodeticAnchor,
    dcm_zyx,
    euler_chain_R_enu_cam,
    quat_to_matrix,
    unity_pos_to_enu,
    unity_quat_to_R_enu_cam,
)


# --------------------------------------------------------------------------------------
# Structural invariants of the frame bridges
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("M,name", [(S_ENU_UNITY, "S"), (F_UNITYCAM_CVCAM, "F")])
def test_bridges_are_involutions_that_flip_handedness(M, name):
    assert np.allclose(M @ M, np.eye(3)), f"{name} must be self-inverse"
    assert np.isclose(np.linalg.det(M), -1.0), f"{name} must flip handedness (det = -1)"


def test_axis_mapping_is_east_north_up():
    """S maps Unity (x=right, y=up, z=fwd) to ENU: E=x, N=z, U=y."""
    assert np.allclose(S_ENU_UNITY @ np.array([1, 0, 0]), [1, 0, 0])  # x -> East
    assert np.allclose(S_ENU_UNITY @ np.array([0, 1, 0]), [0, 0, 1])  # y -> Up
    assert np.allclose(S_ENU_UNITY @ np.array([0, 0, 1]), [0, 1, 0])  # z -> North


def test_helper_matrices_are_proper_rotations():
    """Unlike S and F, these two stay *within* right-handed space, so det must be +1.

    NED and ENU are both right-handed -- NED->ENU only reorders axes and negates Down, which
    is a 180 deg rotation about the NE bisector, not a reflection. If this ever came out -1
    it would mean the frame definitions had drifted.
    """
    assert np.isclose(np.linalg.det(R_CAM_TO_GIMBAL), 1.0)  # axis permutation, Correia Eq. 39
    assert np.isclose(np.linalg.det(R_NED_TO_ENU), 1.0)
    assert np.allclose(R_NED_TO_ENU @ R_NED_TO_ENU, np.eye(3))
    # (n, e, d) -> (e, n, -d)
    assert np.allclose(R_NED_TO_ENU @ np.array([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])


# --------------------------------------------------------------------------------------
# Quaternion path
# --------------------------------------------------------------------------------------


def test_identity_quaternion_gives_identity():
    assert np.allclose(quat_to_matrix(0, 0, 0, 1), np.eye(3))


def test_quat_to_matrix_is_orthonormal_for_random_quaternions():
    rng = np.random.default_rng(20260827)
    for _ in range(500):
        q = rng.normal(size=4)
        if np.linalg.norm(q) < 1e-6:
            continue
        R = quat_to_matrix(*q)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_quat_matches_unity_rotation_operator():
    """Unity's Quaternion*Vector3 is the standard RH formula applied in its LH frame.

    Reproduces Unity's operator explicitly and asserts our matrix agrees, which is the
    assumption the whole quaternion path rests on.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        v = rng.normal(size=3)
        # Unity Quaternion.cs, verbatim
        x2, y2, z2 = x * 2.0, y * 2.0, z * 2.0
        xx, yy, zz = x * x2, y * y2, z * z2
        xy, xz, yz = x * y2, x * z2, y * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        unity = np.array(
            [
                (1 - (yy + zz)) * v[0] + (xy - wz) * v[1] + (xz + wy) * v[2],
                (xy + wz) * v[0] + (1 - (xx + zz)) * v[1] + (yz - wx) * v[2],
                (xz - wy) * v[0] + (yz + wx) * v[1] + (1 - (xx + yy)) * v[2],
            ]
        )
        assert np.allclose(quat_to_matrix(x, y, z, w) @ v, unity, atol=1e-12)


def test_R_enu_cam_is_always_a_proper_rotation():
    """det = (-1)(+1)(-1) = +1. If this ever fails the mosaic is mirrored."""
    rng = np.random.default_rng(99)
    for _ in range(500):
        q = rng.normal(size=4)
        if np.linalg.norm(q) < 1e-6:
            continue
        R = unity_quat_to_R_enu_cam(*q)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_nadir_north_up_golden_case():
    """THE hand-derived case from the Phase-1 blueprint.

    Camera looks straight down; image-up points to Unity +Z (North). Then:
        image-right  -> East
        image-down   -> South   (canvas y grows southward -- correct for an image)
        boresight    -> Down
    """
    # Unity camera basis expressed in Unity world: X->+x, Y->+z, Z->-y
    R_u = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    R = S_ENU_UNITY @ R_u @ F_UNITYCAM_CVCAM
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0])), R

    assert np.allclose(R @ [1, 0, 0], [1, 0, 0])  # image right  -> East
    assert np.allclose(R @ [0, 1, 0], [0, -1, 0])  # image down   -> South
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1])  # optical axis -> Down


def test_nadir_via_quaternion_agrees_with_golden_matrix():
    """Unity pitch of **+90 deg** about X is nose-down -- this is the nadir camera.

    Sign matters and is easy to get backwards: -90 deg points the camera at the sky and
    yields R = I, which looks deceptively reasonable. Guarded explicitly below.
    """
    s = math.sin(math.radians(90) / 2.0)
    c = math.cos(math.radians(90) / 2.0)
    R = unity_quat_to_R_enu_cam(s, 0.0, 0.0, c)  # +90 deg about Unity X
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-12), R
    assert (R @ np.array([0.0, 0.0, 1.0]))[2] < 0, "boresight must point down"

    # the wrong sign points at the sky; assert it is clearly distinguishable
    R_up = unity_quat_to_R_enu_cam(-s, 0.0, 0.0, c)
    assert (R_up @ np.array([0.0, 0.0, 1.0]))[2] > 0, "-90 deg must point up"


def test_yaw_rotates_the_image_in_the_ground_plane():
    """A 90 deg Unity yaw (about +Y = up) must swing image-right from East to South."""
    s = math.sin(math.radians(90) / 2.0)
    c = math.cos(math.radians(90) / 2.0)
    R_yaw = quat_to_matrix(0.0, s, 0.0, c)
    R_nadir = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    R = S_ENU_UNITY @ (R_yaw @ R_nadir) @ F_UNITYCAM_CVCAM
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-12)  # still nadir
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)  # right -> South


def test_position_conversion():
    assert np.allclose(unity_pos_to_enu([3.0, 50.0, 7.0]), [3.0, 7.0, 50.0])
    assert np.allclose(unity_pos_to_enu([3.0, 50.0, 7.0], origin_enu=[1.0, 2.0, 3.0]), [2.0, 5.0, 47.0])


# --------------------------------------------------------------------------------------
# Euler path
# --------------------------------------------------------------------------------------


def test_dcm_zyx_identity_and_orthonormality():
    assert np.allclose(dcm_zyx(0, 0, 0), np.eye(3))
    rng = np.random.default_rng(3)
    for _ in range(200):
        y, p, r = rng.uniform(-math.pi, math.pi, 3)
        p = np.clip(p, -math.pi / 2 + 1e-3, math.pi / 2 - 1e-3)
        R = dcm_zyx(y, p, r)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_dcm_zyx_known_yaw():
    """Yaw 90 deg, NED->body: body-x points East, so world North lands on body -y."""
    R = dcm_zyx(math.radians(90), 0, 0)
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)  # North -> -y_body
    assert np.allclose(R @ [0, 1, 0], [1, 0, 0], atol=1e-12)  # East  -> +x_body


def test_euler_chain_nadir_matches_quaternion_path():
    """Both paths must agree. Gimbal pitch -90 deg (nadir), yaw 0 (north), roll 0."""
    R_euler = euler_chain_R_enu_cam((0.0, -math.pi / 2, 0.0))
    assert np.isclose(np.linalg.det(R_euler), 1.0, atol=1e-12)
    assert np.allclose(R_euler, np.diag([1.0, -1.0, -1.0]), atol=1e-12), R_euler


def test_euler_chain_boresight_follows_gimbal_yaw():
    """Boresight azimuth must track the commanded gimbal yaw, measured from North."""
    for yaw_deg, expect_en in [
        (0.0, (0.0, 1.0)),  # North
        (90.0, (1.0, 0.0)),  # East
        (180.0, (0.0, -1.0)),  # South
        (-90.0, (-1.0, 0.0)),  # West
    ]:
        R = euler_chain_R_enu_cam((math.radians(yaw_deg), math.radians(-45.0), 0.0))
        b = R @ np.array([0.0, 0.0, 1.0])
        horiz = b[:2] / np.linalg.norm(b[:2])
        assert np.allclose(horiz, expect_en, atol=1e-9), (yaw_deg, horiz)
        assert b[2] < 0, "45 deg down-tilt must descend"


# --------------------------------------------------------------------------------------
# Canvas geometry
# --------------------------------------------------------------------------------------


def test_canvas_shape_and_corner_mapping():
    g = CanvasGeometry(e_min=0, n_min=0, e_max=2000, n_max=1000, gsd=0.5)
    assert (g.width, g.height) == (4000, 2000)
    assert g.shape == (2000, 4000)

    # NW corner of the AOI is canvas (0, 0); SE corner is (width, height)
    assert np.allclose(g.enu_to_px(0.0, 1000.0), (0.0, 0.0))
    assert np.allclose(g.enu_to_px(2000.0, 0.0), (4000.0, 2000.0))


def test_canvas_north_is_up():
    g = CanvasGeometry(e_min=0, n_min=0, e_max=100, n_max=100, gsd=1.0)
    _, y_north = g.enu_to_px(50.0, 90.0)
    _, y_south = g.enu_to_px(50.0, 10.0)
    assert y_north < y_south, "increasing north must decrease canvas y"


def test_canvas_roundtrip():
    g = CanvasGeometry(e_min=-500, n_min=100, e_max=1500, n_max=2100, gsd=0.25)
    rng = np.random.default_rng(11)
    e = rng.uniform(-500, 1500, 1000)
    n = rng.uniform(100, 2100, 1000)
    x, y = g.enu_to_px(e, n)
    e2, n2 = g.px_to_enu(x, y)
    assert np.allclose(e, e2, atol=1e-9)
    assert np.allclose(n, n2, atol=1e-9)


def test_canvas_matrix_matches_scalar_path():
    g = CanvasGeometry(e_min=-500, n_min=100, e_max=1500, n_max=2100, gsd=0.25)
    M = g.matrix()
    rng = np.random.default_rng(12)
    for _ in range(100):
        e, n = rng.uniform(-500, 1500), rng.uniform(100, 2100)
        got = M @ np.array([e, n, 1.0])
        assert np.allclose(got[:2] / got[2], g.enu_to_px(e, n), atol=1e-9)


def test_degenerate_aoi_rejected():
    with pytest.raises(ValueError):
        CanvasGeometry(e_min=0, n_min=0, e_max=0, n_max=100, gsd=1.0)
    with pytest.raises(ValueError):
        CanvasGeometry(e_min=0, n_min=0, e_max=100, n_max=100, gsd=0.0)


# --------------------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------------------


def test_geodetic_roundtrip_is_exact():
    anchor = GeodeticAnchor(lat_deg=33.6844, lon_deg=73.0479, alt_m=520.0)  # Islamabad
    rng = np.random.default_rng(5)
    for _ in range(200):
        enu = rng.uniform(-20000, 20000, 3)
        lat, lon, alt = anchor.enu_to_geodetic(*enu)
        back = anchor.geodetic_to_enu(lat, lon, alt)
        assert np.allclose(back, enu, atol=1e-6), (enu, back)


def test_anchor_origin_maps_to_itself():
    anchor = GeodeticAnchor(lat_deg=-12.5, lon_deg=130.9, alt_m=15.0)
    lat, lon, alt = anchor.enu_to_geodetic(0.0, 0.0, 0.0)
    assert np.isclose(lat, -12.5, atol=1e-12)
    assert np.isclose(lon, 130.9, atol=1e-12)
    assert np.isclose(alt, 15.0, atol=1e-6)


def test_east_and_north_move_the_right_way():
    anchor = GeodeticAnchor(lat_deg=45.0, lon_deg=9.0, alt_m=0.0)
    lat_e, lon_e, _ = anchor.enu_to_geodetic(1000.0, 0.0, 0.0)
    lat_n, lon_n, _ = anchor.enu_to_geodetic(0.0, 1000.0, 0.0)
    assert lon_e > 9.0 and abs(lat_e - 45.0) < 1e-3, "east must increase longitude"
    assert lat_n > 45.0 and abs(lon_n - 9.0) < 1e-9, "north must increase latitude"
    # 1 km north is ~0.009 deg of latitude
    assert 0.0085 < (lat_n - 45.0) < 0.0095
