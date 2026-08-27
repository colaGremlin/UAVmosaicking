"""End-to-end: synthetic world -> render -> real UDP -> reassemble -> fuse -> compare.

This is the strongest statement the test suite makes. A known ground texture is rendered
from four independent camera poses, transmitted over genuine UDP sockets using the real
protocol carrying raw Unity-frame values, reassembled, decoded, georeferenced from telemetry
alone, and composited. The resulting mosaic is then compared **against the source texture**.

If any sign, axis, scale or ROI offset were wrong anywhere in that chain, the mosaic would
not line up with the world and the correlation check would fail.
"""

import time

import cv2
import numpy as np
import pytest

from uavmosaic.canvas import Canvas
from uavmosaic.config import AppConfig, EncoderConfig
from uavmosaic.coords import CanvasGeometry
from uavmosaic.frames import SENSOR_EO
from uavmosaic.fusion import FusionEngine
from uavmosaic.ingest import IngestGroup
from uavmosaic.synth import FlightPlan, make_world, render_view

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sim_sender import SimSender  # noqa: E402

# Small AOI keeps the test quick; the geometry is scale-free so nothing is lost.
AOI = CanvasGeometry(e_min=-400, n_min=-400, e_max=400, n_max=400, gsd=0.5)  # 1600 px
PORTS = {0: 5901, 1: 5902, 2: 5903, 3: 5904}


def make_cfg(**kw):
    base = dict(
        aoi=AOI,
        uav_ids=(0, 1, 2, 3),
        eo_ports=dict(PORTS),
        target_hz=10.0,
        max_frame_age_s=5.0,  # generous: the test drives ticks manually
        encoder=EncoderConfig(enabled=False),
    )
    base.update(kw)
    return AppConfig(**base)


@pytest.fixture
def pipeline():
    cfg = make_cfg()
    ingest = IngestGroup()
    for uav_id in cfg.uav_ids:
        ingest.add(uav_id, SENSOR_EO, PORTS[uav_id], host="127.0.0.1")
    ingest.start()
    canvas = Canvas(cfg.aoi)
    engine = FusionEngine(cfg, ingest, canvas, sensor_id=SENSOR_EO)
    try:
        yield cfg, ingest, engine, canvas
    finally:
        engine.close()
        ingest.stop()


def _drain(ingest, engine, expect, timeout=8.0):
    """Tick until ``expect`` frames have been fused or we give up."""
    deadline = time.monotonic() + timeout
    fused = 0
    while fused < expect and time.monotonic() < deadline:
        fused += engine.tick()
        time.sleep(0.01)
    return fused


def test_frames_traverse_real_sockets_and_reach_the_canvas(pipeline):
    cfg, ingest, engine, canvas = pipeline
    sender = SimSender(cfg, host="127.0.0.1", ports=dict(PORTS), jpeg_quality=85)
    try:
        for k in range(3):
            sender.send_tick(k * 0.1)
        fused = _drain(ingest, engine, expect=4)
    finally:
        sender.close()

    assert sender.frames_sent >= 4, "sender produced nothing"
    assert fused >= 4, f"only {fused} frames reached the canvas"
    assert canvas.coverage_fraction() > 0.02
    assert len(canvas.owner_counts()) >= 2, "several UAVs must be contributing"


def test_mosaic_reconstructs_the_source_world(pipeline):
    """THE test: does the fused canvas actually look like the ground it was rendered from?

    Compared only where the canvas has been written, and via normalised cross-correlation on
    a blurred grayscale version -- JPEG artefacts and resampling make exact equality
    meaningless, but a registration error of even a few metres destroys correlation.
    """
    cfg, ingest, engine, canvas = pipeline
    world = make_world(cfg.aoi, seed=11)
    sender = SimSender(cfg, host="127.0.0.1", ports=dict(PORTS), world=world, jpeg_quality=90)
    try:
        for k in range(8):
            sender.send_tick(k * 0.9)
            _drain(ingest, engine, expect=0, timeout=0.05)
        _drain(ingest, engine, expect=24, timeout=8.0)
    finally:
        sender.close()

    covered = canvas.weight > 0
    assert covered.mean() > 0.05, f"only {covered.mean() * 100:.1f}% covered"

    a = cv2.GaussianBlur(cv2.cvtColor(canvas.view(), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(world, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    x = a[covered].astype(np.float64)
    y = b[covered].astype(np.float64)
    ncc = float(np.corrcoef(x, y)[0, 1])

    assert ncc > 0.90, f"mosaic does not match the source world (ncc={ncc:.3f})"


def test_a_misaligned_mosaic_would_fail_that_check():
    """Guards the guard: shifting the mosaic 12 m must destroy the correlation.

    Without this, a bug that made the comparison trivially true (e.g. comparing the world
    with itself) would pass unnoticed.
    """
    world = make_world(AOI, seed=11)
    shift = int(12.0 / AOI.gsd)
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    shifted = cv2.warpAffine(world, M, (AOI.width, AOI.height))

    a = cv2.GaussianBlur(cv2.cvtColor(world, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float64)
    b = cv2.GaussianBlur(cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float64)
    roi = (slice(shift * 2, -shift * 2), slice(shift * 2, -shift * 2))
    ncc = float(np.corrcoef(a[roi].ravel(), b[roi].ravel())[0, 1])
    assert ncc < 0.90, f"a 12 m offset should have broken correlation, got {ncc:.3f}"


def test_stale_frames_are_not_refused_but_old_ones_are(pipeline):
    cfg, ingest, engine, canvas = pipeline
    strict = FusionEngine(
        make_cfg(max_frame_age_s=0.0), ingest, Canvas(cfg.aoi), sensor_id=SENSOR_EO
    )
    sender = SimSender(cfg, host="127.0.0.1", ports=dict(PORTS))
    try:
        sender.send_tick(0.0)
        time.sleep(0.4)
        for _ in range(20):
            strict.tick()
            time.sleep(0.01)
        assert strict.canvas.coverage_fraction() == 0.0, "age gate must reject old frames"
    finally:
        strict.close()
        sender.close()


def test_render_view_is_the_inverse_of_the_fusion_warp():
    """A rendered view, warped forward by the pipeline, must land back on the source pixels.

    This closes the loop inside the simulator itself, independent of sockets and threads.
    """
    from uavmosaic.georef import GroundPlane, compute_footprint

    world = make_world(AOI, seed=3)
    plan = FlightPlan(AOI, n_uavs=4)
    st = plan.state(1, 2.0)

    view = render_view(world, AOI, st.intr, st.R_enu_cam, st.cam_enu)
    assert view is not None

    fp = compute_footprint(
        st.intr, st.R_enu_cam, st.cam_enu, GroundPlane(0.0, "default"), AOI,
        max_incidence_deg=89.0, clamp_factor=1e9, allow_lower_half=False,
    )
    roi = fp.canvas_roi(AOI.shape)
    H = fp.homography_to_roi(roi)
    x0, y0, x1, y1 = roi
    back = cv2.warpPerspective(view, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)

    ref = world[y0:y1, x0:x1]
    mask = back.any(axis=2)
    assert mask.mean() > 0.2

    a = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)
    ncc = float(np.corrcoef(a, b)[0, 1])
    assert ncc > 0.97, f"render/warp round trip is not self-consistent (ncc={ncc:.3f})"
