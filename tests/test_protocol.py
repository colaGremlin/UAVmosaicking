"""Wire-protocol tests: round-trip, fragmentation, loss, reorder, duplication, hostility.

The assembler is the one component that faces the network directly, so it is tested against
a deliberately unpleasant link: reordering, loss, duplicates, supersession, truncation and
outright garbage. None of these may raise; all must degrade to a dropped frame.
"""

import struct

import numpy as np
import pytest

from uavmosaic.frames import SENSOR_EO, SENSOR_IR, Telemetry
from uavmosaic.protocol import (
    DEFAULT_MTU,
    FLAG_AGL_VALID,
    FLAG_LRF_VALID,
    HEADER_SIZE,
    MAGIC,
    TELEM_SIZE,
    FrameAssembler,
    ProtocolError,
    build_packets,
    max_payload_per_fragment,
    parse_packet,
)


def telem(**kw):
    base = dict(
        pos_unity=(12.5, 300.25, -47.75),
        quat_world_cam=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
        img_w=1280,
        img_h=720,
        fx=1108.5125,
        fy=1108.5125,
        cx=640.0,
        cy=360.0,
        lrf_slant_m=305.5,
        agl_m=300.0,
        hfov_deg=60.0,
        zoom=1.0,
    )
    base.update(kw)
    return Telemetry(**base)


def jpeg_of(n, seed=0):
    return bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------


def test_header_and_telemetry_sizes_are_frozen():
    """These are wire constants -- the Unity C# sender hardcodes them."""
    assert HEADER_SIZE == 36
    assert TELEM_SIZE == 64
    assert max_payload_per_fragment(DEFAULT_MTU) == 1300


def test_magic_is_the_ascii_tag():
    assert struct.pack("<I", MAGIC) == b"UAV1"


def test_every_fragment_carries_full_framing():
    pkts = build_packets(0, SENSOR_EO, 1, 1234, telem(), jpeg_of(9000))
    assert len(pkts) > 1
    for p in pkts:
        assert len(p) <= DEFAULT_MTU
        h, t, _ = parse_packet(p)
        assert t is not None, "telemetry must ride EVERY fragment, not just the first"
        assert h.frag_count == len(pkts)


# --------------------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1299, 1300, 1301, 60_000, 250_000])
def test_roundtrip_preserves_payload_exactly(size):
    payload = jpeg_of(size, seed=size)
    t = telem()
    pkts = build_packets(2, SENSOR_EO, 77, 99_887_766, t, payload)
    asm = FrameAssembler()
    out = [asm.push(p) for p in pkts]
    done = [f for f in out if f is not None]
    assert len(done) == 1, "exactly one completion, on the last fragment"
    f = done[0]
    assert f.jpeg == payload
    assert (f.uav_id, f.sensor_id, f.frame_id, f.t_capture_us) == (2, SENSOR_EO, 77, 99_887_766)


def test_roundtrip_preserves_telemetry_within_float32():
    t = telem()
    pkts = build_packets(1, SENSOR_IR, 5, 42, t, jpeg_of(500))
    got = FrameAssembler().push(pkts[0]).telemetry
    assert np.allclose(got.pos_unity, t.pos_unity, rtol=1e-6)
    assert np.allclose(got.quat_world_cam, t.quat_world_cam, rtol=1e-6)
    assert (got.img_w, got.img_h) == (t.img_w, t.img_h)
    assert np.isclose(got.fx, t.fx, rtol=1e-6)
    assert np.isclose(got.lrf_slant_m, t.lrf_slant_m, rtol=1e-6)
    assert np.isclose(got.agl_m, t.agl_m, rtol=1e-6)


def test_zero_length_frame_still_delivers_a_pose():
    """The pose is worth delivering even when the image is not."""
    pkts = build_packets(0, SENSOR_EO, 1, 1, telem(), b"")
    assert len(pkts) == 1
    f = FrameAssembler().push(pkts[0])
    assert f is not None and f.jpeg == b""


# --------------------------------------------------------------------------------------
# Optional-field flags
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lrf,agl", [(None, None), (250.0, None), (None, 300.0), (250.0, 300.0)]
)
def test_invalid_sensor_readings_survive_as_none(lrf, agl):
    t = telem(lrf_slant_m=lrf, agl_m=agl)
    pkts = build_packets(0, SENSOR_EO, 1, 1, t, jpeg_of(100))
    h, got, _ = parse_packet(pkts[0])
    assert (got.lrf_slant_m is None) == (lrf is None)
    assert (got.agl_m is None) == (agl is None)
    assert bool(h.flags & FLAG_LRF_VALID) == (lrf is not None)
    assert bool(h.flags & FLAG_AGL_VALID) == (agl is not None)


# --------------------------------------------------------------------------------------
# Hostile link
# --------------------------------------------------------------------------------------


def test_reordered_fragments_reassemble():
    payload = jpeg_of(40_000, seed=3)
    pkts = build_packets(0, SENSOR_EO, 1, 1, telem(), payload)
    shuffled = list(pkts)
    np.random.default_rng(7).shuffle(shuffled)
    asm = FrameAssembler()
    done = [f for f in (asm.push(p) for p in shuffled) if f is not None]
    assert len(done) == 1 and done[0].jpeg == payload


def test_a_single_lost_fragment_kills_the_whole_frame():
    """A partial JPEG must never be handed on -- it decodes to plausible garbage."""
    pkts = build_packets(0, SENSOR_EO, 1, 1, telem(), jpeg_of(40_000, seed=4))
    asm = FrameAssembler()
    for p in pkts[:-1]:
        assert asm.push(p) is None
    assert asm.stats.frames_completed == 0
    assert asm.inflight == 1


def test_duplicates_are_ignored_not_double_counted():
    payload = jpeg_of(5000, seed=5)
    pkts = build_packets(0, SENSOR_EO, 1, 1, telem(), payload)
    asm = FrameAssembler()
    done = [f for f in (asm.push(p) for p in pkts + pkts) if f is not None]
    assert len(done) == 1 and done[0].jpeg == payload
    assert asm.stats.duplicates >= len(pkts) - 1


def test_a_newer_frame_supersedes_an_incomplete_older_one():
    old = build_packets(0, SENSOR_EO, 1, 100, telem(), jpeg_of(40_000, seed=6))
    new = build_packets(0, SENSOR_EO, 2, 200, telem(), jpeg_of(40_000, seed=7))
    asm = FrameAssembler()
    for p in old[:-2]:
        asm.push(p)
    assert asm.inflight == 1
    done = [f for f in (asm.push(p) for p in new) if f is not None]
    assert len(done) == 1 and done[0].frame_id == 2
    assert asm.stats.frames_dropped_superseded >= 1
    assert asm.inflight == 0, "the stale partial must be gone, not lingering"


def test_late_fragments_of_a_superseded_frame_are_discarded():
    old = build_packets(0, SENSOR_EO, 1, 100, telem(), jpeg_of(9000, seed=8))
    new = build_packets(0, SENSOR_EO, 5, 200, telem(), jpeg_of(9000, seed=9))
    asm = FrameAssembler()
    asm.push(old[0])
    for p in new:
        asm.push(p)
    for p in old[1:]:
        assert asm.push(p) is None, "no resurrection of an obsolete frame"


def test_deadline_expires_a_stalled_frame():
    clock = {"t": 0.0}
    asm = FrameAssembler(deadline_s=0.1, clock=lambda: clock["t"])
    pkts = build_packets(0, SENSOR_EO, 1, 1, telem(), jpeg_of(40_000, seed=10))
    asm.push(pkts[0])
    assert asm.inflight == 1
    clock["t"] = 0.5
    assert asm.sweep() == 1
    assert asm.inflight == 0
    assert asm.stats.frames_dropped_incomplete == 1


def test_inflight_is_bounded_under_sustained_loss():
    """1000 frames, only the first fragment of each. Memory must not grow."""
    asm = FrameAssembler(max_inflight=2)
    for fid in range(1000):
        pkts = build_packets(0, SENSOR_EO, fid, fid, telem(), jpeg_of(40_000, seed=fid % 5))
        asm.push(pkts[0])
    assert asm.inflight <= 2
    assert asm.stats.frames_completed == 0


def test_streams_are_independent():
    """Interleaved UAVs and sensors must not corrupt one another."""
    streams = {(u, s): jpeg_of(9000, seed=u * 10 + s) for u in range(4) for s in (SENSOR_EO, SENSOR_IR)}
    packets = []
    for (u, s), payload in streams.items():
        packets += build_packets(u, s, 1, 1, telem(), payload)
    np.random.default_rng(0).shuffle(packets)

    asm = FrameAssembler()
    done = {f.key: f.jpeg for f in (asm.push(p) for p in packets) if f is not None}
    assert done == streams


# --------------------------------------------------------------------------------------
# Malformed input -- must never raise out of push()
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mangle,why",
    [
        (lambda p: p[:10], "truncated header"),
        (lambda p: b"", "empty"),
        (lambda p: b"\x00" * 200, "bad magic"),
        (lambda p: struct.pack("<I", MAGIC) + b"\xff" * 40, "bad version"),
        (lambda p: p[: HEADER_SIZE + 10], "truncated telemetry"),
        (lambda p: p[: HEADER_SIZE + TELEM_SIZE + 5], "truncated payload"),
    ],
)
def test_parse_rejects_malformed_datagrams(mangle, why):
    good = build_packets(0, SENSOR_EO, 1, 1, telem(), jpeg_of(5000))[0]
    with pytest.raises(ProtocolError):
        parse_packet(mangle(good))


def test_assembler_never_raises_on_garbage():
    rng = np.random.default_rng(1234)
    asm = FrameAssembler()
    for _ in range(2000):
        n = int(rng.integers(0, 300))
        assert asm.push(bytes(rng.integers(0, 256, n, dtype=np.uint8))) is None
    assert asm.stats.packets_bad > 0
    # and a good frame still gets through afterwards
    payload = jpeg_of(3000, seed=11)
    done = [
        f
        for f in (asm.push(p) for p in build_packets(0, SENSOR_EO, 9, 9, telem(), payload))
        if f is not None
    ]
    assert len(done) == 1 and done[0].jpeg == payload


def test_corrupt_fragment_count_does_not_splice_mismatched_data():
    pkts = list(build_packets(0, SENSOR_EO, 1, 1, telem(), jpeg_of(9000, seed=12)))
    asm = FrameAssembler()
    asm.push(pkts[0])
    bad = bytearray(pkts[1])
    struct.pack_into("<H", bad, 22, 999)  # frag_count field
    assert asm.push(bytes(bad)) is None
    assert asm.stats.packets_bad >= 1


def test_mtu_too_small_is_rejected_loudly():
    with pytest.raises(ValueError):
        max_payload_per_fragment(HEADER_SIZE + TELEM_SIZE)


def test_fragment_count_fits_the_counter():
    """A frame needing >65535 fragments must fail loudly rather than silently wrap."""
    with pytest.raises(ValueError, match="fragment"):
        build_packets(0, SENSOR_EO, 1, 1, telem(), b"\x00" * (70_000 * 1300))
