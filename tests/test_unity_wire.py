"""Pins the wire layout the Unity C# sender hardcodes.

``tools/unity/UavStreamer.cs`` writes fields at fixed byte offsets with hand-rolled
little-endian writers. It cannot be compiled here, so instead this test builds a datagram
byte-for-byte the way the C# does -- same offsets, same order, same widths -- and asserts the
Python parser reads back exactly what was written.

If anyone changes ``HEADER_FMT`` or ``TELEM_FMT`` without updating the C#, this fails. That
matters because the failure mode in the field is silent: the backend would just tick up its
``packets_bad`` counter while the operator sees a black screen.
"""

import struct

import numpy as np
import pytest

from uavmosaic.protocol import (
    FLAG_AGL_VALID,
    FLAG_LRF_VALID,
    FLAG_TELEM_PRESENT,
    HEADER_SIZE,
    MAGIC,
    TELEM_SIZE,
    VERSION,
    FrameAssembler,
    parse_packet,
)


def _le_u32(v):
    return bytes(((v >> 0) & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255))


def _le_u16(v):
    return bytes(((v >> 0) & 255, (v >> 8) & 255))


def _le_u64(v):
    return bytes((v >> (8 * i)) & 255 for i in range(8))


def _le_f32(v):
    return _le_u32(struct.unpack("<I", struct.pack("<f", v))[0])


def csharp_packet(
    uav_id, sensor_id, flags, frame_id, t_us, frag_i, frag_n, frag_len, total, offset,
    pos, rot, fx, fy, cx, cy, lrf, agl, hfov, zoom, w, h, payload,
):
    """Mirror of UavStreamer.SendFragmented + PackTelemetry, field for field."""
    head = b"".join([
        _le_u32(MAGIC),
        bytes([VERSION, uav_id, sensor_id, flags]),
        _le_u32(frame_id),
        _le_u64(t_us),
        _le_u16(frag_i),
        _le_u16(frag_n),
        _le_u16(frag_len),
        _le_u32(total),
        _le_u16(TELEM_SIZE),
        _le_u32(offset),
    ])
    telem = b"".join([
        _le_f32(pos[0]), _le_f32(pos[1]), _le_f32(pos[2]),
        _le_f32(rot[0]), _le_f32(rot[1]), _le_f32(rot[2]), _le_f32(rot[3]),
        _le_u16(w), _le_u16(h),
        _le_f32(fx), _le_f32(fy), _le_f32(cx), _le_f32(cy),
        _le_f32(lrf), _le_f32(agl), _le_f32(hfov), _le_f32(zoom),
    ])
    assert len(head) == HEADER_SIZE, f"C# header is {len(head)} B, Python expects {HEADER_SIZE}"
    assert len(telem) == TELEM_SIZE, f"C# telemetry is {len(telem)} B, Python expects {TELEM_SIZE}"
    return head + telem + payload


PAYLOAD = bytes(range(256)) * 3


def a_packet(**over):
    args = dict(
        uav_id=2, sensor_id=0,
        flags=FLAG_TELEM_PRESENT | FLAG_LRF_VALID | FLAG_AGL_VALID,
        frame_id=987_654, t_us=1_234_567_890_123, frag_i=0, frag_n=1,
        frag_len=len(PAYLOAD), total=len(PAYLOAD), offset=0,
        pos=(12.5, 300.25, -47.75), rot=(0.5, -0.5, 0.5, 0.5),
        fx=1108.5125, fy=1108.5125, cx=640.0, cy=360.0,
        lrf=305.5, agl=300.0, hfov=60.0, zoom=1.0,
        w=1280, h=720, payload=PAYLOAD,
    )
    args.update(over)
    return csharp_packet(**args)


def test_csharp_field_sizes_match_python():
    """Fails loudly if either struct format drifts from the C# constants."""
    assert HEADER_SIZE == 36, "UavStreamer.cs hardcodes HeaderSize = 36"
    assert TELEM_SIZE == 64, "UavStreamer.cs hardcodes TelemSize = 64"
    assert struct.calcsize("<I") == 4 and struct.calcsize("<Q") == 8


def test_python_parses_a_csharp_built_datagram():
    h, t, payload = parse_packet(a_packet())

    assert h.version == VERSION
    assert h.uav_id == 2
    assert h.sensor_id == 0
    assert h.frame_id == 987_654
    assert h.t_capture_us == 1_234_567_890_123
    assert (h.frag_index, h.frag_count) == (0, 1)
    assert h.total_len == len(PAYLOAD)
    assert h.frag_offset == 0
    assert payload == PAYLOAD

    assert np.allclose(t.pos_unity, (12.5, 300.25, -47.75))
    assert np.allclose(t.quat_world_cam, (0.5, -0.5, 0.5, 0.5))
    assert (t.img_w, t.img_h) == (1280, 720)
    assert np.isclose(t.fx, 1108.5125, rtol=1e-6)
    assert np.isclose(t.cy, 360.0)
    assert np.isclose(t.lrf_slant_m, 305.5, rtol=1e-6)
    assert np.isclose(t.agl_m, 300.0, rtol=1e-6)
    assert np.isclose(t.hfov_deg, 60.0, rtol=1e-6)


def test_csharp_optional_flags_round_trip():
    _, t, _ = parse_packet(a_packet(flags=FLAG_TELEM_PRESENT))
    assert t.lrf_slant_m is None and t.agl_m is None

    _, t, _ = parse_packet(a_packet(flags=FLAG_TELEM_PRESENT | FLAG_LRF_VALID))
    assert t.lrf_slant_m is not None and t.agl_m is None


def test_csharp_fragmentation_reassembles_through_the_real_assembler():
    """Exercises the explicit frag_offset the C# writes, at the real 1400 B MTU."""
    mtu, big = 1400, bytes(np.random.default_rng(5).integers(0, 256, 40_000, dtype=np.uint8))
    chunk = mtu - HEADER_SIZE - TELEM_SIZE
    n = (len(big) + chunk - 1) // chunk

    packets = [
        a_packet(
            frag_i=i, frag_n=n, frag_len=len(big[i * chunk : (i + 1) * chunk]),
            total=len(big), offset=i * chunk, payload=big[i * chunk : (i + 1) * chunk],
        )
        for i in range(n)
    ]
    assert all(len(p) <= mtu for p in packets), "a C# datagram exceeded the MTU"

    asm = FrameAssembler()
    done = [f for f in (asm.push(p) for p in packets) if f is not None]
    assert len(done) == 1
    assert done[0].jpeg == big, "C#-style fragmentation did not reassemble byte-exactly"


def test_csharp_quaternion_feeds_the_real_conversion():
    """A pose written the C# way must drive the LH->RH path to a valid rotation.

    Unity pitch +90 about X is nose-down, so this must come out as the nadir matrix.
    """
    import math

    s = math.sin(math.radians(90) / 2.0)
    c = math.cos(math.radians(90) / 2.0)
    _, t, _ = parse_packet(a_packet(rot=(s, 0.0, 0.0, c), pos=(10.0, 250.0, -30.0)))

    R = t.R_enu_cam()
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-6)

    # Unity (x, y, z) -> ENU (E, N, U) = (x, z, y)
    assert np.allclose(t.cam_enu(), (10.0, -30.0, 250.0))


@pytest.mark.parametrize("bad_hfov,should_warn", [(60.0, False), (75.0, True)])
def test_fov_cross_check_catches_a_mismatched_unity_camera(bad_hfov, should_warn):
    """The C# derives fx from Unity's vertical FOV and also reports hfov. If the physical
    camera and the FOV describe different optics the two disagree, and that must surface."""
    _, t, _ = parse_packet(a_packet(hfov=bad_hfov))
    assert (t.fov_disagreement_deg() > 0.5) == should_warn
