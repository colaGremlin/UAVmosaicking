"""UDP wire protocol: telemetry and JPEG travel in the *same* datagram.

Every fragment carries the full 36-byte header AND the full 64-byte telemetry block, then a
slice of the JPEG. Repeating the telemetry costs ~64 B x ~65 fragments = 4 KB per frame
(about 5%) and buys the property that matters: **pose and pixels are atomic**. Lose
fragment 0 on a real radio link and the frame is still fully interpretable. Nothing can ever
pair the wrong pose with the right image, because they travelled together.

Layout (little-endian, no struct padding)::

    HEADER  36 B
      u32  magic         0x31564155 == b'UAV1'
      u8   version
      u8   uav_id        0..3
      u8   sensor_id     0=EO 1=IR
      u8   flags         b0 lrf_valid | b1 agl_valid | b2 telem_present
      u32  frame_id      monotonic per (uav, sensor)
      u64  t_capture_us  Unity capture time -- the sync key
      u16  frag_index
      u16  frag_count
      u16  frag_len      JPEG bytes in THIS datagram
      u32  total_len     JPEG bytes in the whole frame
      u16  telem_len     64
      u32  frag_offset   byte offset of this slice within the whole JPEG
    TELEMETRY  64 B
      f32 x3   pos_unity        x, y, z      (Unity world, metres, RAW)
      f32 x4   quat_world_cam   x, y, z, w   (Unity world -> camera, RAW)
      u16 x2   img_w, img_h
      f32 x4   fx, fy, cx, cy   pixels at the current zoom
      f32 x4   lrf_slant_m, agl_m, hfov_deg, zoom
    PAYLOAD  frag_len B of JPEG

The MTU cap is deliberately 1400 B rather than the ~64 KB loopback allows: a protocol that
only ever sees single-datagram frames in simulation would hide the fragmentation bugs that
bite the moment this moves onto a real link.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

from .frames import Telemetry

__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_FMT",
    "HEADER_SIZE",
    "TELEM_FMT",
    "TELEM_SIZE",
    "DEFAULT_MTU",
    "max_payload_per_fragment",
    "FLAG_LRF_VALID",
    "FLAG_AGL_VALID",
    "FLAG_TELEM_PRESENT",
    "PacketHeader",
    "ProtocolError",
    "build_packets",
    "parse_packet",
    "AssembledFrame",
    "FrameAssembler",
]

MAGIC = 0x31564155  # b'UAV1' read as little-endian u32
VERSION = 1

HEADER_FMT = "<IBBBBIQHHHIHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 36

TELEM_FMT = "<3f4f2H4f4f"
TELEM_SIZE = struct.calcsize(TELEM_FMT)  # 64

DEFAULT_MTU = 1400  #: bytes on the wire per datagram, header + telemetry + payload

FLAG_LRF_VALID = 0x01
FLAG_AGL_VALID = 0x02
FLAG_TELEM_PRESENT = 0x04

#: Drop an incomplete frame this long after its first fragment.
DEFAULT_FRAME_DEADLINE_S = 0.150

#: Partial frames held per (uav, sensor). Bounded so packet loss cannot grow memory.
DEFAULT_MAX_INFLIGHT = 2


class ProtocolError(Exception):
    """Malformed datagram. Always recoverable by dropping the datagram."""


def max_payload_per_fragment(mtu: int = DEFAULT_MTU) -> int:
    n = mtu - HEADER_SIZE - TELEM_SIZE
    if n <= 0:
        raise ValueError(f"mtu {mtu} too small for {HEADER_SIZE + TELEM_SIZE} B of framing")
    return n


@dataclass(frozen=True)
class PacketHeader:
    version: int
    uav_id: int
    sensor_id: int
    flags: int
    frame_id: int
    t_capture_us: int
    frag_index: int
    frag_count: int
    frag_len: int
    total_len: int
    telem_len: int
    frag_offset: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.uav_id, self.sensor_id)


# --------------------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------------------


def _pack_telemetry(t: Telemetry) -> tuple[bytes, int]:
    flags = FLAG_TELEM_PRESENT
    lrf = 0.0
    agl = 0.0
    if t.lrf_slant_m is not None:
        flags |= FLAG_LRF_VALID
        lrf = float(t.lrf_slant_m)
    if t.agl_m is not None:
        flags |= FLAG_AGL_VALID
        agl = float(t.agl_m)
    blob = struct.pack(
        TELEM_FMT,
        *(float(v) for v in t.pos_unity),
        *(float(v) for v in t.quat_world_cam),
        int(t.img_w),
        int(t.img_h),
        float(t.fx),
        float(t.fy),
        float(t.cx),
        float(t.cy),
        lrf,
        agl,
        float(t.hfov_deg),
        float(t.zoom),
    )
    return blob, flags


def _unpack_telemetry(blob: bytes, flags: int) -> Telemetry:
    v = struct.unpack(TELEM_FMT, blob)
    return Telemetry(
        pos_unity=(v[0], v[1], v[2]),
        quat_world_cam=(v[3], v[4], v[5], v[6]),
        img_w=v[7],
        img_h=v[8],
        fx=v[9],
        fy=v[10],
        cx=v[11],
        cy=v[12],
        lrf_slant_m=v[13] if flags & FLAG_LRF_VALID else None,
        agl_m=v[14] if flags & FLAG_AGL_VALID else None,
        hfov_deg=v[15],
        zoom=v[16],
    )


def build_packets(
    uav_id: int,
    sensor_id: int,
    frame_id: int,
    t_capture_us: int,
    telemetry: Telemetry,
    jpeg: bytes,
    mtu: int = DEFAULT_MTU,
) -> list[bytes]:
    """Fragment one JPEG frame into datagrams, each self-describing and pose-complete.

    A zero-length JPEG still produces one datagram: the pose is worth delivering even when
    the image is not, and it keeps ``frag_count >= 1`` invariant everywhere downstream.
    """
    if not (0 <= uav_id <= 255 and 0 <= sensor_id <= 255):
        raise ValueError("uav_id and sensor_id must fit in a byte")
    chunk = max_payload_per_fragment(mtu)
    total = len(jpeg)
    n = max(1, (total + chunk - 1) // chunk)
    if n > 0xFFFF:
        raise ValueError(f"frame needs {n} fragments, exceeding the u16 fragment counter")

    telem_blob, flags = _pack_telemetry(telemetry)
    out: list[bytes] = []
    for i in range(n):
        piece = jpeg[i * chunk : (i + 1) * chunk]
        head = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            uav_id,
            sensor_id,
            flags,
            frame_id & 0xFFFFFFFF,
            t_capture_us & 0xFFFFFFFFFFFFFFFF,
            i,
            n,
            len(piece),
            total,
            TELEM_SIZE,
            i * chunk,
        )
        out.append(head + telem_blob + piece)
    return out


# --------------------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------------------


def parse_packet(datagram: bytes) -> tuple[PacketHeader, Telemetry | None, bytes]:
    """Datagram -> ``(header, telemetry, payload_slice)``.

    Raises :class:`ProtocolError` on anything malformed. Callers drop the datagram and carry
    on; a bad packet must never be able to take down a receiver thread.
    """
    if len(datagram) < HEADER_SIZE:
        raise ProtocolError(f"runt datagram: {len(datagram)} B < {HEADER_SIZE} B header")

    (
        magic, version, uav_id, sensor_id, flags, frame_id, t_capture_us,
        frag_index, frag_count, frag_len, total_len, telem_len, frag_offset,
    ) = struct.unpack(HEADER_FMT, datagram[:HEADER_SIZE])

    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08X}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version {version} (expected {VERSION})")
    if frag_count == 0 or frag_index >= frag_count:
        raise ProtocolError(f"bad fragment index {frag_index}/{frag_count}")

    off = HEADER_SIZE
    telemetry = None
    if telem_len:
        if telem_len != TELEM_SIZE:
            raise ProtocolError(f"telem_len {telem_len} != {TELEM_SIZE}")
        if len(datagram) < off + telem_len:
            raise ProtocolError("datagram truncated inside the telemetry block")
        telemetry = _unpack_telemetry(datagram[off : off + telem_len], flags)
        off += telem_len

    if len(datagram) < off + frag_len:
        raise ProtocolError(
            f"datagram truncated: need {off + frag_len} B, got {len(datagram)} B"
        )
    payload = datagram[off : off + frag_len]

    header = PacketHeader(
        version=version, uav_id=uav_id, sensor_id=sensor_id, flags=flags,
        frame_id=frame_id, t_capture_us=t_capture_us, frag_index=frag_index,
        frag_count=frag_count, frag_len=frag_len, total_len=total_len,
        telem_len=telem_len, frag_offset=frag_offset,
    )
    return header, telemetry, payload


@dataclass
class AssembledFrame:
    """A complete JPEG plus the pose it travelled with. Not yet decoded."""

    uav_id: int
    sensor_id: int
    frame_id: int
    t_capture_us: int
    telemetry: Telemetry
    jpeg: bytes

    @property
    def key(self) -> tuple[int, int]:
        return (self.uav_id, self.sensor_id)


@dataclass
class _Partial:
    frame_id: int
    total_len: int
    frag_count: int
    telemetry: Telemetry | None
    buf: bytearray
    seen: set = field(default_factory=set)
    first_seen: float = 0.0
    bytes_in: int = 0

    @property
    def complete(self) -> bool:
        return len(self.seen) == self.frag_count and self.bytes_in >= self.total_len


@dataclass
class AssemblerStats:
    frames_completed: int = 0
    frames_dropped_incomplete: int = 0
    frames_dropped_superseded: int = 0
    packets_bad: int = 0
    duplicates: int = 0


class FrameAssembler:
    """Reassembles fragments per ``(uav_id, sensor_id)`` with bounded memory.

    Policy, straight from the blueprint:

    * A partial frame is dropped whole the moment a **newer** ``frame_id`` starts arriving
      or its deadline passes. A partial JPEG is never handed on -- half an image with a
      valid pose is worse than no image, because it looks plausible.
    * At most ``max_inflight`` partial frames live per stream, so sustained loss cannot grow
      memory without bound.
    * Duplicate fragments (a real possibility on a retransmitting link) are ignored rather
      than double-counted.
    """

    def __init__(
        self,
        deadline_s: float = DEFAULT_FRAME_DEADLINE_S,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        clock=time.monotonic,
    ) -> None:
        self.deadline_s = deadline_s
        self.max_inflight = max_inflight
        self._clock = clock
        self._streams: dict[tuple[int, int], dict[int, _Partial]] = {}
        self._last_done: dict[tuple[int, int], int] = {}
        self.stats = AssemblerStats()

    def push(self, datagram: bytes) -> AssembledFrame | None:
        """Feed one datagram; returns a frame only when it completes on this packet."""
        try:
            header, telemetry, payload = parse_packet(datagram)
        except ProtocolError:
            self.stats.packets_bad += 1
            return None

        now = self._clock()
        parts = self._streams.setdefault(header.key, {})

        # Late or duplicated fragments of an already-delivered frame. Without this the
        # partial is recreated from scratch and the frame is emitted a second time.
        if header.frame_id <= self._last_done.get(header.key, -1):
            self.stats.duplicates += 1
            return None

        # Anything older than the newest frame_id we have seen is dead weight.
        newest = max([header.frame_id, *parts.keys()])
        for fid in [f for f in parts if f < newest]:
            del parts[fid]
            self.stats.frames_dropped_superseded += 1
        if header.frame_id < newest:
            self.stats.frames_dropped_superseded += 1
            return None

        for fid in [f for f in parts if now - parts[f].first_seen > self.deadline_s]:
            del parts[fid]
            self.stats.frames_dropped_incomplete += 1

        p = parts.get(header.frame_id)
        if p is None:
            while len(parts) >= self.max_inflight:
                del parts[min(parts)]
                self.stats.frames_dropped_incomplete += 1
            p = _Partial(
                frame_id=header.frame_id,
                total_len=header.total_len,
                frag_count=header.frag_count,
                telemetry=telemetry,
                buf=bytearray(header.total_len),
                first_seen=now,
            )
            parts[header.frame_id] = p
        elif p.telemetry is None and telemetry is not None:
            p.telemetry = telemetry

        if header.frag_index in p.seen:
            self.stats.duplicates += 1
            return None
        if header.frag_count != p.frag_count or header.total_len != p.total_len:
            # Two senders using one (uav, sensor) id, or a corrupted header that still
            # passed the cheap checks. Restart the frame rather than splice mismatched data.
            self.stats.packets_bad += 1
            del parts[header.frame_id]
            return None

        start = header.frag_offset
        if start + len(payload) > p.total_len:
            self.stats.packets_bad += 1
            return None
        p.buf[start : start + len(payload)] = payload
        p.seen.add(header.frag_index)
        p.bytes_in += len(payload)

        if not p.complete:
            return None

        del parts[header.frame_id]
        self._last_done[header.key] = header.frame_id
        if p.telemetry is None:  # cannot happen while telemetry rides every fragment
            self.stats.frames_dropped_incomplete += 1
            return None

        self.stats.frames_completed += 1
        return AssembledFrame(
            uav_id=header.uav_id,
            sensor_id=header.sensor_id,
            frame_id=header.frame_id,
            t_capture_us=header.t_capture_us,
            telemetry=p.telemetry,
            jpeg=bytes(p.buf),
        )

    def sweep(self) -> int:
        """Expire timed-out partials. Call periodically when traffic is sparse."""
        now = self._clock()
        n = 0
        for parts in self._streams.values():
            for fid in [f for f in parts if now - parts[f].first_seen > self.deadline_s]:
                del parts[fid]
                n += 1
        self.stats.frames_dropped_incomplete += n
        return n

    @property
    def inflight(self) -> int:
        return sum(len(p) for p in self._streams.values())
