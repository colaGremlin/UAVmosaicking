"""Non-blocking UDP ingestion: one daemon receiver per stream, feeding 1-deep mailboxes.

Why threads rather than processes
---------------------------------
Every expensive call on this path -- ``cv2.imdecode``, and later ``cv2.warpPerspective`` --
releases the GIL, so these threads achieve real parallelism. Processes would force a 48 MB
canvas through a pickle/pipe round trip every tick, which is strictly worse.

Why a 1-deep mailbox rather than a queue
----------------------------------------
:class:`LatestSlot` holds exactly one frame and the newest always wins. A queue would absorb
a burst and then hand the fusion loop progressively staler frames, building a latency debt it
can never repay -- the classic failure mode for this kind of pipeline. Dropping an old frame
is free; showing an old frame is not.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .frames import SENSOR_NAMES, FrameBundle
from .protocol import DEFAULT_MTU, FrameAssembler

log = logging.getLogger(__name__)

__all__ = ["LatestSlot", "ReceiverStats", "UdpReceiver", "IngestGroup"]

#: 8 MB. A 1280x720 JPEG frame is ~65 fragments; four streams at 10 Hz burst hard enough
#: that the default (often 64 KB) drops packets under any scheduling hiccup.
DEFAULT_RCVBUF = 8 << 20

_SOCKET_TIMEOUT_S = 0.5  #: lets a stopped thread notice and exit promptly


class LatestSlot:
    """Thread-safe single-slot mailbox. Newest write wins; reads never block a writer."""

    __slots__ = ("_value", "_lock", "_writes", "_drops")

    def __init__(self) -> None:
        self._value: FrameBundle | None = None
        self._lock = threading.Lock()
        self._writes = 0
        self._drops = 0

    def put(self, value: FrameBundle) -> None:
        with self._lock:
            if self._value is not None:
                self._drops += 1  # fusion never consumed the previous one
            self._value = value
            self._writes += 1

    def peek(self) -> FrameBundle | None:
        """Read without consuming -- the canvas is persistent, so a repeat is harmless."""
        with self._lock:
            return self._value

    def take(self) -> FrameBundle | None:
        """Read and clear, so the same frame is not composited twice."""
        with self._lock:
            v, self._value = self._value, None
            return v

    @property
    def counters(self) -> tuple[int, int]:
        """``(writes, superseded)`` -- a rising drop count means fusion is behind."""
        with self._lock:
            return self._writes, self._drops


@dataclass
class ReceiverStats:
    packets: int = 0
    bytes_in: int = 0
    frames: int = 0
    decode_failures: int = 0
    last_frame_at: float = 0.0
    fov_warnings: int = 0
    _t0: float = field(default_factory=time.monotonic)

    @property
    def fps(self) -> float:
        dt = time.monotonic() - self._t0
        return self.frames / dt if dt > 0 else 0.0

    @property
    def mbit_s(self) -> float:
        dt = time.monotonic() - self._t0
        return (self.bytes_in * 8 / 1e6) / dt if dt > 0 else 0.0


class UdpReceiver(threading.Thread):
    """One daemon thread per (uav, sensor): recv -> reassemble -> decode -> publish.

    Never touches the canvas and never blocks the fusion loop. Every failure mode --
    malformed packet, undecodable JPEG, socket error -- degrades to a dropped frame and a
    counter, because a receiver thread dying silently would look exactly like a UAV going
    quiet.
    """

    def __init__(
        self,
        uav_id: int,
        sensor_id: int,
        port: int,
        slot: LatestSlot,
        host: str = "0.0.0.0",
        mtu: int = DEFAULT_MTU,
        rcvbuf: int = DEFAULT_RCVBUF,
        fov_tolerance_deg: float = 0.5,
    ) -> None:
        super().__init__(
            name=f"rx-uav{uav_id}-{SENSOR_NAMES.get(sensor_id, sensor_id)}", daemon=True
        )
        self.uav_id = uav_id
        self.sensor_id = sensor_id
        self.port = port
        self.host = host
        self.slot = slot
        self.mtu = mtu
        self.rcvbuf = rcvbuf
        self.fov_tolerance_deg = fov_tolerance_deg
        self.stats = ReceiverStats()
        self.assembler = FrameAssembler()
        # NB: named _stop_evt, not _stop. threading.Thread has its own private _stop()
        # method that join() calls internally; shadowing it with an Event makes every
        # join() raise "'Event' object is not callable" during interpreter teardown.
        self._stop_evt = threading.Event()
        #: set once the socket is bound and actually receiving. Until then, datagrams sent
        #: to this port are silently discarded by the OS -- a startup race that looks
        #: exactly like a dead UAV, so callers wait on it rather than sleeping and hoping.
        self.ready = threading.Event()
        self.bind_error: OSError | None = None
        self._sock: socket.socket | None = None

    # -- lifecycle ---------------------------------------------------------------------

    def _open(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        except OSError:  # some platforms cap this; not fatal
            log.warning("%s: could not raise SO_RCVBUF to %d", self.name, self.rcvbuf)
        s.bind((self.host, self.port))
        s.settimeout(_SOCKET_TIMEOUT_S)
        actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        log.info("%s listening on %s:%d (rcvbuf=%d)", self.name, self.host, self.port, actual)
        return s

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=join_timeout)

    # -- hot loop ----------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._sock = self._open()
        except OSError as exc:
            self.bind_error = exc
            log.error("%s: cannot bind port %d: %s", self.name, self.port, exc)
            return
        finally:
            self.ready.set()  # set even on failure so start() never hangs

        buf = self.mtu + 64  # headroom so an oversized datagram is visible, not truncated
        try:
            while not self._stop_evt.is_set():
                try:
                    datagram, _addr = self._sock.recvfrom(buf)
                except socket.timeout:
                    self.assembler.sweep()
                    continue
                except OSError as exc:
                    if self._stop_evt.is_set():
                        break
                    log.warning("%s: socket error: %s", self.name, exc)
                    continue

                self.stats.packets += 1
                self.stats.bytes_in += len(datagram)

                assembled = self.assembler.push(datagram)
                if assembled is None:
                    continue

                bundle = self._decode(assembled)
                if bundle is not None:
                    self.slot.put(bundle)
        finally:
            if self._sock is not None:
                self._sock.close()
            log.info("%s stopped (%d frames, %d packets)", self.name, self.stats.frames,
                     self.stats.packets)

    def _decode(self, assembled) -> FrameBundle | None:
        img = cv2.imdecode(np.frombuffer(assembled.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.stats.decode_failures += 1
            log.debug("%s: undecodable JPEG, frame %d", self.name, assembled.frame_id)
            return None

        t = assembled.telemetry
        if img.shape[1] != t.img_w or img.shape[0] != t.img_h:
            # Telemetry and pixels disagree about the frame size, so K would be wrong.
            self.stats.decode_failures += 1
            log.warning(
                "%s: image %dx%d != telemetry %dx%d, dropping",
                self.name, img.shape[1], img.shape[0], t.img_w, t.img_h,
            )
            return None

        if self.stats.fov_warnings == 0:
            d = t.fov_disagreement_deg()
            if d > self.fov_tolerance_deg:
                self.stats.fov_warnings += 1
                log.warning(
                    "%s: reported hfov %.2f deg disagrees with fx by %.2f deg -- sensor size "
                    "and Unity camera FOV describe different optics",
                    self.name, t.hfov_deg, d,
                )

        self.stats.frames += 1
        self.stats.last_frame_at = time.monotonic()
        return FrameBundle(
            uav_id=assembled.uav_id,
            sensor_id=assembled.sensor_id,
            frame_id=assembled.frame_id,
            t_capture_us=assembled.t_capture_us,
            telemetry=t,
            image=img,
            t_received=time.monotonic(),
        )


class IngestGroup:
    """Starts and supervises one receiver per stream, and exposes their mailboxes."""

    def __init__(self) -> None:
        self.slots: dict[tuple[int, int], LatestSlot] = {}
        self.receivers: dict[tuple[int, int], UdpReceiver] = {}

    def add(self, uav_id: int, sensor_id: int, port: int, **kw) -> UdpReceiver:
        key = (uav_id, sensor_id)
        if key in self.receivers:
            raise ValueError(f"stream {key} already registered")
        slot = LatestSlot()
        rx = UdpReceiver(uav_id, sensor_id, port, slot, **kw)
        self.slots[key] = slot
        self.receivers[key] = rx
        return rx

    def start(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Start every receiver and, by default, block until all sockets are bound.

        Without the wait there is a window in which the ports are not yet listening and any
        datagram that arrives is dropped by the OS with no error anywhere -- indistinguishable
        from a UAV that never transmitted.
        """
        for rx in self.receivers.values():
            rx.start()
        if not wait:
            return
        deadline = time.monotonic() + timeout
        for key, rx in self.receivers.items():
            if not rx.ready.wait(max(0.0, deadline - time.monotonic())):
                raise TimeoutError(f"receiver {key} did not bind within {timeout}s")
        failed = {k: r.bind_error for k, r in self.receivers.items() if r.bind_error}
        if failed:
            raise OSError(f"receivers failed to bind: {failed}")

    def stop(self) -> None:
        for rx in self.receivers.values():
            rx.stop()

    def snapshot(self, sensor_id: int, max_age_s: float, now: float | None = None):
        """Freshest frame per UAV for one sensor, dropping anything older than ``max_age_s``.

        Uses ``take`` so a frame is composited once. The canvas is persistent, so a UAV that
        goes quiet simply keeps its existing pixels rather than blanking.
        """
        now = time.monotonic() if now is None else now
        out: dict[int, FrameBundle] = {}
        for (uav_id, sid), slot in self.slots.items():
            if sid != sensor_id:
                continue
            f = slot.take()
            if f is not None and (now - f.t_received) <= max_age_s:
                out[uav_id] = f
        return out

    def summary(self) -> str:
        parts = []
        for (u, s), rx in sorted(self.receivers.items()):
            parts.append(
                f"uav{u}/{SENSOR_NAMES.get(s, s)}: {rx.stats.frames}f "
                f"{rx.stats.fps:.1f}Hz {rx.stats.mbit_s:.1f}Mb/s"
            )
        return " | ".join(parts)
