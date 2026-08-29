"""MJPEG-over-HTTP server -- the low-friction way to get the mosaic into Mission Planner.

Mission Planner can consume an MJPEG HTTP stream natively: right-click the HUD and choose
Set MJPEG Source. That path needs nothing installed. The alternative, RTP/H.264, is Mission
Planner's other documented route but requires a specific GStreamer runtime on the machine,
which is the usual place that setup stalls.

MJPEG costs more bandwidth than H.264 because every frame is a complete JPEG with no
inter-frame prediction. Measured here at 1280x720, quality 80, 10 Hz: about 2 Mbit/s on a
sparsely covered canvas, rising with coverage since a fuller mosaic compresses less well.
Loopback absorbs that without noticing. Over a real radio link, use the H.264 path.

Design mirrors :class:`~uavmosaic.encoder.FfmpegSink`: one latest-frame slot, a server that
never blocks the fusion loop, and a slow client that gets stale frames rather than being
allowed to apply backpressure.
"""

from __future__ import annotations

import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["MjpegSink"]

_BOUNDARY = "uavmosaicframe"

_INDEX = """<!doctype html><meta charset=utf-8>
<title>UAV Mosaic</title>
<style>html,body{margin:0;background:#0d1117;color:#c9d1d9;
font-family:system-ui,sans-serif;height:100%%;display:flex;flex-direction:column}
header{padding:8px 14px;font-size:13px;border-bottom:1px solid #21262d}
code{background:#161b22;padding:2px 6px;border-radius:3px}
img{flex:1;min-height:0;object-fit:contain;background:#000}</style>
<header>UAV mosaic &mdash; live. Mission Planner source:
<code>http://%(host)s:%(port)d/stream.mjpg</code></header>
<img src="/stream.mjpg" alt="live mosaic">
"""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # one response per connection; simplest and most compatible
    server_version = "uavmosaic/1.0"

    def log_message(self, fmt, *args):  # noqa: D102 - silence per-request stdout spam
        log.debug("mjpeg %s - %s", self.address_string(), fmt % args)

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        sink: MjpegSink = self.server.sink  # type: ignore[attr-defined]

        if self.path in ("/", "/index.html"):
            page = (_INDEX % {"host": sink.advertise_host, "port": sink.port}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if self.path.split("?")[0] not in ("/stream.mjpg", "/stream", "/video", "/mjpg"):
            self.send_error(404, "try /stream.mjpg")
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()

        sink._client_joined()
        last = -1
        try:
            while not sink.stopping:
                jpeg, seq = sink.wait_for_frame(last, timeout=1.0)
                if jpeg is None:
                    continue  # keep the connection open through a quiet spell
                last = seq
                self.wfile.write(b"--" + _BOUNDARY.encode() + b"\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # viewer closed the tab; entirely normal
        except OSError as exc:
            log.debug("mjpeg client dropped: %s", exc)
        finally:
            sink._client_left()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegSink:
    """Serves the newest canvas frame as MJPEG over HTTP.

    Same interface as :class:`~uavmosaic.encoder.FfmpegSink` so the app can hold a list of
    sinks and submit to all of them without special-casing.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        quality: int = 80,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.host = host
        self.port = port
        self.quality = int(np.clip(quality, 1, 100))
        self.width = width
        self.height = height

        self._jpeg: bytes | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

        self.stopping = False
        self.clients = 0
        self.frames_written = 0
        self.frames_dropped = 0  # kept for interface parity; MJPEG never queues

    # -- helpers ----------------------------------------------------------------------

    @property
    def advertise_host(self) -> str:
        """A host a viewer can actually dial. 0.0.0.0 is a bind address, not an address.

        Resolves to loopback when bound to all interfaces, because Mission Planner almost
        always runs on the same machine. gethostbyname() was tried first and rejected: on a
        box with Hyper-V or VirtualBox installed it happily returns a virtual adapter address
        that nothing can reach. :meth:`lan_hosts` reports the real ones for the remote case.
        """
        if self.host not in ("0.0.0.0", "", "::"):
            return self.host
        return "127.0.0.1"

    def lan_hosts(self) -> list[str]:
        """Addresses another machine on the network could use, best guess first."""
        out = []
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))          # no packet is sent; picks the default route
            out.append(probe.getsockname()[0])
            probe.close()
        except OSError:
            pass
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip not in out and not ip.startswith("127."):
                    out.append(ip)
        except OSError:
            pass
        return out

    @property
    def url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}/stream.mjpg"

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self._server = _Server((self.host, self.port), _Handler)
        self._server.sink = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mjpeg", daemon=True
        )
        self._thread.start()
        log.info("VIDEO FEED IS UP.")
        log.info("  Watch it in a browser:  %s", self.url)
        log.info("  Or in Mission Planner: right-click the HUD, choose Set MJPEG Source,")
        log.info("  and give it that same address.")
        others = list(self.lan_hosts())
        if self.host in ("0.0.0.0", "", "::") and others:
            log.info("  From another computer on this network, swap 127.0.0.1 for one of: %s",
                     ", ".join(others[:3]))

    def stop(self, timeout: float = 2.0) -> None:
        self.stopping = True
        with self._new:
            self._new.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("MJPEG server stopped (%d frames served)", self.frames_written)

    # -- data path ---------------------------------------------------------------------

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Encode and publish the newest frame. Never blocks the fusion loop."""
        if self.stopping:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._new:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self.frames_written += 1
            self._new.notify_all()

    def wait_for_frame(self, since_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``since_seq`` exists. ``(None, seq)`` on timeout."""
        with self._new:
            if self._seq <= since_seq:
                self._new.wait(timeout)
            if self._seq <= since_seq or self._jpeg is None:
                return None, self._seq
            return self._jpeg, self._seq

    def _client_joined(self) -> None:
        with self._lock:
            self.clients += 1
        log.info("MJPEG viewer connected (%d now watching)", self.clients)

    def _client_left(self) -> None:
        with self._lock:
            self.clients = max(0, self.clients - 1)
        log.info("MJPEG viewer disconnected (%d still watching)", self.clients)
