"""H.264 output to Mission Planner via an ffmpeg subprocess.

Encoding happens in a **separate OS process**, so it costs the fusion loop nothing but one
memcpy. Codec, bitrate and latency tuning are a command-line string away, there is no
build-time dependency (a static ``ffmpeg.exe`` beside the app is enough on Windows), and if
the encoder stalls the pipe backpressure is visible and recoverable rather than silently
wedging the pipeline.

A dedicated writer thread sits between the fusion loop and the pipe, fed by a 1-deep slot.
If ffmpeg falls behind, frames are **dropped** rather than queued: a live feed that is
2 seconds late is worse than one that skipped a frame.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from .config import EncoderConfig

log = logging.getLogger(__name__)

__all__ = ["FfmpegSink", "EncoderUnavailable"]


class EncoderUnavailable(RuntimeError):
    """ffmpeg is not usable; the pipeline can still run headless or write stills."""


class FfmpegSink:
    """Push BGR frames to ffmpeg -> H.264 -> UDP."""

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._slot: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self.frames_written = 0
        self.frames_dropped = 0
        self._broken = False

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def url(self) -> str:
        c = self.cfg
        if c.container == "rtp":
            return f"rtp://{c.host}:{c.port}"
        return f"udp://{c.host}:{c.port}?pkt_size=1316"

    def command(self) -> list[str]:
        c = self.cfg
        return [
            c.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{c.width}x{c.height}", "-r", str(c.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", c.preset,
            "-tune", c.tune,
            "-b:v", c.bitrate,
            "-g", str(max(c.fps, 1)),        # one keyframe a second: fast receiver join
            "-bf", "0",                        # no B-frames -- they add reorder latency
            "-pix_fmt", "yuv420p",             # what every player actually accepts
            "-f", c.container,
            self.url,
        ]

    def start(self) -> None:
        if shutil.which(self.cfg.ffmpeg) is None:
            raise EncoderUnavailable(
                f"{self.cfg.ffmpeg!r} not found on PATH -- run with --no-encoder, "
                f"or point AppConfig.encoder.ffmpeg at the binary"
            )
        cmd = self.command()
        log.info("encoder: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=0,
            )
        except OSError as exc:
            raise EncoderUnavailable(f"could not start ffmpeg: {exc}") from exc

        self._thread = threading.Thread(target=self._pump, name="encoder", daemon=True)
        self._thread.start()
        threading.Thread(target=self._drain_stderr, name="encoder-err", daemon=True).start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._new.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=timeout)
            except Exception:
                self._proc.kill()
        log.info("encoder stopped (%d written, %d dropped)", self.frames_written, self.frames_dropped)

    # -- data path ---------------------------------------------------------------------

    def submit(self, canvas_bgr: np.ndarray) -> None:
        """Hand the newest canvas to the encoder. Never blocks; supersedes any pending frame."""
        if self._broken:
            return
        c = self.cfg
        if canvas_bgr.shape[1] != c.width or canvas_bgr.shape[0] != c.height:
            frame = cv2.resize(canvas_bgr, (c.width, c.height), interpolation=cv2.INTER_AREA)
        else:
            frame = canvas_bgr
        # copy: the canvas keeps mutating under us the moment this returns
        frame = np.ascontiguousarray(frame)
        with self._lock:
            if self._slot is not None:
                self.frames_dropped += 1
            self._slot = frame
        self._new.set()

    def _pump(self) -> None:
        while not self._stop.is_set():
            if not self._new.wait(timeout=0.25):
                continue
            self._new.clear()
            with self._lock:
                frame, self._slot = self._slot, None
            if frame is None:
                continue
            proc = self._proc
            if proc is None or proc.stdin is None or proc.poll() is not None:
                self._broken = True
                log.error("encoder: ffmpeg exited, stopping output")
                return
            try:
                proc.stdin.write(frame.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, OSError) as exc:
                self._broken = True
                log.error("encoder: pipe closed (%s)", exc)
                return

    def _drain_stderr(self) -> None:
        """Keep ffmpeg's stderr flowing; a full pipe would block the encoder."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            if self._stop.is_set():
                return
            msg = line.decode(errors="replace").strip()
            if msg:
                log.warning("ffmpeg: %s", msg)


class NullSink:
    """Stand-in when the encoder is disabled, so callers need no conditionals."""

    frames_written = 0
    frames_dropped = 0

    def start(self) -> None:
        pass

    def submit(self, canvas_bgr) -> None:
        self.frames_written += 1

    def stop(self, timeout: float = 0.0) -> None:
        pass
