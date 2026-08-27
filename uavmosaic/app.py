"""Backend entry point: wires ingestion, fusion, HUD and the encoder, then runs the loop.

    python -m uavmosaic.app                       # listen, fuse, stream to Mission Planner
    python -m uavmosaic.app --no-encoder --save out.png
    python -m uavmosaic.app --duration 20 --stats

View the output without Mission Planner:

    ffplay -fflags nobuffer -flags low_delay -i udp://127.0.0.1:5600
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2
import numpy as np

from .canvas import Canvas
from .config import AppConfig, EncoderConfig
from .coords import CanvasGeometry, GeodeticAnchor
from .encoder import EncoderUnavailable, FfmpegSink, NullSink
from .frames import SENSOR_EO
from .fusion import FusionEngine
from .hud import draw_hud
from .ingest import IngestGroup

log = logging.getLogger("uavmosaic")


class Application:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.canvas = Canvas(cfg.aoi)
        self.ingest = IngestGroup()
        for uav_id in cfg.uav_ids:
            for sensor_id in cfg.active_sensors():
                self.ingest.add(
                    uav_id, sensor_id, cfg.ports_for(sensor_id)[uav_id], host=cfg.bind_host
                )
        self.fusion = FusionEngine(cfg, self.ingest, self.canvas, sensor_id=SENSOR_EO)

        self.sink = NullSink()
        if cfg.encoder.enabled:
            try:
                self.sink = FfmpegSink(cfg.encoder)
                self.sink.start()
            except EncoderUnavailable as exc:
                log.warning("%s -- continuing without video output", exc)
                self.sink = NullSink()

        self._running = False
        # Letterbox rather than stretch. A square 2 km AOI forced into 16:9 would be
        # squashed to 56% height, which silently makes every distance read off the video
        # wrong -- unacceptable for a feed whose purpose is locating things.
        e = cfg.encoder
        self._out_scale = min(e.width / cfg.aoi.width, e.height / cfg.aoi.height)
        self._out_size = (
            int(round(cfg.aoi.width * self._out_scale)),
            int(round(cfg.aoi.height * self._out_scale)),
        )
        self._out_offset = (
            (e.width - self._out_size[0]) // 2,
            (e.height - self._out_size[1]) // 2,
        )
        self._frame = np.zeros((e.height, e.width, 3), dtype=np.uint8)

    def publish(self) -> np.ndarray:
        """Downscale, overlay, hand to the encoder. Never mutates the canvas."""
        nw, nh = self._out_size
        ox, oy = self._out_offset
        frame = self._frame
        cv2.resize(
            self.canvas.view(), (nw, nh), dst=frame[oy : oy + nh, ox : ox + nw],
            interpolation=cv2.INTER_AREA,
        )
        if self.cfg.hud:
            t = self.fusion.timer.percentiles().get("tick_total")
            extra = (f"tick p50 {t[1]:.0f}ms p99 {t[3]:.0f}ms",) if t else ()
            draw_hud(
                frame, self.canvas, self.cfg.aoi, self.fusion.footprints,
                stale={u: self.fusion.is_stale(u) for u in self.cfg.uav_ids},
                stats_lines=extra, scale=self._out_scale, offset=self._out_offset,
            )
        self.sink.submit(frame)
        return frame

    def run(self, duration: float | None = None, stats_every: float = 5.0) -> None:
        self.ingest.start()
        self._running = True
        period = 1.0 / self.cfg.target_hz
        t_start = time.monotonic()
        next_tick = t_start
        next_stats = t_start + stats_every
        n = 0

        log.info("running -- ctrl-c to stop")
        while self._running:
            now = time.monotonic()
            if duration is not None and (now - t_start) >= duration:
                break
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.02))
                continue

            self.fusion.tick()
            self.publish()
            n += 1
            next_tick = t_start + (n + 1) * period
            if next_tick < now:  # fell behind: resync rather than spiral
                next_tick = now + period

            if now >= next_stats:
                log.info("%s | %s", self.fusion.summary(), self.ingest.summary())
                next_stats = now + stats_every

        self.stop()

    def stop(self) -> None:
        self._running = False
        self.ingest.stop()
        self.fusion.close()
        self.sink.stop()

    def report(self) -> str:
        lines = [
            "", "=" * 74,
            self.fusion.summary(),
            self.ingest.summary(),
            f"encoder: {self.sink.frames_written} written, {self.sink.frames_dropped} dropped",
            "", "per-stage latency (ms):", self.fusion.timer.table(),
        ]
        if self.fusion.stats.reject_reasons:
            lines += ["", "rejections:"]
            lines += [
                f"  {v:5d}  {k}"
                for k, v in sorted(
                    self.fusion.stats.reject_reasons.items(), key=lambda kv: -kv[1]
                )
            ]
        lines.append("=" * 74)
        return "\n".join(lines)


def build_config(a) -> AppConfig:
    aoi = CanvasGeometry(
        e_min=-a.extent, n_min=-a.extent, e_max=a.extent, n_max=a.extent, gsd=a.gsd
    )
    anchor = None
    if a.anchor:
        lat, lon, *rest = [float(v) for v in a.anchor.split(",")]
        anchor = GeodeticAnchor(lat, lon, rest[0] if rest else 0.0)
    return AppConfig(
        aoi=aoi,
        anchor=anchor,
        target_hz=a.hz,
        bind_host=a.bind,
        ir_enabled=a.ir,
        hud=not a.no_hud,
        encoder=EncoderConfig(
            enabled=not a.no_encoder, host=a.out_host, port=a.out_port,
            width=a.out_width, height=a.out_height, fps=int(round(a.hz)),
            bitrate=a.bitrate, ffmpeg=a.ffmpeg,
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extent", type=float, default=1000.0, help="AOI half-size, metres")
    ap.add_argument("--gsd", type=float, default=0.5, help="canvas resolution, m/px")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--anchor", default=None, metavar="LAT,LON[,ALT]")
    ap.add_argument("--ir", action="store_true", help="also composite the IR streams")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--save", default=None, help="write the final mosaic to this PNG")
    ap.add_argument("--stats", action="store_true", help="print the latency table at exit")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--out-host", default="127.0.0.1")
    ap.add_argument("--out-port", type=int, default=5600)
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--bitrate", default="4M")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = build_config(a)
    print(cfg.describe())
    app = Application(cfg)
    signal.signal(signal.SIGINT, lambda *_: app.stop())

    try:
        app.run(duration=a.duration)
    except KeyboardInterrupt:
        app.stop()

    if a.save:
        cv2.imwrite(a.save, app.canvas.view())
        log.info("mosaic -> %s", a.save)
    if a.stats:
        print(app.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
