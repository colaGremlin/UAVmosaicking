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
from .config import AppConfig, EncoderConfig, MjpegConfig
from .coords import CanvasGeometry, GeodeticAnchor
from .encoder import EncoderUnavailable, FfmpegSink, NullSink
from .frames import SENSOR_EO
from .fusion import FusionEngine
from .hud import draw_hud
from .ingest import IngestGroup
from .mjpeg import MjpegSink
from .tiles import TileServer

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

        # Several sinks may run at once: H.264 for a radio link or Mission Planner's
        # GStreamer path, MJPEG for Mission Planner's built-in viewer. Each gets the same
        # finished frame and each drops independently if it falls behind.
        self.sinks = []
        if cfg.encoder.enabled:
            try:
                sink = FfmpegSink(cfg.encoder)
                sink.start()
                self.sinks.append(sink)
            except EncoderUnavailable as exc:
                log.warning("%s -- continuing without H.264 output", exc)
        if cfg.mjpeg.enabled:
            try:
                m = MjpegSink(cfg.mjpeg.host, cfg.mjpeg.port, cfg.mjpeg.quality,
                              cfg.mjpeg.width, cfg.mjpeg.height)
                m.start()
                self.sinks.append(m)
            except OSError as exc:
                log.error("MJPEG server could not bind port %d: %s", cfg.mjpeg.port, exc)
        if not self.sinks:
            self.sinks.append(NullSink())

        # Map layer. Needs an anchor: without one the canvas has no geographic position and
        # there is nothing meaningful to serve tiles for.
        self.tiles = None
        if cfg.tiles_enabled:
            if cfg.anchor is None:
                log.error("--tiles needs --anchor LAT,LON[,ALT]; the mosaic has no geographic "
                          "position without it. Map layer disabled.")
            else:
                try:
                    self.tiles = TileServer(self.canvas, cfg.aoi, cfg.anchor,
                                            cfg.tiles_host, cfg.tiles_port)
                    self.tiles.start()
                except OSError as exc:
                    log.error("tile server could not bind port %d: %s", cfg.tiles_port, exc)
                    self.tiles = None

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
        self._view = None                     # eased view rect, canvas pixels
        self._has_letterbox = (self._out_size[0] != e.width or self._out_size[1] != e.height)
        if self._has_letterbox:
            waste = 1.0 - (self._out_size[0] * self._out_size[1]) / float(e.width * e.height)
            log.info("output %dx%d letterboxed into %dx%d (%.0f%% of the frame is bars); "
                     "omit --out-width/--out-height to size the output to the map instead",
                     self._out_size[0], self._out_size[1], e.width, e.height, waste * 100)

    def _view_rect(self) -> tuple[int, int, int, int]:
        """Which part of the canvas the video should show, in canvas pixels.

        Fixed to the whole area of interest unless ``--fit-view`` is on. With it on, the view
        tracks the imaged region and grows as coverage grows, so early in a sortie the feed
        shows the imagery filling the frame instead of a small patch adrift in black. The map
        layer is unaffected -- it stays georeferenced regardless, because a map you pan is a
        different thing from a video you watch.
        """
        cw, ch = self.cfg.aoi.width, self.cfg.aoi.height
        if not self.cfg.fit_view:
            return 0, 0, cw, ch

        box = self.canvas.covered_bbox()
        if box is None:
            return 0, 0, cw, ch

        x0, y0, x1, y1 = box
        pad = max((x1 - x0), (y1 - y0)) * 0.06 + 20
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

        # Match the output aspect so nothing is squashed, then clamp inside the canvas.
        ar = self.cfg.encoder.width / self.cfg.encoder.height
        w, h = x1 - x0, y1 - y0
        if w / h < ar:
            grow = (h * ar - w) / 2.0
            x0, x1 = x0 - grow, x1 + grow
        else:
            grow = (w / ar - h) / 2.0
            y0, y1 = y0 - grow, y1 + grow
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(cw), x1), min(float(ch), y1)

        # Ease toward the target so the view does not jump every time a footprint lands
        # slightly outside it. Purely cosmetic, but a twitching map is unreadable.
        prev = self._view
        if prev is None:
            self._view = (x0, y0, x1, y1)
        else:
            k = 0.12
            self._view = tuple(p + (t - p) * k for p, t in zip(prev, (x0, y0, x1, y1)))
        vx0, vy0, vx1, vy1 = self._view
        return int(vx0), int(vy0), max(int(vx1), int(vx0) + 2), max(int(vy1), int(vy0) + 2)

    def publish(self) -> np.ndarray:
        """Downscale, overlay, hand to the encoder. Never mutates the canvas."""
        frame = self._frame
        e = self.cfg.encoder
        sx0, sy0, sx1, sy1 = self._view_rect()
        src = self.canvas.view()[sy0:sy1, sx0:sx1]

        scale = min(e.width / (sx1 - sx0), e.height / (sy1 - sy0))
        nw = max(2, int(round((sx1 - sx0) * scale)))
        nh = max(2, int(round((sy1 - sy0) * scale)))
        ox, oy = (e.width - nw) // 2, (e.height - nh) // 2

        # The resize only repaints the map rectangle. Anything drawn on the surrounding bars
        # last tick would otherwise survive and accumulate -- footprint outlines built up into
        # solid blocks of colour there before this was added.
        if nw != e.width or nh != e.height:
            frame[:] = 0
        cv2.resize(
            src, (nw, nh), dst=frame[oy : oy + nh, ox : ox + nw],
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )

        if self.cfg.hud != "off":
            t = self.fusion.timer.percentiles().get("tick_total")
            extra = (f"tick p50 {t[1]:.0f}ms p99 {t[3]:.0f}ms",) if t else ()
            draw_hud(
                frame, self.canvas, self.cfg.aoi, self.fusion.footprints,
                stale={u: self.fusion.is_stale(u) for u in self.cfg.uav_ids},
                stats_lines=extra, scale=scale,
                offset=(ox - sx0 * scale, oy - sy0 * scale),
                level=self.cfg.hud,
                map_rect=(ox, oy, nw, nh),
                view_origin=(sx0 * scale, sy0 * scale),
            )
        for sink in self.sinks:
            sink.submit(frame)
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
        for sink in self.sinks:
            sink.stop()
        if self.tiles is not None:
            self.tiles.stop()

    def report(self) -> str:
        lines = [
            "", "=" * 74,
            self.fusion.summary(),
            self.ingest.summary(),
            *[f"{type(s).__name__}: {s.frames_written} written, {s.frames_dropped} dropped"
              for s in self.sinks],
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
    # Size the output to the map unless told otherwise. A square area of interest forced
    # into 16:9 spends 44% of every frame on black bars, which is what made the earlier feed
    # look like it was only filling the middle of the screen.
    ow, oh = a.out_width, a.out_height
    if ow is None or oh is None:
        longest = 1280.0
        k = longest / max(aoi.width, aoi.height)
        ow = int(round(aoi.width * k / 2) * 2)
        oh = int(round(aoi.height * k / 2) * 2)

    return AppConfig(
        aoi=aoi,
        anchor=anchor,
        default_plane_z=a.plane_z,
        max_incidence_deg=a.max_incidence,
        target_hz=a.hz,
        bind_host=a.bind,
        ir_enabled=a.ir,
        hud=a.hud,
        encoder=EncoderConfig(
            enabled=not a.no_encoder, host=a.out_host, port=a.out_port,
            width=ow, height=oh, fps=int(round(a.hz)),
            bitrate=a.bitrate, ffmpeg=a.ffmpeg, container=a.container,
            # One keyframe every half second rather than every second. A viewer joining a
            # live stream shows decoder warnings until the first keyframe arrives, so a
            # shorter interval halves that window for a negligible bitrate cost.
            fps_gop_divisor=2,
        ),
        tiles_enabled=a.tiles, tiles_host=a.tiles_host, tiles_port=a.tiles_port,
        fit_view=a.fit_view,
        feather=a.feather,
        max_frame_age_s=a.max_age,
        mjpeg=MjpegConfig(
            enabled=a.mjpeg, host=a.mjpeg_host, port=a.mjpeg_port,
            quality=a.mjpeg_quality, width=ow, height=oh,
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extent", type=float, default=1000.0, help="AOI half-size, metres")
    ap.add_argument("--gsd", type=float, default=0.5, help="canvas resolution, m/px")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--anchor", default=None, metavar="LAT,LON[,ALT]")
    ap.add_argument("--plane-z", type=float, default=0.0, dest="plane_z",
                    help="fallback ground elevation (metres) used only when neither the LRF "
                         "nor the AGL probe returns a valid range")
    ap.add_argument("--max-incidence", type=float, default=65.0,
                    help="reject a frame whose worst corner exceeds this angle off nadir")
    ap.add_argument("--ir", action="store_true", help="also composite the IR streams")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--save", default=None, help="write the final mosaic to this PNG")
    ap.add_argument("--stats", action="store_true", help="print the latency table at exit")
    ap.add_argument("--hud", choices=("off", "minimal", "full"), default="minimal",
                    help="minimal = operator view (default); full adds the engineering layer")
    ap.add_argument("--no-encoder", action="store_true", help="disable the H.264 output")
    ap.add_argument("--container", choices=("mpegts", "rtp"), default="mpegts",
                    help="mpegts for ffplay/VLC; rtp for Mission Planner's GStreamer path")
    ap.add_argument("--mjpeg", action="store_true",
                    help="also serve MJPEG over HTTP -- Mission Planner reads this natively "
                         "with nothing installed (Set MJPEG Source)")
    ap.add_argument("--mjpeg-host", default="0.0.0.0")
    ap.add_argument("--mjpeg-port", type=int, default=8080)
    ap.add_argument("--mjpeg-quality", type=int, default=80)
    ap.add_argument("--tiles", action="store_true",
                    help="serve the mosaic as a MAP LAYER (XYZ tiles + WMS) so it appears on "
                         "Mission Planner's map rather than only in the HUD panel. "
                         "Requires --anchor")
    ap.add_argument("--tiles-host", default="0.0.0.0")
    ap.add_argument("--tiles-port", type=int, default=8081)
    ap.add_argument("--max-age", type=float, default=0.5, dest="max_age", metavar="SECONDS",
                    help="ignore frames older than this. Raise it when the aircraft send "
                         "slowly: at 3 Hz frames arrive every 0.33 s, so use 1.5")
    ap.add_argument("--feather", type=float, default=0.3, metavar="F",
                    help="cross-fade width between aircraft, 0 = hard seam. Hides the exposure "
                         "step where two frames meet; does not fix terrain-driven misalignment")
    ap.add_argument("--fit-view", action="store_true",
                    help="zoom the video output to the imaged area instead of showing the whole "
                         "area of interest, so early coverage fills the frame. Does not affect "
                         "the map layer, which stays georeferenced")
    ap.add_argument("--out-host", default="127.0.0.1")
    ap.add_argument("--out-port", type=int, default=5600)
    ap.add_argument("--out-width", type=int, default=None,
                    help="output width; defaults to matching the map aspect so there are no "
                         "black bars")
    ap.add_argument("--out-height", type=int, default=None)
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
