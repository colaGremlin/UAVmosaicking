"""The fusion loop: snapshot -> georeference -> warp in parallel -> composite -> publish.

Runs at a fixed tick. Per tick it takes the freshest frame from each UAV's mailbox, drops
anything stale, projects and warps the survivors concurrently (``cv2.warpPerspective``
releases the GIL, so the pool is genuinely parallel), then composites them serially into the
canvas under the max-weight rule.

Compositing is deliberately single-threaded. The max-weight rule is a read-modify-write on
shared pixels, and two threads writing overlapping ROIs would race. It is also cheap -- a
vectorised ``np.where``-style masked assignment -- so there is nothing to gain.

Per-stage timings are recorded as p50/p90/p99, mirroring the GROMS Table 6 format, so a
regression shows up as a number rather than as a feeling that it got slower.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Intrinsics
from .canvas import Canvas
from .config import AppConfig
from .frames import SENSOR_EO, FrameBundle
from .georef import Footprint, GeorefError, compute_footprint, solve_ground_plane
from .ingest import IngestGroup
from .weights import frame_weight_map

log = logging.getLogger(__name__)

__all__ = ["StageTimer", "WarpProduct", "FusionEngine", "FusionStats"]

_INTERP = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}


class StageTimer:
    """Rolling per-stage latency percentiles, in the shape GROMS reports them."""

    def __init__(self, window: int = 300) -> None:
        self._d: dict[str, deque] = {}
        self._window = window
        self._lock = threading.Lock()

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._d.setdefault(stage, deque(maxlen=self._window)).append(seconds * 1000.0)

    def percentiles(self) -> dict[str, tuple[float, float, float, float]]:
        """``{stage: (mean, p50, p90, p99)}`` in milliseconds."""
        with self._lock:
            snap = {k: list(v) for k, v in self._d.items() if v}
        return {
            k: (
                float(np.mean(v)),
                float(np.percentile(v, 50)),
                float(np.percentile(v, 90)),
                float(np.percentile(v, 99)),
            )
            for k, v in snap.items()
        }

    def table(self) -> str:
        rows = self.percentiles()
        if not rows:
            return "(no timings yet)"
        w = max(len(k) for k in rows)
        out = [f"{'stage'.ljust(w)}   mean    p50    p90    p99   (ms)"]
        for k, (m, p50, p90, p99) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            out.append(f"{k.ljust(w)} {m:6.2f} {p50:6.2f} {p90:6.2f} {p99:6.2f}")
        return "\n".join(out)


@dataclass
class WarpProduct:
    """One frame projected and resampled into its canvas ROI, ready to composite."""

    uav_id: int
    roi: tuple[int, int, int, int]
    color: np.ndarray
    weight: np.ndarray
    footprint: Footprint
    frame: FrameBundle


@dataclass
class FusionStats:
    ticks: int = 0
    frames_fused: int = 0
    frames_rejected: int = 0
    frames_outside_aoi: int = 0
    overruns: int = 0  #: ticks that ran past their budget
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def note_reject(self, reason: str) -> None:
        key = reason.split(" -- ")[-1][:60]
        self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1


class FusionEngine:
    """Owns the canvas and the tick loop."""

    def __init__(
        self,
        cfg: AppConfig,
        ingest: IngestGroup,
        canvas: Canvas | None = None,
        sensor_id: int = SENSOR_EO,
    ) -> None:
        self.cfg = cfg
        self.ingest = ingest
        self.sensor_id = sensor_id
        self.canvas = canvas if canvas is not None else Canvas(cfg.aoi)
        self.timer = StageTimer()
        self.stats = FusionStats()
        self._pool = ThreadPoolExecutor(
            max_workers=cfg.warp_workers, thread_name_prefix="warp"
        )
        self._interp = _INTERP[cfg.warp_interpolation]
        self._last_footprints: dict[int, Footprint] = {}
        self._stale: dict[int, bool] = {}

    # -- one frame ---------------------------------------------------------------------

    def _prepare(self, frame: FrameBundle) -> WarpProduct | None:
        """Georeference and warp one frame. Runs in the pool; touches no shared state."""
        t = frame.telemetry
        intr: Intrinsics = t.intrinsics()
        R = t.R_enu_cam()
        cam = t.cam_enu()

        plane = solve_ground_plane(
            cam, R,
            lrf_slant_m=t.lrf_slant_m,
            agl_m=t.agl_m,
            default_z=self.cfg.default_plane_z,
        )

        try:
            fp = compute_footprint(
                intr, R, cam, plane, self.cfg.aoi,
                max_incidence_deg=self.cfg.max_incidence_deg,
                clamp_factor=self.cfg.clamp_factor,
                allow_lower_half=self.cfg.allow_lower_half,
            )
        except GeorefError as exc:
            self.stats.frames_rejected += 1
            self.stats.note_reject(str(exc))
            return None

        roi = fp.canvas_roi(self.canvas.shape)
        if roi is None:
            self.stats.frames_outside_aoi += 1
            return None

        x0, y0, x1, y1 = roi
        size = (x1 - x0, y1 - y0)
        H = fp.homography_to_roi(roi)

        src = frame.image
        if fp.used_lower_half:
            src = src[src.shape[0] // 2 :]

        color = cv2.warpPerspective(
            src, H, size, flags=self._interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        w_src = frame_weight_map(
            intr, R, fp.gsd_m_per_px, self.cfg.aoi.gsd,
            radial_power=self.cfg.radial_power,
            incidence_power=self.cfg.incidence_power,
            gsd_power=self.cfg.gsd_power,
        )
        if fp.used_lower_half:
            w_src = w_src[w_src.shape[0] // 2 :]

        # BORDER_CONSTANT 0 makes the weight zero wherever the warp had no source data,
        # which doubles as the validity mask -- no separate alpha channel needed.
        weight = cv2.warpPerspective(
            w_src, H, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return WarpProduct(
            uav_id=frame.uav_id, roi=roi, color=color, weight=weight, footprint=fp, frame=frame
        )

    # -- one tick ----------------------------------------------------------------------

    def tick(self, now: float | None = None) -> int:
        """Fuse one round. Returns how many frames made it onto the canvas."""
        t_tick = time.perf_counter()
        now = time.monotonic() if now is None else now

        t0 = time.perf_counter()
        fresh = self.ingest.snapshot(self.sensor_id, self.cfg.max_frame_age_s, now=now)
        self.timer.record("snapshot", time.perf_counter() - t0)

        for uav_id in self.cfg.uav_ids:
            self._stale[uav_id] = uav_id not in fresh

        if not fresh:
            self.stats.ticks += 1
            return 0

        t0 = time.perf_counter()
        products = [
            p for p in self._pool.map(self._prepare, fresh.values()) if p is not None
        ]
        self.timer.record("georef+warp", time.perf_counter() - t0)

        t0 = time.perf_counter()
        for p in products:
            self.canvas.composite(p.uav_id, p.roi, p.color, p.weight, now)
            self._last_footprints[p.uav_id] = p.footprint
        self.timer.record("composite", time.perf_counter() - t0)

        self.stats.ticks += 1
        self.stats.frames_fused += len(products)

        elapsed = time.perf_counter() - t_tick
        self.timer.record("tick_total", elapsed)
        if elapsed > 1.0 / self.cfg.target_hz:
            self.stats.overruns += 1
        return len(products)

    # -- introspection -----------------------------------------------------------------

    @property
    def footprints(self) -> dict[int, Footprint]:
        """Most recent accepted footprint per UAV, for the HUD overlay."""
        return dict(self._last_footprints)

    def is_stale(self, uav_id: int) -> bool:
        return self._stale.get(uav_id, True)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def summary(self) -> str:
        s = self.stats
        return (
            f"ticks={s.ticks} fused={s.frames_fused} rejected={s.frames_rejected} "
            f"outside_aoi={s.frames_outside_aoi} overruns={s.overruns} "
            f"coverage={self.canvas.coverage_fraction() * 100:.1f}%"
        )
