"""The fused canvas: fixed world-anchored buffers plus the max-weight compositing rule.

Four parallel buffers, allocated once and never resized -- allocation inside a 10 Hz loop is
the single biggest source of frame-time jitter in Python::

    color   (H, W, 3) uint8    the mosaic
    weight  (H, W)    float32  best weight seen at each pixel, 0 = never written
    owner   (H, W)    uint8    which UAV owns the pixel, 255 = unowned
    stamp   (H, W)    float32  when it was last written, for staleness shading

The update rule
---------------
::

    same_owner = owner[roi] == uav_id
    update = valid & (same_owner | (w_new > weight[roi]))

The ``same_owner`` clause is **not** present in any of the reference implementations, and
without it the mosaic freezes under a moving aircraft: as a UAV drifts, its own older,
higher-weight pixels outrank its fresh ones and the imagery under it stops updating. A UAV
always overwrites itself; max-weight arbitrates only *between* different UAVs.

The alternative -- decaying the whole weight buffer over time -- was rejected because it
costs a full-canvas multiply every tick and makes the winner depend on tick timing rather
than on view quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .coords import CanvasGeometry

__all__ = ["UNOWNED", "STAT_STRIDE", "Canvas", "CompositeResult"]

UNOWNED = 255  #: owner sentinel for a pixel no UAV has written

#: Subsampling stride for HUD statistics. See :meth:`Canvas.coverage_fraction`.
STAT_STRIDE = 4


@dataclass
class CompositeResult:
    """What one composite call actually changed -- for the HUD and the stats line."""

    uav_id: int
    pixels_written: int
    pixels_in_roi: int
    roi: tuple[int, int, int, int]

    @property
    def write_fraction(self) -> float:
        return self.pixels_written / self.pixels_in_roi if self.pixels_in_roi else 0.0


@dataclass
class CanvasStats:
    composites: int = 0
    pixels_written: int = 0


class Canvas:
    """Fixed-extent world-anchored mosaic buffers.

    Not internally locked: the fusion loop composites all four UAVs from one thread after
    the warps have completed in parallel. Adding a lock here would be pure overhead and
    would encourage a design where two threads write overlapping ROIs, which the max-weight
    rule is not safe under.
    """

    def __init__(self, geom: CanvasGeometry, channels: int = 3) -> None:
        self.geom = geom
        h, w = geom.shape
        self.channels = channels
        self.color = np.zeros((h, w, channels), dtype=np.uint8)
        self.weight = np.zeros((h, w), dtype=np.float32)
        self.owner = np.full((h, w), UNOWNED, dtype=np.uint8)
        self.stamp = np.zeros((h, w), dtype=np.float32)
        self.stats = CanvasStats()

    # -- introspection -----------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.geom.shape

    @property
    def nbytes(self) -> int:
        return int(
            self.color.nbytes + self.weight.nbytes + self.owner.nbytes + self.stamp.nbytes
        )

    def coverage_fraction(self, stride: int = STAT_STRIDE) -> float:
        """Fraction of the AOI that has ever been imaged.

        Subsampled by ``stride`` in each axis. On a 4000x4000 canvas an exact
        ``count_nonzero`` costs 35 ms *per tick*; sampling every 4th pixel costs 2 ms and
        still averages a million samples, which is far more than a percentage readout needs.
        Pass ``stride=1`` when an exact figure actually matters.
        """
        w = self.weight[::stride, ::stride]
        return float(np.count_nonzero(w)) / w.size

    def owner_counts(self, stride: int = STAT_STRIDE) -> dict[int, int]:
        """Pixels owned by each UAV (in sampled units) -- who is contributing what.

        ``np.bincount`` on a subsample, not ``np.unique``: on a full 4000x4000 owner buffer
        ``np.unique`` measured 99 ms and ``bincount`` 30 ms, both per tick. Subsampled
        bincount is ~2 ms. These are HUD proportions, so sampling error is irrelevant.
        """
        flat = np.ascontiguousarray(self.owner[::stride, ::stride]).ravel()
        counts = np.bincount(flat, minlength=256)
        return {i: int(c) for i, c in enumerate(counts) if c and i != UNOWNED}

    def covered_bbox(self, stride: int = STAT_STRIDE):
        """``(x0, y0, x1, y1)`` enclosing every imaged pixel, or None if nothing is imaged.

        Subsampled like the other statistics: this drives a display viewport, so a few pixels
        of slack costs nothing and a full-canvas scan every tick would not be free.
        """
        w = self.weight[::stride, ::stride]
        rows = np.flatnonzero(w.any(axis=1))
        cols = np.flatnonzero(w.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return None
        return (int(cols[0] * stride), int(rows[0] * stride),
                int((cols[-1] + 1) * stride), int((rows[-1] + 1) * stride))

    def clear(self) -> None:
        self.color[:] = 0
        self.weight[:] = 0.0
        self.owner[:] = UNOWNED
        self.stamp[:] = 0.0

    # -- the hot path ------------------------------------------------------------------

    def composite(
        self,
        uav_id: int,
        roi: tuple[int, int, int, int],
        warped_color: np.ndarray,
        warped_weight: np.ndarray,
        t_now: float,
        feather: float = 0.0,
    ) -> CompositeResult:
        """Blend one warped frame into the canvas under the max-weight rule.

        ``warped_color`` and ``warped_weight`` must already be in ROI-local coordinates and
        exactly ``(y1 - y0, x1 - x0)`` in size. ``warped_weight`` is zero wherever the warp
        produced no data, which doubles as the validity mask -- no separate alpha channel
        needed, because :data:`uavmosaic.weights.WEIGHT_FLOOR` guarantees real pixels are
        strictly positive.

        ``feather`` softens the boundary between two aircraft. At 0 the winner takes the pixel
        outright, which is fast but draws a visible line wherever two frames differ in exposure
        or sharpness. Above 0 it cross-fades over a band of that width, expressed as a fraction
        of the incoming frame's peak weight: 0.3 blends where the two weights are within 30% of
        each other and stays a clean hard choice everywhere else. An aircraft still always
        replaces its own pixels outright, so fresh imagery is never diluted by its own stale
        imagery.

        Note what this does and does not fix. It hides the *photometric* seam -- the visible
        step where two exposures meet. It cannot fix *geometric* misregistration, where a river
        appears to kink because the two frames projected it onto a flat plane from different
        angles over real terrain relief. That is inherent to the flat-ground assumption and
        only a surface model removes it; blending merely smears the join.
        """
        x0, y0, x1, y1 = roi
        rh, rw = y1 - y0, x1 - x0
        if warped_weight.shape != (rh, rw):
            raise ValueError(f"weight {warped_weight.shape} does not match roi {(rh, rw)}")
        if warped_color.shape[:2] != (rh, rw):
            raise ValueError(f"color {warped_color.shape[:2]} does not match roi {(rh, rw)}")
        if uav_id == UNOWNED:
            raise ValueError(f"uav_id {UNOWNED} is reserved as the unowned sentinel")

        cur_w = self.weight[y0:y1, x0:x1]
        cur_o = self.owner[y0:y1, x0:x1]

        valid = warped_weight > 0.0

        if feather > 0.0:
            peak = float(warped_weight.max())
            if peak > 0.0:
                # alpha = 0.5 + (w_new - w_old)/band, clipped. Written division-free and in
                # place: the obvious w_new/(w_new+w_old) form measured 7.6 ms on a 1.9 Mpx
                # ROI against 1.7 ms for this, and the two are equivalent after clipping.
                inv_band = np.float32(1.0 / max(feather * peak, 1e-9))
                alpha = np.subtract(warped_weight, cur_w)
                alpha *= inv_band
                alpha += np.float32(0.5)
                np.clip(alpha, 0.0, 1.0, out=alpha)
                alpha *= valid                       # nothing outside the warp
                # Two cases must take the new pixel whole rather than cross-fade:
                #   * canvas never written here -- blending against black would darken every
                #     leading edge of the mosaic. The ground-truth test caught this as a drop
                #     from 0.995 to 0.885 correlation against the source world.
                #   * same aircraft -- fresh imagery is never diluted by its own stale imagery.
                np.putmask(alpha, valid & (cur_w <= 0.0), np.float32(1.0))
                np.putmask(alpha, valid & (cur_o == uav_id), np.float32(1.0))

                inv_alpha = np.subtract(np.float32(1.0), alpha)
                dst = self.color[y0:y1, x0:x1]
                cv2.blendLinear(warped_color, dst, alpha, inv_alpha, dst=dst)

                # Provenance follows the dominant contributor, not the blend.
                update = alpha > 0.5
                np.copyto(cur_w, warped_weight, where=update)
                np.copyto(cur_o, np.uint8(uav_id), where=update)
                np.copyto(self.stamp[y0:y1, x0:x1], np.float32(t_now), where=update)

                n = int(np.count_nonzero(update))
                self.stats.composites += 1
                self.stats.pixels_written += n
                return CompositeResult(
                    uav_id=uav_id, pixels_written=n, pixels_in_roi=rh * rw, roi=roi
                )

        update = valid & ((cur_o == uav_id) | (warped_weight > cur_w))

        # The colour buffer is copied with cv2.copyTo, not numpy. Measured on a 2069x1533
        # ROI, the three ways of doing a masked 3-channel copy differ by two orders of
        # magnitude:
        #
        #   np.copyto(where=mask[:, :, None])    22.88 ms   broadcasts the mask per channel
        #   np.copyto(where=<materialised 3ch>)   1.60 ms   avoids the broadcast
        #   cv2.copyTo(src, mask, dst)            0.28 ms   SIMD, native multi-channel mask
        #
        # `update` is a contiguous bool array, and numpy bool is one byte, so .view(uint8)
        # reinterprets it as OpenCV's 0/255-style mask at zero cost. cv2.copyTo writes
        # through the non-contiguous canvas slice correctly (covered by the canvas tests).
        cv2.copyTo(warped_color, update.view(np.uint8), self.color[y0:y1, x0:x1])
        np.copyto(cur_w, warped_weight, where=update)
        np.copyto(cur_o, np.uint8(uav_id), where=update)
        np.copyto(self.stamp[y0:y1, x0:x1], np.float32(t_now), where=update)

        n = int(np.count_nonzero(update))
        self.stats.composites += 1
        self.stats.pixels_written += n

        return CompositeResult(
            uav_id=uav_id, pixels_written=n, pixels_in_roi=rh * rw, roi=roi
        )

    # -- output ------------------------------------------------------------------------

    def view(self) -> np.ndarray:
        """The colour buffer. A view, not a copy -- the encoder copies once on write."""
        return self.color

    def staleness_seconds(self, t_now: float) -> np.ndarray:
        """Age of every written pixel; ``inf`` where nothing has ever been written."""
        age = t_now - self.stamp
        age[self.weight == 0.0] = np.inf
        return age
