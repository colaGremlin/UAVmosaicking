"""Telemetry and frame containers passed between the ingest threads and the fusion loop.

:class:`Telemetry` holds exactly what arrives on the wire -- raw Unity floats, unconverted.
It exposes the derived quantities (intrinsics, rotation, ENU position) as methods so the
conversion happens in one place, :mod:`uavmosaic.coords`, and never in the sender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Intrinsics
from .coords import unity_pos_to_enu, unity_quat_to_R_enu_cam

__all__ = ["SENSOR_EO", "SENSOR_IR", "SENSOR_NAMES", "Telemetry", "FrameBundle"]

SENSOR_EO = 0
SENSOR_IR = 1
SENSOR_NAMES = {SENSOR_EO: "EO", SENSOR_IR: "IR"}


@dataclass(frozen=True)
class Telemetry:
    """One pose + optics sample, as sent. Nothing here has been converted yet.

    ``pos_unity`` and ``quat_world_cam`` are in Unity's left-handed world frame exactly as
    the C# sender read them off the transform. Keeping them raw means a coordinate bug can
    only ever live in one module.
    """

    pos_unity: tuple[float, float, float]
    quat_world_cam: tuple[float, float, float, float]  #: (x, y, z, w), Unity order
    img_w: int
    img_h: int
    fx: float
    fy: float
    cx: float
    cy: float
    lrf_slant_m: float | None  #: None when the sender flagged the return invalid
    agl_m: float | None
    hfov_deg: float  #: redundant, cross-checked against fx
    zoom: float

    def intrinsics(self, dist: tuple[float, ...] | None = None) -> Intrinsics:
        return Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.img_w,
            height=self.img_h,
            dist=dist,
        )

    def R_enu_cam(self) -> np.ndarray:
        """Rotation mapping CV-camera directions into local ENU."""
        return unity_quat_to_R_enu_cam(*self.quat_world_cam)

    def cam_enu(self, origin_enu=None) -> np.ndarray:
        """Camera centre in local ENU metres."""
        return unity_pos_to_enu(self.pos_unity, origin_enu)

    def fov_disagreement_deg(self) -> float:
        """How far the reported hfov is from the one implied by ``fx``.

        Above ~0.5 deg the sensor size and the Unity camera FOV describe different optics --
        a configuration bug worth catching at frame 1 rather than from a smeared mosaic.
        """
        return abs(self.intrinsics().hfov_deg - self.hfov_deg)


@dataclass(frozen=True)
class FrameBundle:
    """A decoded frame with the pose that was captured in the same Unity frame."""

    uav_id: int
    sensor_id: int
    frame_id: int
    t_capture_us: int
    telemetry: Telemetry
    image: np.ndarray  #: decoded BGR (or grayscale for IR)
    t_received: float  #: local monotonic clock, for staleness

    @property
    def sensor_name(self) -> str:
        return SENSOR_NAMES.get(self.sensor_id, f"S{self.sensor_id}")

    @property
    def key(self) -> tuple[int, int]:
        return (self.uav_id, self.sensor_id)

    def __repr__(self) -> str:  # keeps log lines readable -- never dumps the pixels
        h, w = self.image.shape[:2]
        return (
            f"<FrameBundle uav{self.uav_id}/{self.sensor_name} "
            f"frame={self.frame_id} {w}x{h} t={self.t_capture_us}us>"
        )
