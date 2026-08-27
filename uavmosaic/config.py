"""Runtime configuration. One object, constructed once, never mutated during a run."""

from __future__ import annotations

from dataclasses import dataclass, field

from .coords import CanvasGeometry, GeodeticAnchor
from .frames import SENSOR_EO, SENSOR_IR
from .georef import DEFAULT_CLAMP_FACTOR, DEFAULT_MAX_INCIDENCE_DEG

__all__ = ["AppConfig", "EncoderConfig", "DEFAULT_EO_PORTS", "DEFAULT_IR_PORTS"]

DEFAULT_EO_PORTS = {0: 5001, 1: 5002, 2: 5003, 3: 5004}
DEFAULT_IR_PORTS = {0: 5011, 1: 5012, 2: 5013, 3: 5014}


@dataclass(frozen=True)
class EncoderConfig:
    """H.264 out to Mission Planner via an ffmpeg subprocess."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5600
    width: int = 1280  #: canvas is downscaled to this for the wire
    height: int = 720
    fps: int = 10
    bitrate: str = "4M"
    preset: str = "ultrafast"
    tune: str = "zerolatency"
    ffmpeg: str = "ffmpeg"
    container: str = "mpegts"  #: 'mpegts' plays with ffplay/MP out of the box; 'rtp' needs an SDP


@dataclass(frozen=True)
class AppConfig:
    """Everything the pipeline needs. Defaults describe a 2 km x 2 km AOI at 0.5 m/px."""

    aoi: CanvasGeometry = field(
        default_factory=lambda: CanvasGeometry(
            e_min=-1000.0, n_min=-1000.0, e_max=1000.0, n_max=1000.0, gsd=0.5
        )
    )
    anchor: GeodeticAnchor | None = None  #: set to read targets out as lat/lon

    uav_ids: tuple[int, ...] = (0, 1, 2, 3)
    eo_ports: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_EO_PORTS))
    ir_ports: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_IR_PORTS))
    ir_enabled: bool = False  #: protocol-live, compositing dormant -- flip to switch it on

    bind_host: str = "127.0.0.1"

    target_hz: float = 10.0
    #: A frame older than this is not re-warped. Its pixels stay on the canvas -- the
    #: mosaic is persistent -- and the HUD marks the UAV stale.
    max_frame_age_s: float = 0.5

    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG
    clamp_factor: float = DEFAULT_CLAMP_FACTOR
    allow_lower_half: bool = True
    default_plane_z: float = 0.0

    radial_power: float = 2.0
    incidence_power: float = 2.0
    gsd_power: float = 1.0

    warp_workers: int = 4
    warp_interpolation: str = "linear"  #: 'linear' or 'nearest'

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    hud: bool = True

    def ports_for(self, sensor_id: int) -> dict[int, int]:
        return self.eo_ports if sensor_id == SENSOR_EO else self.ir_ports

    def active_sensors(self) -> tuple[int, ...]:
        return (SENSOR_EO, SENSOR_IR) if self.ir_enabled else (SENSOR_EO,)

    def describe(self) -> str:
        lines = [
            self.aoi.describe(),
            f"UAVs {list(self.uav_ids)}  EO ports {[self.eo_ports[u] for u in self.uav_ids]}",
            f"IR {'ON ' + str([self.ir_ports[u] for u in self.uav_ids]) if self.ir_enabled else 'dormant'}",
            f"fusion {self.target_hz:.1f} Hz, max frame age {self.max_frame_age_s * 1000:.0f} ms",
            f"gates: incidence <= {self.max_incidence_deg:.0f} deg, extent <= {self.clamp_factor:.0f}x nadir diag",
        ]
        if self.encoder.enabled:
            e = self.encoder
            lines.append(f"out: H.264 {e.width}x{e.height}@{e.fps} -> {e.host}:{e.port} ({e.container})")
        if self.anchor:
            lines.append(
                f"anchor: {self.anchor.lat_deg:.6f}, {self.anchor.lon_deg:.6f} @ {self.anchor.alt_m:.1f} m"
            )
        return "\n".join("  " + ln for ln in lines)
