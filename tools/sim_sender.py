"""Stand-in for the Unity fleet: renders four synthetic UAV feeds and streams them over UDP.

Speaks the real wire protocol and transmits **raw Unity-frame values**, so the backend
exercises the genuine LH->RH path rather than being handed pre-converted data. That means the
whole pipeline can be validated -- and demoed -- before the Unity project exists, and that any
disagreement later is in the C# sender, not in the maths.

    python tools/sim_sender.py --duration 20 --hz 10

Run the backend in another terminal:

    python -m uavmosaic.app --duration 20
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import random
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavmosaic.config import DEFAULT_EO_PORTS, AppConfig  # noqa: E402
from uavmosaic.frames import SENSOR_EO, Telemetry  # noqa: E402
from uavmosaic.protocol import DEFAULT_MTU, build_packets  # noqa: E402
from uavmosaic.synth import FlightPlan, make_world, render_view, unity_from_enu_pose  # noqa: E402

log = logging.getLogger("sim_sender")


def telemetry_for(state, jpeg_size_hint=None) -> Telemetry:
    """Build the on-wire telemetry, converting the pose back to raw Unity values."""
    pos_unity, quat = unity_from_enu_pose(state.cam_enu, state.R_enu_cam)
    i = state.intr
    return Telemetry(
        pos_unity=pos_unity,
        quat_world_cam=quat,
        img_w=i.width,
        img_h=i.height,
        fx=i.fx,
        fy=i.fy,
        cx=i.cx,
        cy=i.cy,
        lrf_slant_m=state.lrf_slant_m,
        agl_m=state.agl_m,
        hfov_deg=i.hfov_deg,
        zoom=state.zoom,
    )


class SimSender:
    def __init__(
        self,
        cfg: AppConfig,
        host: str = "127.0.0.1",
        ports: dict[int, int] | None = None,
        hz: float = 10.0,
        jpeg_quality: int = 80,
        mtu: int = DEFAULT_MTU,
        world: np.ndarray | None = None,
        seed: int = 7,
        loss: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.host = host
        self.ports = ports or dict(DEFAULT_EO_PORTS)
        self.hz = hz
        self.jpeg_quality = jpeg_quality
        self.loss = float(loss)
        self._rng = random.Random(20260828)
        self.packets_dropped = 0
        self.mtu = mtu
        self.world = make_world(cfg.aoi, seed) if world is None else world
        self.plan = FlightPlan(cfg.aoi, n_uavs=len(cfg.uav_ids))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        self.frame_ids = {u: 0 for u in cfg.uav_ids}
        self.packets_sent = 0
        self.bytes_sent = 0
        self.frames_sent = 0
        self.frames_skipped = 0

    def send_tick(self, t: float) -> int:
        """Render and transmit one frame per UAV. Returns how many were sent."""
        sent = 0
        for uav_id in self.cfg.uav_ids:
            st = self.plan.state(uav_id, t)
            view = render_view(self.world, self.cfg.aoi, st.intr, st.R_enu_cam, st.cam_enu)
            if view is None:
                self.frames_skipped += 1
                continue

            ok, buf = cv2.imencode(
                ".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                self.frames_skipped += 1
                continue

            self.frame_ids[uav_id] += 1
            packets = build_packets(
                uav_id=uav_id,
                sensor_id=SENSOR_EO,
                frame_id=self.frame_ids[uav_id],
                t_capture_us=int(t * 1e6),
                telemetry=telemetry_for(st),
                jpeg=buf.tobytes(),
                mtu=self.mtu,
            )
            addr = (self.host, self.ports[uav_id])
            for p in packets:
                # Loopback never loses a datagram, so without this the receiver's loss
                # handling is never actually exercised before it meets a real radio.
                if self.loss > 0.0 and self._rng.random() < self.loss:
                    self.packets_dropped += 1
                    continue
                self.sock.sendto(p, addr)
                self.bytes_sent += len(p)
            self.packets_sent += len(packets)
            self.frames_sent += 1
            sent += 1
        return sent

    def run(self, duration: float) -> None:
        period = 1.0 / self.hz
        t_start = time.monotonic()
        n = 0
        log.info(
            "streaming %d UAVs -> %s ports %s at %.1f Hz for %.0fs",
            len(self.cfg.uav_ids), self.host,
            [self.ports[u] for u in self.cfg.uav_ids], self.hz, duration,
        )
        while True:
            t = time.monotonic() - t_start
            if t >= duration:
                break
            self.send_tick(t)
            n += 1
            sleep = (n * period) - (time.monotonic() - t_start)
            if sleep > 0:
                time.sleep(sleep)
        dt = time.monotonic() - t_start
        log.info(
            "sent %d frames / %d packets / %.1f MB in %.1fs (%.1f Hz, %.1f Mb/s)%s",
            self.frames_sent, self.packets_sent, self.bytes_sent / 1e6, dt,
            self.frames_sent / max(dt, 1e-9) / max(len(self.cfg.uav_ids), 1),
            self.bytes_sent * 8 / 1e6 / max(dt, 1e-9),
            f", {self.frames_skipped} skipped" if self.frames_skipped else "",
        )

    def close(self) -> None:
        self.sock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    ap.add_argument("--mtu", type=int, default=DEFAULT_MTU)
    ap.add_argument("--gsd", type=float, default=0.5, help="world resolution, m/px")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loss", type=float, default=0.0, metavar="FRACTION",
                    help="drop this fraction of datagrams, e.g. 0.03 for 3 %%. "
                         "Simulates a radio link; loopback itself never loses any")
    ap.add_argument("--save-world", type=str, default=None, help="write the ground truth PNG")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from uavmosaic.coords import CanvasGeometry

    cfg = AppConfig(
        aoi=CanvasGeometry(e_min=-1000, n_min=-1000, e_max=1000, n_max=1000, gsd=a.gsd)
    )
    sender = SimSender(
        cfg, host=a.host, hz=a.hz, jpeg_quality=a.quality, mtu=a.mtu, seed=a.seed,
        loss=a.loss
    )
    if a.save_world:
        cv2.imwrite(a.save_world, sender.world)
        log.info("ground-truth world -> %s", a.save_world)
    try:
        sender.run(a.duration)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        sender.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
