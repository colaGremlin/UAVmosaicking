"""Serve the live mosaic as a georeferenced map layer, so it sits on the map itself.

Mission Planner's ``Set MJPEG Source`` fills the HUD panel -- the small artificial-horizon
box. Its map is a different thing entirely: a tile-based map control that draws map tiles,
the aircraft icon, waypoints and polygons. It has no concept of a video layer. To get the
mosaic *on the map*, underneath the aircraft icons and in true geographic position, it has to
be served as map tiles.

Two protocols are served from the same endpoint, because different clients ask differently:

``/tiles/{z}/{x}/{y}.png``   slippy-map XYZ tiles in Web Mercator (EPSG:3857). What
                             Mission Planner's custom map source, QGIS, Leaflet and almost
                             every other map client speak.
``/wms?...``                 WMS 1.1.1/1.3.0 GetMap, for Mission Planner's Custom WMS
                             provider and desktop GIS. EPSG:4326 and EPSG:3857 both accepted.

Geometry
--------
The canvas is a local ENU grid; a map client asks in lat/lon or Web Mercator metres. Over one
tile the conversion is indistinguishable from a projective map, so the four requested corners
are converted exactly through :class:`~uavmosaic.coords.GeodeticAnchor`, and the canvas is
sampled through the resulting homography. Error is far below one pixel, and no per-pixel
geodesy runs in the request path.

Uncovered ground is transparent, not black, so the mosaic composites over whatever base map
is underneath instead of blanking it out.
"""

from __future__ import annotations

import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .canvas import Canvas
from .coords import CanvasGeometry, GeodeticAnchor

log = logging.getLogger(__name__)

__all__ = ["TileServer", "mercator_tile_bounds", "parse_wms_bbox"]

TILE = 256
_EARTH_R = 6378137.0
_ORIGIN_SHIFT = math.pi * _EARTH_R


def mercator_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """XYZ tile -> ``(west, south, east, north)`` in degrees."""
    n = 1 << z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def parse_wms_bbox(bbox: list[float], crs: str, version: str) -> tuple[float, float, float, float]:
    """WMS BBOX -> ``(west, south, east, north)`` in degrees.

    Handles the trap that catches most WMS implementations: **WMS 1.3.0 orders EPSG:4326 as
    lat,lon**, while 1.1.1 uses lon,lat. Reading 1.3.0 as though it were 1.1.1 silently places
    the layer in the wrong hemisphere -- it renders, it just lies about where it is.
    """
    crs = (crs or "EPSG:4326").upper()
    if "3857" in crs or "900913" in crs or "102100" in crs:
        w, s = _merc_to_lonlat(bbox[0], bbox[1])
        e, n = _merc_to_lonlat(bbox[2], bbox[3])
        return w, s, e, n
    if crs.endswith("4326") and str(version or "1.1.1").startswith("1.3"):
        return bbox[1], bbox[0], bbox[3], bbox[2]      # lat,lon,lat,lon -> lon,lat,lon,lat
    return bbox[0], bbox[1], bbox[2], bbox[3]


def _merc_to_lonlat(mx: float, my: float) -> tuple[float, float]:
    lon = mx / _ORIGIN_SHIFT * 180.0
    lat = math.degrees(2 * math.atan(math.exp((my / _ORIGIN_SHIFT) * math.pi)) - math.pi / 2)
    return lon, lat


class TileRenderer:
    """Turns a lat/lon bounding box into a transparent-backed PNG cut from the canvas."""

    def __init__(self, canvas: Canvas, geom: CanvasGeometry, anchor: GeodeticAnchor) -> None:
        self.canvas = canvas
        self.geom = geom
        self.anchor = anchor

    def _corners_to_canvas(self, w: float, s: float, e: float, n: float) -> np.ndarray:
        """The four bbox corners, in canvas pixels. Order: NW, NE, SE, SW."""
        pts = []
        for lat, lon in ((n, w), (n, e), (s, e), (s, w)):
            enu = self.anchor.geodetic_to_enu(lat, lon, 0.0)
            px, py = self.geom.enu_to_px(enu[0], enu[1])
            pts.append((float(px), float(py)))
        return np.array(pts, dtype=np.float32)

    def render(self, w, s, e, n, width=TILE, height=TILE) -> np.ndarray:
        """BGRA image of that bbox. Fully transparent where the mosaic has no coverage."""
        out = np.zeros((height, width, 4), dtype=np.uint8)
        src = self._corners_to_canvas(w, s, e, n)

        cw, ch = self.geom.width, self.geom.height
        if (src[:, 0].max() <= 0 or src[:, 0].min() >= cw
                or src[:, 1].max() <= 0 or src[:, 1].min() >= ch):
            return out  # request lies entirely off the canvas

        # Degenerate at extreme zoom-out, where a whole tile collapses to a few canvas pixels.
        if (src[:, 0].max() - src[:, 0].min() < 1e-3
                or src[:, 1].max() - src[:, 1].min() < 1e-3):
            return out

        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        colour = cv2.warpPerspective(
            self.canvas.color, H, (width, height),
            flags=cv2.INTER_AREA if width < cw else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # The weight buffer doubles as the coverage mask: zero means never imaged.
        cover = cv2.warpPerspective(
            (self.canvas.weight > 0).astype(np.uint8) * 255, H, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        out[:, :, :3] = colour
        out[:, :, 3] = cover
        return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "uavmosaic-tiles/1.0"

    def log_message(self, fmt, *args):
        log.debug("tiles %s - %s", self.address_string(), fmt % args)

    def _png(self, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            self.send_error(500, "encode failed")
            return
        data = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        srv: TileServer = self.server.owner  # type: ignore[attr-defined]
        u = urlparse(self.path)
        path = u.path

        try:
            if path in ("/", "/index.html"):
                self._page(srv)
                return

            if path.startswith("/tiles/"):
                parts = path[len("/tiles/"):].split("/")
                if len(parts) != 3:
                    self.send_error(404, "expected /tiles/{z}/{x}/{y}.png")
                    return
                z = int(parts[0]); x = int(parts[1]); y = int(parts[2].split(".")[0])
                if not (0 <= z <= 24):
                    self.send_error(404, "zoom out of range")
                    return
                w, s, e, n = mercator_tile_bounds(z, x, y)
                srv.tiles_served += 1
                self._png(srv.renderer.render(w, s, e, n))
                return

            if path == "/wms":
                self._wms(srv, parse_qs(u.query))
                return

            self.send_error(404, "try /tiles/{z}/{x}/{y}.png or /wms")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except ValueError as exc:
            self.send_error(400, f"bad request: {exc}")
        except Exception as exc:  # a bad tile request must never take the server down
            log.exception("tile request failed: %s", exc)
            try:
                self.send_error(500, "render failed")
            except OSError:
                pass

    def _wms(self, srv: "TileServer", q: dict) -> None:
        low = {k.lower(): v[0] for k, v in q.items()}
        req = low.get("request", "").lower()

        if req == "getcapabilities":
            self._capabilities(srv)
            return
        if req and req != "getmap":
            self.send_error(400, f"unsupported WMS request {req!r}")
            return

        bbox = [float(v) for v in low.get("bbox", "").split(",")] if low.get("bbox") else None
        if not bbox or len(bbox) != 4:
            self.send_error(400, "BBOX required as minx,miny,maxx,maxy")
            return
        width = int(low.get("width", TILE))
        height = int(low.get("height", TILE))
        w, s, e, n = parse_wms_bbox(
            bbox, low.get("srs") or low.get("crs") or "EPSG:4326", low.get("version", "1.1.1")
        )
        srv.wms_served += 1
        self._png(srv.renderer.render(w, s, e, n, width, height))

    def _capabilities(self, srv: "TileServer") -> None:
        b = srv.layer_bounds_deg()
        # xmlns:xlink MUST be declared. Without it the document has an unbound prefix on
        # every xlink:href, which is not well-formed XML -- Mission Planner loads this with
        # XmlDocument and would reject the server outright. Caught by
        # tests/test_mission_planner_wms.py, which replays MP's own parser.
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<WMT_MS_Capabilities version="1.1.1" xmlns:xlink="http://www.w3.org/1999/xlink">
 <Service><Name>OGC:WMS</Name><Title>UAV Mosaic</Title>
  <OnlineResource xlink:href="http://{srv.advertise_host}:{srv.port}/wms"/></Service>
 <Capability><Request><GetMap>
   <Format>image/png</Format>
   <DCPType><HTTP><Get><OnlineResource
     xlink:href="http://{srv.advertise_host}:{srv.port}/wms?"/></Get></HTTP></DCPType>
  </GetMap></Request>
  <Layer>
   <Title>UAV Mosaic</Title>
   <Layer queryable="0">
    <Name>mosaic</Name><Title>Live UAV mosaic</Title>
    <SRS>EPSG:4326</SRS><SRS>EPSG:3857</SRS>
    <LatLonBoundingBox minx="{b[0]:.6f}" miny="{b[1]:.6f}"
                       maxx="{b[2]:.6f}" maxy="{b[3]:.6f}"/>
   </Layer>
  </Layer>
 </Capability>
</WMT_MS_Capabilities>"""
        data = xml.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.ogc.wms_xml")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _page(self, srv: "TileServer") -> None:
        b = srv.layer_bounds_deg()
        clat, clon = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
        z = srv.native_zoom()
        n = 1 << z
        tx = int((clon + 180.0) / 360.0 * n)
        lat_r = math.radians(clat)
        ty = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
        body = f"""<!doctype html><meta charset=utf-8><title>UAV Mosaic tiles</title>
<style>body{{background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;margin:0;padding:18px}}
code{{background:#161b22;padding:2px 6px;border-radius:3px;font-size:13px}}
table{{border-collapse:collapse;margin:14px 0}}td,th{{padding:5px 12px;text-align:left;
border-bottom:1px solid #21262d;font-size:14px}}img{{border:1px solid #30363d;background:#000}}
h2{{font-size:15px;margin:18px 0 6px}}</style>
<h2>Live mosaic, served as a map layer</h2>
<table>
<tr><th>XYZ tiles</th><td><code>http://{srv.advertise_host}:{srv.port}/tiles/{{z}}/{{x}}/{{y}}.png</code></td></tr>
<tr><th>WMS</th><td><code>http://{srv.advertise_host}:{srv.port}/wms</code> &mdash; layer <code>mosaic</code></td></tr>
<tr><th>Coverage</th><td>{b[1]:.5f}, {b[0]:.5f} &rarr; {b[3]:.5f}, {b[2]:.5f}</td></tr>
<tr><th>Native zoom</th><td>z{z} &nbsp; (canvas {srv.geom.gsd:.2f} m/px)</td></tr>
</table>
<h2>Centre tile at native zoom &mdash; if this shows imagery, the tiles are correct</h2>
<img src="/tiles/{z}/{tx}/{ty}.png" width="256" height="256" alt="centre tile">
<img src="/tiles/{z}/{tx+1}/{ty}.png" width="256" height="256" alt="tile east">
<img src="/tiles/{z}/{tx}/{ty+1}.png" width="256" height="256" alt="tile south">
<img src="/tiles/{z}/{tx+1}/{ty+1}.png" width="256" height="256" alt="tile southeast">
<p style="font-size:13px;color:#8b949e">Transparent areas are ground not yet imaged. Reload to update.</p>"""
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class TileServer:
    """HTTP server publishing the canvas as XYZ tiles and WMS."""

    def __init__(
        self,
        canvas: Canvas,
        geom: CanvasGeometry,
        anchor: GeodeticAnchor,
        host: str = "0.0.0.0",
        port: int = 8081,
    ) -> None:
        self.canvas = canvas
        self.geom = geom
        self.anchor = anchor
        self.host = host
        self.port = port
        self.renderer = TileRenderer(canvas, geom, anchor)
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self.tiles_served = 0
        self.wms_served = 0

    @property
    def advertise_host(self) -> str:
        return "127.0.0.1" if self.host in ("0.0.0.0", "", "::") else self.host

    def layer_bounds_deg(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` of the AOI, in degrees."""
        lats, lons = [], []
        for e, n in ((self.geom.e_min, self.geom.n_min), (self.geom.e_min, self.geom.n_max),
                     (self.geom.e_max, self.geom.n_min), (self.geom.e_max, self.geom.n_max)):
            lat, lon, _ = self.anchor.enu_to_geodetic(e, n, 0.0)
            lats.append(lat); lons.append(lon)
        return min(lons), min(lats), max(lons), max(lats)

    def native_zoom(self) -> int:
        """The Web Mercator zoom whose resolution best matches the canvas."""
        lat = self.anchor.lat_deg
        for z in range(0, 23):
            res = 156543.03392 * math.cos(math.radians(lat)) / (1 << z)
            if res <= self.geom.gsd:
                return z
        return 22

    def start(self) -> None:
        self._server = _Server((self.host, self.port), _Handler)
        self._server.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="tiles", daemon=True
        )
        self._thread.start()
        b = self.layer_bounds_deg()
        # Written as instructions, not as a line to copy. An earlier version printed the URL
        # with a trailing "(native z16)" and it got pasted into a shell, which of course failed.
        host, port = self.advertise_host, self.port
        log.info("MAP LAYER IS UP.")
        log.info("  Step 1  check it in a browser:  http://%s:%d/", host, port)
        log.info("  Step 2  in Mission Planner, Flight Plan screen, set the map dropdown to WMS")
        log.info("  Step 3  when it asks for the server, give it exactly this address:")
        log.info("              http://%s:%d/wms", host, port)
        log.info("  Step 4  it then lists the layers. Choose the one named: mosaic")
        log.info("  The layer covers latitude %.5f to %.5f, longitude %.5f to %.5f.",
                 b[1], b[3], b[0], b[2])
        log.info("  It is sharpest at map zoom %d; other zooms are resampled from the canvas.",
                 self.native_zoom())

    def stop(self, timeout: float = 2.0) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("tile server stopped (%d tiles, %d WMS requests)",
                 self.tiles_served, self.wms_served)
