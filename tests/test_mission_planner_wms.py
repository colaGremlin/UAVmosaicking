"""Conformance test against Mission Planner's actual WMS client.

Not a guess at what Mission Planner wants. These requests and checks are transcribed from its
source, which is on this machine:

    GCSViews/FlightPlanner.cs
        BuildGetCapabilitityRequest()      appends version=1.1.0&Request=GetCapabilities&service=WMS
        ProcessWmsCapabilitesRequest()     requires //WMT_MS_Capabilities, exactly one //GetMap,
                                           a //Format containing image/png, a //SRS containing
                                           EPSG:4326, and enumerates //Layer/Layer for Name/Title
    ExtLibs/Maps/WMSProvider.cs
        MakeTileImageUrl()                 VERSION=1.1.1&REQUEST=GetMap&SERVICE=WMS&layers=...
                                           &styles=&bbox=w,s,e,n&width=&height=&srs=EPSG:4326
                                           &format=image/png

If any of these fail, Mission Planner shows a message box and refuses the server, so a test
that only checked "a PNG came back" would pass while the integration was broken.
"""

import threading
import urllib.request
from xml.etree import ElementTree

import numpy as np
import pytest

from uavmosaic.canvas import Canvas
from uavmosaic.coords import CanvasGeometry, GeodeticAnchor
from uavmosaic.tiles import TileServer

GEOM = CanvasGeometry(e_min=-4000, n_min=-4000, e_max=4000, n_max=4000, gsd=2.5)
ANCHOR = GeodeticAnchor(lat_deg=33.6844, lon_deg=73.0479, alt_m=540.0)


@pytest.fixture(scope="module")
def server():
    canvas = Canvas(GEOM)
    h, w = GEOM.shape
    canvas.composite(0, (0, 0, w, h), np.full((h, w, 3), (60, 180, 90), np.uint8),
                     np.ones((h, w), np.float32), t_now=0.0)
    srv = TileServer(canvas, GEOM, ANCHOR, host="127.0.0.1", port=8099)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def build_capability_request(server_url: str) -> str:
    """Transcribed from FlightPlanner.BuildGetCapabilitityRequest."""
    if "?" not in server_url:
        server_url += "?"
    elif not server_url.endswith("?"):
        server_url += "&"
    return server_url + "version=1.1.0&Request=GetCapabilities&service=WMS"


def test_mission_planner_accepts_our_capabilities(server):
    """Runs Mission Planner's own five acceptance checks. All must pass."""
    url = build_capability_request(f"http://127.0.0.1:{server.port}/wms")
    status, ctype, body = _get(url)
    assert status == 200, f"Mission Planner would see HTTP {status}"

    root = ElementTree.fromstring(body)

    # 1. must be a WMT_MS_Capabilities document, else MP bails out early
    assert root.tag.endswith("WMT_MS_Capabilities"), f"root tag is {root.tag!r}"

    # 2. exactly one GetMap element, or MP shows "Invalid number of GetMap elements"
    getmaps = root.findall(".//GetMap")
    assert len(getmaps) == 1, f"found {len(getmaps)} GetMap elements, MP requires exactly 1"

    # 3. a Format containing image/png, or "Server unable to return PNG images"
    formats = [f.text or "" for f in root.findall(".//Format")]
    assert any("image/png" in f for f in formats), f"no image/png in {formats}"

    # 4. an SRS containing EPSG:4326
    srs = [s.text or "" for s in root.findall(".//SRS")]
    assert any("EPSG:4326" in s for s in srs), f"no EPSG:4326 in {srs}"

    # 5. //Layer/Layer entries carrying a Name, which is what MP offers you to pick
    layers = root.findall(".//Layer/Layer")
    names = [ln.findtext("Name") for ln in layers if ln.find("Name") is not None]
    assert names, "MP found no selectable layer"
    assert "mosaic" in names, f"expected a 'mosaic' layer, got {names}"


def test_the_exact_getmap_url_mission_planner_builds(server):
    """Byte-for-byte the request from WMSProvider.MakeTileImageUrl."""
    w, s, e, n = server.layer_bounds_deg()
    # MP passes p1 = bottom-left then p2 = top-right, i.e. lon,lat,lon,lat
    url = (
        f"http://127.0.0.1:{server.port}/wms?"
        f"VERSION=1.1.1&REQUEST=GetMap&SERVICE=WMS&layers=mosaic"
        f"&styles=&bbox={w},{s},{e},{n}&width=256&height=256"
        f"&srs=EPSG:4326&format=image/png"
    )
    status, ctype, body = _get(url)
    assert status == 200
    assert "image/png" in ctype, f"MP expects PNG, got {ctype!r}"

    import cv2
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None, "Mission Planner would fail to decode this"
    assert img.shape[:2] == (256, 256), f"asked for 256x256, got {img.shape[:2]}"
    assert img.shape[2] == 4, "needs an alpha channel so uncovered ground stays transparent"
    assert img[:, :, 3].max() == 255, "the covered canvas should come back opaque"


def test_getmap_without_a_layer_parameter_also_works(server):
    """MP omits `layers=` when no layer was selected. That path must not 400."""
    w, s, e, n = server.layer_bounds_deg()
    url = (
        f"http://127.0.0.1:{server.port}/wms?"
        f"VERSION=1.1.1&REQUEST=GetMap&SERVICE=WMS"
        f"&styles=&bbox={w},{s},{e},{n}&width=256&height=256"
        f"&srs=EPSG:4326&format=image/png"
    )
    status, ctype, _ = _get(url)
    assert status == 200 and "image/png" in ctype


def test_tiles_outside_coverage_return_transparent_png_not_an_error(server):
    """MP walks the whole visible map. Tiles off the edge must be quiet, not 500s."""
    url = (
        f"http://127.0.0.1:{server.port}/wms?"
        f"VERSION=1.1.1&REQUEST=GetMap&SERVICE=WMS&layers=mosaic"
        f"&styles=&bbox=-70.1,40.0,-70.0,40.1&width=256&height=256"
        f"&srs=EPSG:4326&format=image/png"
    )
    status, ctype, body = _get(url)
    assert status == 200 and "image/png" in ctype
    import cv2
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img[:, :, 3].max() == 0


def test_server_survives_a_burst_of_concurrent_tile_requests(server):
    """MP requests many tiles at once when you pan. One slow or failed tile must not stall it."""
    results = []

    def fetch(i):
        try:
            st, _, _ = _get(f"http://127.0.0.1:{server.port}/tiles/16/{46060 + i}/26248.png")
            results.append(st)
        except Exception as exc:  # noqa: BLE001
            results.append(exc)

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert len(results) == 12, f"only {len(results)} of 12 requests returned"
    assert all(r == 200 for r in results), f"non-200 responses: {results}"
