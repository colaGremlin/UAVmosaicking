"""Map-layer tests. The thing that matters is geographic correctness, not that a PNG appears.

A tile layer that renders but sits in the wrong place is worse than no tile layer: it looks
authoritative and it lies. So these tests plant a marker at a known lat/lon and check it comes
back at the right pixel, check the tile grid is self-consistent, and check the WMS 1.3.0
axis-order trap that puts layers in the wrong hemisphere.
"""

import math

import numpy as np
import pytest

from uavmosaic.canvas import Canvas
from uavmosaic.coords import CanvasGeometry, GeodeticAnchor
from uavmosaic.tiles import TILE, TileRenderer, TileServer, mercator_tile_bounds

GEOM = CanvasGeometry(e_min=-4000, n_min=-4000, e_max=4000, n_max=4000, gsd=2.0)
ANCHOR = GeodeticAnchor(lat_deg=33.6844, lon_deg=73.0479, alt_m=540.0)


def filled_canvas(colour=(40, 200, 90)):
    """A canvas covered everywhere, so coverage never confuses a geometry failure."""
    c = Canvas(GEOM)
    h, w = GEOM.shape
    c.composite(0, (0, 0, w, h), np.full((h, w, 3), colour, np.uint8),
                np.ones((h, w), np.float32), t_now=0.0)
    return c


def renderer(canvas=None):
    return TileRenderer(canvas or filled_canvas(), GEOM, ANCHOR)


# --------------------------------------------------------------------------------------
# Tile grid maths
# --------------------------------------------------------------------------------------


def test_tile_bounds_cover_the_world_at_zoom_zero():
    w, s, e, n = mercator_tile_bounds(0, 0, 0)
    assert (w, e) == (-180.0, 180.0)
    assert n == pytest.approx(85.051129, abs=1e-4)
    assert s == pytest.approx(-85.051129, abs=1e-4)


def test_tiles_tile_without_gaps_or_overlap():
    """Neighbouring tiles must share an edge exactly, or the mosaic will show seams."""
    z = 14
    for x in (8000, 8001):
        for y in (5000, 5001):
            w, s, e, n = mercator_tile_bounds(z, x, y)
            we, se, ee, ne = mercator_tile_bounds(z, x + 1, y)
            assert e == pytest.approx(we, abs=1e-12), "east edge must meet the next tile west edge"
            wd, sd, ed, nd = mercator_tile_bounds(z, x, y + 1)
            assert s == pytest.approx(nd, abs=1e-12), "south edge must meet the next tile north edge"


def test_native_zoom_matches_canvas_resolution():
    srv = TileServer(filled_canvas(), GEOM, ANCHOR)
    z = srv.native_zoom()
    res = 156543.03392 * math.cos(math.radians(ANCHOR.lat_deg)) / (1 << z)
    assert res <= GEOM.gsd, "native zoom must be at least as fine as the canvas"
    coarser = 156543.03392 * math.cos(math.radians(ANCHOR.lat_deg)) / (1 << (z - 1))
    assert coarser > GEOM.gsd, "and should be the coarsest zoom that still qualifies"


# --------------------------------------------------------------------------------------
# Geographic correctness -- the tests that actually matter
# --------------------------------------------------------------------------------------


def test_a_marker_lands_at_its_true_latlon():
    """Plant a marker at a known ENU point, request the tile containing its lat/lon, and
    check the marker appears at the pixel the map projection predicts."""
    canvas = Canvas(GEOM)
    h, w = GEOM.shape
    canvas.composite(0, (0, 0, w, h), np.zeros((h, w, 3), np.uint8),
                     np.ones((h, w), np.float32), t_now=0.0)

    mark_e, mark_n = 1200.0, -800.0
    mx, my = GEOM.enu_to_px(mark_e, mark_n)
    canvas.color[int(my) - 12 : int(my) + 12, int(mx) - 12 : int(mx) + 12] = (0, 0, 255)

    lat, lon, _ = ANCHOR.enu_to_geodetic(mark_e, mark_n, 0.0)

    z = 15
    n_tiles = 1 << z
    tx = int((lon + 180.0) / 360.0 * n_tiles)
    ty = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n_tiles)

    img = renderer(canvas).render(*mercator_tile_bounds(z, tx, ty))
    red = (img[:, :, 2] > 150) & (img[:, :, 0] < 100) & (img[:, :, 3] > 0)
    assert red.any(), "the marker did not appear in the tile containing its coordinates"

    # Where the map projection says it should be inside that tile
    w_deg, s_deg, e_deg, n_deg = mercator_tile_bounds(z, tx, ty)
    fx = (lon - w_deg) / (e_deg - w_deg) * TILE
    ymerc = lambda d: math.asinh(math.tan(math.radians(d)))
    fy = (ymerc(n_deg) - ymerc(lat)) / (ymerc(n_deg) - ymerc(s_deg)) * TILE

    ys, xs = np.nonzero(red)
    assert abs(xs.mean() - fx) < 4.0, f"marker x off by {abs(xs.mean() - fx):.1f} px"
    assert abs(ys.mean() - fy) < 4.0, f"marker y off by {abs(ys.mean() - fy):.1f} px"


def test_north_is_up_in_a_rendered_tile():
    """A north-south gradient must come back dark at the top, not flipped."""
    canvas = Canvas(GEOM)
    h, w = GEOM.shape
    grad = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    canvas.composite(0, (0, 0, w, h), np.dstack([grad] * 3),
                     np.ones((h, w), np.float32), t_now=0.0)
    # canvas row 0 is the NORTH edge and is dark, so a tile must be dark at its top
    img = renderer(canvas).render(*_centre_tile_bounds())
    top = img[:20, :, 0][img[:20, :, 3] > 0]
    bot = img[-20:, :, 0][img[-20:, :, 3] > 0]
    assert top.size and bot.size
    assert top.mean() < bot.mean(), "north/south is flipped in the tile output"


def _centre_tile_bounds():
    srv = TileServer(filled_canvas(), GEOM, ANCHOR)
    z = srv.native_zoom()
    n = 1 << z
    tx = int((ANCHOR.lon_deg + 180.0) / 360.0 * n)
    ty = int((1.0 - math.asinh(math.tan(math.radians(ANCHOR.lat_deg))) / math.pi) / 2.0 * n)
    return mercator_tile_bounds(z, tx, ty)


def test_layer_bounds_contain_the_anchor():
    srv = TileServer(filled_canvas(), GEOM, ANCHOR)
    w, s, e, n = srv.layer_bounds_deg()
    assert w < ANCHOR.lon_deg < e
    assert s < ANCHOR.lat_deg < n
    # 8 km box near 34 deg N is about 0.072 deg of latitude
    assert 0.05 < (n - s) < 0.10, f"latitude span {n - s:.4f} deg looks wrong for an 8 km box"


# --------------------------------------------------------------------------------------
# Coverage and transparency
# --------------------------------------------------------------------------------------


def test_uncovered_ground_is_transparent_not_black():
    """Uncovered ground must let the base map show through, not blank it out."""
    canvas = Canvas(GEOM)  # nothing ever composited
    img = renderer(canvas).render(*_centre_tile_bounds())
    assert img.shape == (TILE, TILE, 4)
    assert img[:, :, 3].max() == 0, "empty canvas must be fully transparent"


def test_covered_ground_is_opaque():
    img = renderer().render(*_centre_tile_bounds())
    assert img[:, :, 3].min() == 255


def test_partial_coverage_gives_partial_alpha():
    """Half the AOI imaged -> that half opaque, the other transparent, in the right half."""
    canvas = Canvas(GEOM)
    h, w = GEOM.shape
    # canvas row 0 is the NORTH edge, so this images the northern half
    canvas.composite(0, (0, 0, w, h // 2), np.full((h // 2, w, 3), 200, np.uint8),
                     np.ones((h // 2, w), np.float32), t_now=0.0)

    w_deg, s_deg, e_deg, n_deg = TileServer(canvas, GEOM, ANCHOR).layer_bounds_deg()
    img = renderer(canvas).render(w_deg, s_deg, e_deg, n_deg, 128, 128)
    a = img[:, :, 3]
    assert a[:40].mean() > 200, "northern half should be opaque"
    assert a[-40:].mean() < 40, "southern half should be transparent"


def test_tile_far_from_the_aoi_is_empty_not_an_error():
    r = renderer()
    img = r.render(-70.0, 40.0, -69.9, 40.1)  # somewhere over the Atlantic
    assert img[:, :, 3].max() == 0


def test_degenerate_request_does_not_raise():
    r = renderer()
    for bbox in [(0, 0, 0, 0), (73.0, 33.0, 73.0, 33.1), (180, 85, -180, -85)]:
        img = r.render(*bbox)
        assert img.shape == (TILE, TILE, 4)


# --------------------------------------------------------------------------------------
# WMS axis order -- the classic wrong-hemisphere bug
# --------------------------------------------------------------------------------------


def test_wms_axis_order_is_handled_per_version():
    """The 1.3.0 lat,lon vs 1.1.1 lon,lat trap, tested on the parser itself.

    Getting this wrong does not error -- it renders the layer in the wrong hemisphere, which
    is precisely why it needs a test rather than a comment.
    """
    from uavmosaic.tiles import parse_wms_bbox

    lon_lat = [73.0, 33.6, 73.1, 33.7]                # west, south, east, north
    lat_lon = [33.6, 73.0, 33.7, 73.1]                # what 1.3.0 sends for the same box

    assert parse_wms_bbox(lon_lat, "EPSG:4326", "1.1.1") == (73.0, 33.6, 73.1, 33.7)
    assert parse_wms_bbox(lat_lon, "EPSG:4326", "1.3.0") == (73.0, 33.6, 73.1, 33.7)
    assert parse_wms_bbox(lon_lat, "CRS:84", "1.3.0") == (73.0, 33.6, 73.1, 33.7)


def test_wms_web_mercator_bbox_converts_to_degrees():
    from uavmosaic.tiles import parse_wms_bbox

    # 73.0479 E, 33.6844 N expressed in EPSG:3857 metres
    mx = 73.0479 * 20037508.34 / 180.0
    my = math.log(math.tan((90 + 33.6844) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * 20037508.34 / 180.0
    w, s, e, n = parse_wms_bbox([mx, my, mx + 1000, my + 1000], "EPSG:3857", "1.1.1")
    assert w == pytest.approx(73.0479, abs=1e-4)
    assert s == pytest.approx(33.6844, abs=1e-3)
    assert e > w and n > s
