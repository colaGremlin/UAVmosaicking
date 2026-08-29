"""Sections 5, 6 and the appendices."""

from __future__ import annotations

from make_docx import Doc

CANONICAL = ("python -m uavmosaic.app --extent 7000 --gsd 2.5 \\\n"
             "    --anchor 33.6844,73.0479,540 --plane-z 904 \\\n"
             "    --mjpeg --tiles --fit-view")


# ------------------------------------------------------------------------------------
# 5. Deployment and operation
# ------------------------------------------------------------------------------------
def section5(doc: Doc):
    doc.h1("Deployment and Operation", "5")

    doc.h2("5.1  Quick Start")
    doc.p("Two terminals, in this order, every time.")

    doc.steps([
        "**Install once.** `pip install opencv-python numpy`. FFmpeg is optional and needed "
        "only for the H.264 route in 5.3.",
        "**Terminal 1 - start the ground station.** Change to the project directory and run "
        "the command below. Wait for `running -- ctrl-c to stop`.",
    ])

    doc.code(CANONICAL,
             "**Listing 5.1 - The canonical command.** Every section of this document uses "
             "this exact line. `--extent 7000` is required: at the widest zoom a footprint "
             "near the edge of the 8 km survey box reaches about 2.7 km beyond it, and a "
             "smaller extent clips those frames.")

    doc.steps([
        "**Terminal 2 - start the aircraft.** Either press **Play** in Unity, or run the "
        "simulator: `python tools/sim_sender.py --duration 180 --hz 10 --gsd 2.5`",
        "**Verify.** Open `http://127.0.0.1:8080/stream.mjpg` in a browser. A picture should "
        "build within a few seconds.",
        "**Display.** Configure Mission Planner as in 5.2.",
        "**Stop.** Stop Play in Unity, then Ctrl+C in terminal 1.",
    ])

    doc.callout("stop", "Ground station first, aircraft second", [
        "UDP has no handshake. If Unity transmits before anything is listening, those "
        "datagrams are discarded by the operating system and are gone. The symptom is an "
        "empty canvas with **no error message anywhere** - not in Unity, not in the backend, "
        "not in the system log.",
    ])

    doc.h3("Reading the status line")
    doc.code("ticks=1103 fused=1452 rejected=0 outside_aoi=0 overruns=29 coverage=20.9%\n"
             "  | uav0/EO: 363f 3.3Hz 11.6Mb/s | uav1/EO: 363f 3.3Hz 12.4Mb/s | ...")
    doc.table(
        ["Counter", "Meaning", "Expected"],
        [["`fused`", "Frames composited onto the canvas", "Rising steadily"],
         ["`rejected`", "Frames failing the descent test, incidence gate or extent clamp",
          "0 in level flight; a few during sharp turns"],
         ["`outside_aoi`", "Frames whose footprint falls outside the mapped area",
          "0. If rising, `--extent` is too small or the aircraft left the box"],
         ["`overruns`", "Ticks that exceeded the 100 ms budget",
          "A few is harmless. Hundreds means the machine cannot keep up"],
         ["`coverage`", "Fraction of the area imaged at least once",
          "Climbs throughout; a full survey reaches 60-80 %"]],
        widths=[0.85, 2.5, 2.15], size=8.8)

    doc.h2("5.2  Mission Planner: the Map Layer (recommended)")
    doc.p("This route places the mosaic **on the map screen, beneath the aircraft icons**, "
          "rather than in the small video pane. Mission Planner already loads map imagery "
          "from tile servers; the backend presents itself as one.")

    doc.steps([
        "Start the ground station and the aircraft. Let it run for a minute so there is "
        "coverage to see.",
        "Verify in a browser first: **http://127.0.0.1:8081/**. If that page does not load, "
        "fix it before touching Mission Planner, which reports errors far less clearly.",
        "In Mission Planner click **FLIGHT PLAN** in the top button row. The map source "
        "control exists only on this screen, not on FLIGHT DATA.",
        "At the top-right of the map, open the map-source dropdown - it normally reads "
        "**GoogleSatelliteMap**.",
        "Select **WMS** from the list.",
        "A prompt asks for the server URL. Enter **http://127.0.0.1:8081/wms** and click OK.",
        "Mission Planner requests the capabilities document and then lists the available "
        "layers, numbered. Enter the number for the layer named **mosaic** and click OK.",
        "The map turns grey or blank. This is expected - you have selected a map that covers "
        "only the survey area. **Pan and zoom to the survey area** and the mosaic appears. "
        "The startup log prints the exact corner coordinates.",
    ])

    doc.figure("doc_tiles.png",
               "**Figure 5.1 - The map layer as Mission Planner renders it.** Imaged ground is "
               "opaque; everything else is transparent, so an underlying base map still shows "
               "through where the aircraft have not yet flown.", width=4.4)

    doc.callout("info", "Conformance is tested against Mission Planner's own source", [
        "Five tests in `tests/test_mission_planner_wms.py` replay the exact requests Mission "
        "Planner constructs and assert against each of its five acceptance checks, "
        "transcribed from `GCSViews/FlightPlanner.cs` and `ExtLibs/Maps/WMSProvider.cs`: a "
        "`WMT_MS_Capabilities` root element, exactly one `GetMap` element, a `Format` "
        "containing `image/png`, an `SRS` containing `EPSG:4326`, and at least one "
        "`//Layer/Layer` carrying a `Name`.",
        "This found a real defect. The capabilities document emitted `xlink:href` without "
        "declaring the `xlink` namespace. .NET's `XmlDocument` rejects that as malformed, so "
        "Mission Planner would have displayed a message box and refused the server, while a "
        "test that only checked 'a PNG came back' would have passed. Fixed by declaring "
        "`xmlns:xlink` on the root element.",
    ])

    doc.h2("5.3  Mission Planner: Video Routes")
    doc.p("Two video paths exist in addition to the map layer. Both place the mosaic in the "
          "HUD pane on the FLIGHT DATA screen, behind the instruments.")

    doc.table(
        ["Route", "Backend flag", "Mission Planner configuration", "Notes"],
        [["**MJPEG over HTTP**", "`--mjpeg`",
          "FLIGHT DATA screen. Right-click the artificial horizon, choose **Set MJPEG "
          "Source**, enter `http://127.0.0.1:8080/stream.mjpg`",
          "Native. Nothing to install. The more reliable of the two."],
         ["**H.264 over UDP**", "on by default, `--out-port 5600`",
          "FLIGHT DATA screen. Right-click the HUD, choose the GStreamer or UDP video option, "
          "and point it at UDP port 5600",
          "Requires FFmpeg on the sending side and a working GStreamer install in Mission "
          "Planner. Verify with `ffplay udp://127.0.0.1:5600` first."]],
        widths=[1.05, 1.05, 2.35, 1.55], size=8.6)

    doc.code("ffmpeg -f rawvideo -pix_fmt bgr24 -s 1280x1280 -r 10 -i - -an \\\n"
             "  -c:v libx264 -preset ultrafast -tune zerolatency -b:v 4M -g 5 -bf 0 \\\n"
             "  -pix_fmt yuv420p -f mpegts udp://127.0.0.1:5600?pkt_size=1316",
             "**Listing 5.2 - The encoder command the backend builds.** `-g 5` places a "
             "keyframe every half second, so a late-joining viewer synchronises quickly. "
             "`-bf 0` removes B-frames, which would add reordering latency. Switch to "
             "`--container rtp` for Mission Planner's GStreamer path.")

    doc.callout("warn", "Recommendation", [
        "Use the **map layer** (5.2) as the primary display. It is georeferenced, it sits "
        "under the aircraft icons where the operator is already looking, it pans and zooms "
        "with the map, and it needs no codec.",
        "Use **MJPEG** as a secondary view or for a quick check in a browser.",
        "Use **H.264** only when the mosaic must travel to another machine over a "
        "bandwidth-constrained link, which is the one case where the codec earns its "
        "complexity.",
    ])

    doc.figure("doc_hud.png",
               "**Figure 5.2 - Overlay levels.** Left: `--hud minimal`, the default operator "
               "view - aircraft markers, north arrow, scale bar. Right: `--hud full`, adding "
               "camera angles, plane-cascade tier, per-aircraft coverage share and tick "
               "timings. Full is a diagnostic view, not an operator view.")

    doc.h2("5.4  Target Coordinate Extraction")
    doc.p("A target may be designated either on the canvas or in an aircraft's source frame. "
          "Both resolve to latitude and longitude; the two paths differ in how the ray is "
          "constructed.")

    doc.h3("From a canvas pixel")
    doc.formula([
        "E  =  E_min + x_px · GSD",
        "N  =  N_max - y_px · GSD          // inverse of Formula 2.6, exact",
        "",
        "then  (E, N, z_plane)  →  ECEF  →  WGS-84 latitude, longitude, altitude",
    ], "**Formula 5.1** - `TargetResolver.from_canvas_px`. The canvas-to-ENU mapping is a "
       "fixed affine, so this inversion is exact and costs no search.")

    doc.h3("From a source-frame pixel")
    doc.p("This is the higher-accuracy path, because it uses that frame's own pose and plane "
          "rather than the composited result:")

    doc.formula([
        "d_c  =  K⁻¹ [u, v, 1]ᵀ",
        "d_w  =  R_E←C · d_c",
        "λ    =  (z_plane - C.z) / d_w.z",
        "G    =  C + λ · d_w         →  ECEF  →  WGS-84",
    ], "**Formula 5.2** - `TargetResolver.from_source_px`. A single ray, back-projected and "
       "intersected with that frame's plane. No interpolation.")

    doc.callout("info", "Why not interpolate between the four corners", [
        "The Argus reference implementation reads a target out by bilinear interpolation "
        "across the four projected footprint corners. For a perspective quadrilateral that is "
        "wrong: perspective is not affine, so the interpolant deviates from the true "
        "projection everywhere except at the corners themselves, with the error largest at "
        "the centre - exactly where an operator is most likely to click.",
        "`targets.bilinear_error_metres` quantifies the difference for any given footprint, "
        "and the exact inverse homography is used instead.",
    ])

    doc.h3("The geodetic anchor")
    doc.p("`--anchor lat,lon,alt` fixes the ENU origin to a point on the Earth. Without it "
          "the mosaic is metrically correct but has no absolute position, the map layer "
          "cannot be served, and target read-out returns ENU only.")
    doc.p("The conversion uses exact WGS-84 ECEF mathematics implemented in `coords.py`, "
          "with no `pyproj` dependency. Every target record carries `uav_id`, `sensor`, "
          "`t_capture`, canvas pixel, ENU, latitude and longitude, the plane-cascade tier "
          "used, and the incidence angle - so accuracy is auditable per fix rather than "
          "assumed.")

    doc.table(
        ["Quality label", "Condition", "Interpretation"],
        [["`good`", "Tier 1 plane, incidence below 30 degrees",
          "Laser-measured ground elevation, near-nadir geometry"],
         ["`fair`", "Tier 1 or 2, incidence 30 to 50 degrees",
          "Usable; parallax error grows with the tangent of the angle"],
         ["`poor`", "Tier 3 plane, or incidence above 50 degrees",
          "Ground elevation is a configured default, or the geometry is shallow. Treat as "
          "indicative"]],
        widths=[0.85, 1.85, 2.8], size=8.8)

    doc.h2("5.5  Radio Link Deployment")
    doc.p("The ground station runs **on the ground**, on the same machine as Mission "
          "Planner. It must: the mosaic is built from all four aircraft, so all four streams "
          "have to arrive in one place. There is therefore exactly one constrained link, "
          "aircraft to ground. Everything downstream is inside one computer.")

    doc.table(
        ["Change", "Where", "Value"],
        [["Accept traffic from the radio interface", "backend", "`--bind 0.0.0.0`"],
         ["Tolerate slower arrival", "backend", "`--max-age 1.5`"],
         ["Ground-station address", "`UavStreamer.Host`", "the radio-side IP"],
         ["Transmission rate", "`UavStreamer.Target Hz`", "3"],
         ["Picture size", "`Capture Width / Height`", "960 x 540"],
         ["Compression", "`Jpeg Quality`", "60"]],
        widths=[2.0, 1.5, 2.0], size=8.8)

    doc.h3("Bandwidth, measured on this terrain, all four aircraft")
    doc.table(
        ["Picture", "Quality", "Rate", "Per frame", "Four aircraft", "Suitable for"],
        [["1280 x 720", "65", "10 Hz", "124 KB", "40.5 Mbit/s", "Cable or loopback only"],
         ["1280 x 720", "55", "3 Hz", "106 KB", "10.4 Mbit/s", "A strong link, short range"],
         ["**960 x 540**", "**60**", "**3 Hz**", "**71 KB**", "**6.9 Mbit/s**",
          "**Recommended default**"],
         ["640 x 360", "55", "3 Hz", "37 KB", "3.6 Mbit/s", "Weak or long-range link"]],
        widths=[0.95, 0.6, 0.55, 0.75, 0.95, 1.7], size=8.5,
        align=[None, "r", "r", "r", "r", None])

    doc.p("A typical UAV datalink carries 2 to 10 Mbit/s for the whole fleet, so the middle "
          "two rows are the realistic options. Reduce frame rate before reducing picture "
          "size: a sharp frame arriving less often builds a better mosaic than a soft one "
          "arriving constantly.")

    doc.h3("Packet loss is the harder constraint")
    doc.p("A frame is discarded unless **every** one of its fragments arrives, because a "
          "truncated JPEG is not a partial image. Survival therefore falls as "
          "(1 - loss) raised to the fragment count. Measured against the reassembler "
          "directly, with sockets and timing removed:")

    doc.table(
        ["Picture", "Fragments", "1 %", "2 %", "3 %", "5 %", "10 %"],
        [["1280 x 720", "25", "77 %", "59 %", "44 %", "27 %", "7 %"],
         ["960 x 540", "15", "87 %", "72 %", "62 %", "48 %", "22 %"],
         ["640 x 360", "8", "92 %", "84 %", "78 %", "67 %", "44 %"]],
        widths=[1.1, 0.85, 0.65, 0.65, 0.65, 0.65, 0.65], size=8.8,
        align=[None, "r", "r", "r", "r", "r", "r"])

    doc.p("At 3 % loss - an ordinary radio, not a poor one - a 1280 x 720 stream loses more "
          "than half its frames, while 640 x 360 keeps three quarters. This is a stronger "
          "argument for a smaller picture than bandwidth is.")

    doc.callout("info", "Frame loss degrades a mosaic far less than it degrades video", [
        "The canvas accumulates. A lost frame is not a gap in a sequence; it is a frame that "
        "does not contribute, and the ground it would have covered is covered by the next one "
        "a third of a second later.",
        "At 3 Hz with half the frames lost, each aircraft still delivers about 1.5 usable "
        "frames per second. At 60 m/s that is 40 m of travel between frames against a "
        "footprint over 1600 m wide - overlap remains enormous. The mosaic fills in more "
        "slowly and nothing else changes: no tearing, no misregistration, no corruption.",
    ])

    doc.code("python tools/sim_sender.py --duration 180 --hz 3 --gsd 2.5 "
             "--quality 60 --loss 0.03",
             "**Listing 5.3** - Rehearse a lossy link before flying. `--loss` discards that "
             "fraction of datagrams at the sender, exercising reassembly paths that loopback "
             "never reaches.")
    doc.pagebreak()


# ------------------------------------------------------------------------------------
# 6. Limits and troubleshooting
# ------------------------------------------------------------------------------------
def section6(doc: Doc):
    doc.h1("System Limits and Troubleshooting", "6")

    doc.h2("6.1  Flat-Plane Parallax")
    doc.p("The dominant error source, and the one that cannot be removed within this "
          "architecture. The projection assumes the ground is a plane. Any object or terrain "
          "standing off that plane is drawn where its ray meets the plane, not where it "
          "actually is.")

    doc.formula([
        "displacement  ≈  Δh · tan θ",
        "",
        "//  Δh  height of the point above the assumed plane, metres",
        "//  θ   incidence angle of the ray, degrees off nadir",
    ], "**Formula 6.1** - Parallax displacement. It is independent of altitude and of focal "
       "length; only height above the plane and viewing angle matter.")

    doc.figure("fig_parallax.png",
               "**Figure 6.1 - Why a building leans.** The ray from the roof is drawn where it "
               "crosses the assumed plane. The roof is therefore displaced outward from nadir "
               "by approximately h·tanθ.", width=6.0)

    doc.table(
        ["Height above plane", "0 deg", "10 deg", "20 deg", "30 deg", "45 deg", "65 deg"],
        [["10 m", "0.0 m", "1.8 m", "3.6 m", "5.8 m", "10.0 m", "21.4 m"],
         ["50 m", "0.0 m", "8.8 m", "18.2 m", "28.9 m", "50.0 m", "107.2 m"],
         ["200 m", "0.0 m", "35.3 m", "72.8 m", "115.5 m", "200.0 m", "428.9 m"],
         ["500 m", "0.0 m", "88.2 m", "182.0 m", "288.7 m", "500.0 m", "1072.3 m"]],
        widths=[1.25, 0.68, 0.72, 0.72, 0.72, 0.72, 0.78], size=8.6,
        align=[None, "r", "r", "r", "r", "r", "r"])

    doc.callout("stop", "What this means on the delivered terrain", [
        "Ground under the survey box spans 283 m to 1766 m, a relief of **1483 m**, against a "
        "flying height of about 3300 m. The relief-to-height ratio is **0.45**, which is high "
        "- flat-plane projection is normally applied where that ratio is a few per cent.",
        "The laser cascade limits the damage: each frame gets the elevation of the ground its "
        "boresight actually strikes, so Δh is *local* relief within one footprint rather than "
        "the full 1483 m. But two aircraft viewing the same ridge from different angles still "
        "place it in slightly different positions, and where their frames meet, a linear "
        "feature such as a river shows a visible kink.",
        "**Seam feathering cannot fix this.** Feathering removes the brightness step; the "
        "geometric offset is already baked into each frame at projection time. The only "
        "complete fix is to project onto a digital surface model instead of a plane, which "
        "costs roughly 25x more per frame and requires elevation data.",
    ])

    doc.figure("seam_compare.png",
               "**Figure 6.2 - What feathering does and does not do.** Top: hard weight "
               "comparison, `--feather 0`. Bottom: the default `--feather 0.3`. A deliberately "
               "harsh test - two frames of the same grid at different exposure. The brightness "
               "step falls from 108.5 to 53.6 levels of 255. Geometric alignment is identical "
               "in both; feathering is a photometric fix only.", width=3.6)

    doc.h3("Mitigations already in place")
    doc.table(
        ["Measure", "Effect"],
        [["65 degree incidence gate",
          "Truncates the tail of the tan curve. Worst measured corner on the delivered "
          "configuration was 48.5 degrees, so nothing is rejected in normal flight"],
         ["Incidence weighting, cos²θ",
          "Where two aircraft overlap, the more nearly vertical view wins automatically"],
         ["Per-frame laser plane",
          "Δh is local relief within one footprint, not relief across the whole area"],
         ["Radial weighting, (1-r/r_max)²",
          "Frame edges, where projection error and lens distortion are both worst, lose to "
          "frame centres in the overlap"],
         ["Ground-sample weighting",
          "The frame that resolves more detail wins, so a zoomed-in near-nadir view "
          "outranks a distant oblique one"]],
        widths=[1.6, 4.4], size=8.8)

    doc.h2("6.2  Other Stated Limits")
    doc.bullets([
        "**Accuracy is bounded by telemetry, not by this software.** In Unity the pose is "
        "ground truth, so Correia's two-decimal result is the realistic ceiling. On real "
        "hardware expect the 1-3 m class: Map2DFusion reports 3.07 and 5.71 m, GROMS reports "
        "1.11 and 0.64 m mean absolute error.",
        "**No loop closure, no bundle adjustment, no drift correction** - by design. There is "
        "nothing to correct, because every frame is independently anchored to the world "
        "rather than to its neighbours.",
        "**Hard exposure seams** where aircraft with differing gain meet. Unity's identical "
        "virtual cameras largely avoid this; feathering handles the residual. Multi-band "
        "blending is designed in as a quality toggle if real cameras make it necessary.",
        "**Fixed area of interest.** A frame whose footprint falls outside the declared box "
        "is counted and discarded. This buys constant memory and an allocation-free loop, at "
        "the cost of robustness if an aircraft strays. Appropriate because a survey has a "
        "defined box.",
        "**One planar ground per aircraft per frame** - not a single global plane. Each "
        "aircraft solves its own from its own laser return.",
    ])

    doc.h2("6.3  Coordinate Frame Validation Checklist")
    doc.p("Run this before trusting any mosaic from a new scene. A frame error produces a "
          "mosaic that looks plausible but is mirrored, rotated or transposed.")

    doc.table(
        ["#", "Check", "Expected", "If it fails"],
        [["1", "`python -m pytest tests/test_coords.py -q`",
          "All pass, including the golden nadir case",
          "Stop. Nothing downstream is trustworthy"],
         ["2", "Determinant of R_E←C for random quaternions", "+1 to within 1e-12",
          "S or F has been modified; both must have determinant -1"],
         ["3", "Orthonormality: R'R = I", "Identity to within 1e-12",
          "The quaternion is not normalised at the sender"],
         ["4", "North arrow in the overlay points to the top of the canvas",
          "Always", "The N flip in Formula 2.6 is inverted"],
         ["5", "Aircraft moving east moves right on the canvas",
          "Always", "The S matrix rows are transposed"],
         ["6", "`rejected` stays near zero in level flight",
          "0 with an 8 degree gimbal wander",
          "The mapping camera is not pointing down. Run **Check Scene**"],
         ["7", "A known landmark reads out at its true latitude and longitude",
          "Within a few metres in simulation",
          "The `--anchor` value is wrong, or the Unity origin is not at the anchor"]],
        widths=[0.25, 1.95, 1.6, 2.2], size=8.4)

    doc.callout("warn", "If your Unity scene has a rotated world", [
        "This build assumes Unity +Z is North and +X is East. A scene rotated by yaw angle "
        "psi about the vertical needs one extra rotation folded into S. It belongs in "
        "`coords.py` beside the existing matrices, **not** in the Unity sender - the sender "
        "transmits raw values by design, and splitting the frame conversion across two "
        "languages is how it becomes unfindable.",
    ])

    doc.h2("6.4  Failure Modes")
    doc.table(
        ["Symptom", "Cause", "Remedy"],
        [["`'python' is not recognized`", "Python not on PATH",
          "Reinstall Python with **Add python.exe to PATH** ticked"],
         ["A log line pasted into the shell errors",
          "The startup banner prints addresses, not commands",
          "Addresses go in a browser or into Mission Planner when prompted"],
         ["Canvas stays black, no per-aircraft counters",
          "Nothing is arriving",
          "Confirm the ground station started **before** Unity. Check `Host` and port in the "
          "streamer"],
         ["Canvas black, counters present, `outside_aoi` rising",
          "Footprints fall outside the mapped area",
          "`--extent 7000`, and `Survey Half Extent` 4000 in Unity"],
         ["`rejected` climbing steadily",
          "Cameras beyond the 65 degree gate",
          "The mapping camera is not nadir. Run **Tools > Mosaicking > Check Scene**"],
         ["Frames arrive but coverage is patchy",
          "Fragments lost, frames never completing",
          "Reduce capture size, which reduces fragments per frame. See 5.5"],
         ["Hundreds of `overruns`", "Machine cannot sustain the tick",
          "Raise `--gsd` to 5, or lower `--hz` to 5. GSD is by far the larger lever"],
         ["`Address already in use`", "A previous run did not release the port",
          "Close the old process, or change `--tiles-port` and `--mjpeg-port`"],
         ["Mission Planner refuses the WMS server",
          "Wrong URL, or the backend is not running",
          "The address must end in `/wms`. Verify port 8081 in a browser first"],
         ["Mission Planner accepts it, map stays empty",
          "Viewport is elsewhere in the world",
          "Pan to the coordinates printed at startup. This is the most common cause"],
         ["FFmpeg errors on startup", "FFmpeg missing or misconfigured",
          "Harmless for the map and MJPEG routes. Add `--no-encoder` to silence it"],
         ["Unity Console shows red errors", "`MosaickingSetup.cs` in the wrong folder",
          "It must sit directly inside a folder named `Editor`"],
         ["Mosaic freezes under one aircraft while it still transmits",
          "The same-owner clause has been removed or broken",
          "See 2.5. Verify `tests/test_canvas.py` passes"]],
        widths=[1.75, 1.75, 2.5], size=8.3)

    doc.callout("info", "First diagnostic step, always", [
        "`python -m pytest tests/ -q`. If all 168 pass, the mathematics, the protocol and the "
        "compositing rules are intact, and the fault is in wiring: a port, a folder, a "
        "startup order, a firewall. If any fail, start with the failing test rather than with "
        "the symptom.",
    ])
    doc.pagebreak()


# ------------------------------------------------------------------------------------
# Appendices
# ------------------------------------------------------------------------------------
def appendices(doc: Doc):
    doc.h1("Source Material", "A")
    doc.p("Every source below was read in full, including its mathematics. Where a formula, "
          "sign or convention was ambiguous in the text, it was verified against that "
          "project's own source code; the derivations in Section 2 follow the code, not the "
          "prose.")

    doc.h2("A.1  Papers")
    doc.table(
        ["Source", "Contribution to this design", "Used / Rejected"],
        [["**Correia et al.**, *Sensors* 2022, 22, 604\n"
          "`Research Papers/Correia et al.pdf`",
          "The mathematical spine. Intrinsics under variable zoom (Eq. 32-33), the full "
          "transformation chain (Eq. 51-54), the z_C solution (Eq. 60-61), field of view "
          "(Eq. 66). Validated in Unreal Engine 4 with AirSim against a virtual UTM origin, "
          "recovering a target to two decimal places - the same class of experiment as this "
          "Unity setup.",
          "**Used.** Formulas 2.4 and 2.5"],
         ["**Hinzmann et al.**, *aerial_mapper* (ETH ASL)\n"
          "`Research Papers/hinzmann2017mapping.pdf`",
          "The corner-projection to `getPerspectiveTransform` to `warpPerspective` pattern, "
          "and the best-viewing-angle rule that became the incidence weight. Timing anchor: "
          "17.4 ms per image at 752x480 in C++.",
          "**Used.** Section 1.3, 2.5"],
         ["**Bu et al.**, *Map2DFusion*, IROS 2016\n"
          "`Research Papers/Map2DFusion_...pdf`",
          "The plane-local frame concept, the incidence gate (axis.dot(downLook) < 0.4, about "
          "66 degrees), the radial weight map floored at 1e-5, explicit metres-to-pixels "
          "scaling, and per-pixel max-weight selection rather than weighted averaging.",
          "**Used.** Section 2.5"],
         ["**Yao et al.**, *GROMS*\n`Research Papers/GROMS.pdf`",
          "The hybrid block/pixel orthorectification argument that formally justifies one "
          "homography per frame on a plane. Runtime budget anchor (Table 6). Its critique of "
          "flat-plane methods is the honest caveat in Section 6.1.",
          "**Partially used.** Its map-prior and SLAM pose path violates the no-prior-map "
          "constraint and is rejected"],
         ["**Ruiz et al.**, *MGRAPH*, RA-L 2018\n`Research Papers/MGRAPH.pdf`",
          "The deliberate counter-example: ORB, brute-force Hamming, RANSAC, graph fusion. "
          "Its Eq. 3 zero-overlap distance is retained as a contention predictor. Its "
          "GCS-overlay split - send transform plus image, composite at the ground station - "
          "matches this architecture.",
          "**Rejected for registration.** Section 1.2"],
         ["**SkyPin**, *Drones* 2026, 10, 500\n`Research Papers/drones-10-00500.pdf`",
          "A third independent statement of the ray/plane formula (Eq. 2-4). Benchmarks "
          "matchers at 100-200 m altitude and -30 to -60 degree pitch, reporting that even "
          "the best struggle at shallow angles - direct evidence for the incidence gate.",
          "**Partially used.** Its map-prior and feature-matching pipeline is excluded"]],
        widths=[1.35, 3.05, 1.6], size=8.2)

    doc.h2("A.2  Reference Implementations")
    doc.table(
        ["Repository", "Taken", "Not taken"],
        [["**Map2DFusion** (Bu et al.)\n`github repo 1 - Map2d fusion/`",
          "`MultiBandMap2DCPU.cpp:311-558` - weight map, max-weight selection, incidence "
          "gate. `Map2D.h:54-61`",
          "Monocular SLAM, RANSAC plane fitting (the laser replaces it), the growing tiled "
          "canvas, OpenGL, Qt, Svar"],
         ["**aerial_mapper** (Hinzmann et al.)\n`github repo 2 - Aerial Mapper/`",
          "`ortho-forward-homography.cc:74-132` - the corner-to-ground-to-warp pattern",
          "ROS, aslam, grid_map, GDAL, the digital surface model, the per-cell loop. **Three "
          "defects were identified and deliberately not reproduced**: `batch()` uses the "
          "mosaic width for both x and y offsets (line 154-158); "
          "`layer_num_observations(x,y) += layer_num_observations(x,y)` never increments off "
          "zero (line 93); and the forward path has no metres-to-pixels scale at all, "
          "implicitly fixing ground sample distance at 1 m/px"],
         ["**Argus**\n`github repo 3 - Argus/`",
          "`advanced_mapping.py:132-401` - the cleanest Python reference in the set. "
          "Region-of-interest local warping, the 20x footprint sanity clamp, the lower-half "
          "retry, alpha as a validity mask",
          "FastAPI, SQLAlchemy, COLMAP, OpenDroneMap, multiprocessing pools, Voronoi seams. "
          "`spatial.py:18-45` reads targets out by bilinear interpolation across four "
          "corners, which is wrong for a perspective quadrilateral - the exact inverse "
          "homography is used instead (5.4)"]],
        widths=[1.3, 2.3, 2.4], size=8.2)

    doc.callout("warn", "One published result could not be reproduced", [
        "Correia et al. Table 1 gives a worked example. This implementation reproduces the "
        "easting to 7 micrometres and the image v coordinate to 0.000 px, but the northing "
        "differs by 2.89 m. Matching their published northing exactly requires a rotation "
        "matrix with determinant -1, which is not a rotation.",
        "The most likely explanation is the paper's own documented pitfall: when gimbal "
        "angles are world-referenced, the airframe attitude must be set to identity or the "
        "rotation is counted twice. The quaternion-based design used here makes that "
        "particular error impossible. The discrepancy is recorded rather than hidden, and the "
        "build was not gated on it; the synthetic ground-truth test in Appendix C is the "
        "stronger check.",
    ])

    doc.pagebreak()
    doc.h1("Command-Line Reference", "B")

    for title, rows in (
        ("Area and geometry", [
            ["`--extent`", "7000", "Half-width of the mapped area in metres. 7000 gives a "
                                   "14 km square, large enough that a wide-zoom footprint "
                                   "near the survey edge is not clipped"],
            ["`--gsd`", "2.5", "Canvas resolution, metres per pixel. Halving it quadruples "
                               "memory. The largest single performance lever"],
            ["`--anchor`", "required", "`lat,lon[,alt]` of the area centre. Required for the "
                                       "map layer and for target read-out"],
            ["`--plane-z`", "0.0", "Ground elevation used only when both laser tiers fail"],
            ["`--max-incidence`", "65.0", "Reject a frame whose worst corner exceeds this "
                                          "angle off nadir"],
        ]),
        ("Output", [
            ["`--tiles`", "off", "Serve the mosaic as a map layer, XYZ tiles plus WMS. "
                                 "Requires `--anchor`"],
            ["`--mjpeg`", "off", "Serve MJPEG over HTTP. Mission Planner reads this natively"],
            ["`--fit-view`", "off", "Zoom the video output to the imaged region. Does not "
                                    "affect the map layer, which stays georeferenced"],
            ["`--hud`", "minimal", "`off`, `minimal` (operator), or `full` (engineering)"],
            ["`--feather`", "0.3", "Cross-fade width between aircraft. 0 gives a hard seam"],
            ["`--save`", "none", "Write the final mosaic to this PNG on exit"],
            ["`--no-encoder`", "off", "Disable the H.264 output entirely"],
            ["`--container`", "mpegts", "`mpegts` for ffplay and VLC, `rtp` for Mission "
                                        "Planner's GStreamer path"],
            ["`--out-width`, `--out-height`", "auto", "Output size. Defaults to matching the "
                                                      "area aspect so there are no black bars"],
        ]),
        ("Network", [
            ["`--bind`", "127.0.0.1", "Interface to listen on. `0.0.0.0` for a radio link"],
            ["`--max-age`", "0.5", "Ignore frames older than this. Raise to 1.5 at 3 Hz"],
            ["`--tiles-port`", "8081", "Map layer port"],
            ["`--mjpeg-port`", "8080", "MJPEG port"],
            ["`--out-host`, `--out-port`", "127.0.0.1:5600", "H.264 destination"],
            ["`--bitrate`", "4M", "H.264 target bitrate"],
            ["`--hz`", "10", "Fusion loop rate. Independent of the aircraft transmit rate"],
        ]),
        ("Diagnostics", [
            ["`--stats`", "off", "Print the per-stage latency table on exit"],
            ["`--duration`", "unlimited", "Stop after this many seconds"],
            ["`--ir`", "off", "Also composite the infrared streams"],
            ["`-v`", "off", "Verbose logging"],
        ]),
    ):
        doc.h2(title)
        doc.table(["Flag", "Default", "Effect"], rows,
                  widths=[1.25, 0.75, 4.0], size=8.5)

    doc.pagebreak()
    doc.h1("Verification Inventory", "C")
    doc.p("168 automated tests. The list below states what each file proves, not how many "
          "assertions it contains.")

    doc.table(
        ["Test file", "Proves"],
        [["`test_coords.py`",
          "The golden nadir north-up case hand-derived in 2.2, plus 45 degree pitch and roll "
          "cases. Orthonormality and determinant +1 for randomised quaternions. **Nothing "
          "else is trustworthy if this fails**"],
         ["`test_georef.py`",
          "Ray/plane intersection against Correia's published worked example - an "
          "independent, peer-reviewed ground truth rather than a self-consistency check. The "
          "3-tier cascade, the descent test, the incidence gate, the extent clamp and the "
          "lower-half retry"],
         ["`test_protocol.py`",
          "Round trip, fragment reordering, fragment loss, duplicate fragments, oversized "
          "frames, and a bounded-memory assertion under sustained loss"],
         ["`test_canvas.py`",
          "The same-owner overwrite rule, cross-owner max-weight arbitration, area clipping, "
          "region-of-interest translation, and the feathering guards from Listing 2.2"],
         ["`test_targets.py`",
          "Exact inverse-homography read-out, and the measured error of the bilinear "
          "approximation it replaces"],
         ["`test_unity_wire.py`",
          "Byte-level agreement between the C# sender's layout and the Python parser, so a "
          "field added on one side cannot silently misalign the other"],
         ["`test_end_to_end.py`",
          "Full pipeline against a synthetic ground-truth world: NCC floor 0.99, correlation "
          "peak at zero offset. This is the test that caught the feathering regression "
          "described in 2.5"],
         ["`test_tiles.py`",
          "Web Mercator tile boundaries, WMS bounding-box parsing including the 1.3.0 "
          "latitude-longitude axis-order reversal, and transparent responses outside coverage"],
         ["`test_mission_planner_wms.py`",
          "Mission Planner's own five acceptance checks, transcribed from its source. Found "
          "the namespace defect described in 5.2"]],
        widths=[1.35, 4.65], size=8.5)

    doc.code("cd C:\\Users\\Workstation\\Fatima\\UAVmosaicking\n"
             "python -m pytest tests/ -q",
             "**Listing C.1** - Expect `168 passed` in about 15 seconds.")

    doc.h2("C.1  Measured Performance")
    doc.table(
        ["Stage", "Median", "Note"],
        [["Full tick", "30.3 ms", "Against a 100 ms budget at 10 Hz, 0 overruns"],
         ["JPEG decode x 4", "2-4 ms", "Parallel, GIL released"],
         ["Projection and homography x 4", "< 0.5 ms", "Four corners each, pure arithmetic"],
         ["Weight maps x 4", "~2 ms", "Radial cached, incidence at 1/16 scale then resized"],
         ["`warpPerspective` x 4", "12-30 ms", "Parallel, dominant cost"],
         ["Composite 4 regions", "2-5 ms", "`cv2.copyTo`, 0.28 ms versus 22.88 ms for "
                                           "`np.copyto(where=...)`"],
         ["Overlay and encoder copy", "3-5 ms", "Encoding itself is out of process"]],
        widths=[1.9, 0.85, 3.25], size=8.6)

    doc.spacer(14)
    doc.p("*End of document.*", size=9)
