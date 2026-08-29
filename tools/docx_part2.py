"""Sections 3 and 4 of the technical documentation."""

from __future__ import annotations

from make_docx import Doc


# ------------------------------------------------------------------------------------
# 3. Network architecture and wire protocol
# ------------------------------------------------------------------------------------
def section3(doc: Doc):
    doc.h1("Network Architecture and Wire Protocol", "3")

    doc.h2("3.1  Dataflow Topology")
    doc.figure("fig_flow.png",
               "**Figure 3.1 - End-to-end dataflow.** Four aircraft, four dedicated receiver "
               "threads, one fusion loop, three output paths. Everything to the right of the "
               "receivers runs inside a single Python process.")

    doc.table(
        ["Stage", "Concurrency", "Blocks on", "Never does"],
        [["`UdpReceiver` x 4", "one daemon thread each", "`recvfrom`, 0.5 s timeout",
          "touch the canvas"],
         ["`LatestSlot` x 4", "lock-guarded, 1 deep", "nothing measurable",
          "grow, queue or allocate"],
         ["`FusionEngine`", "one daemon thread", "a fixed-rate 10 Hz tick", "socket I/O"],
         ["warp pool", "4 worker threads", "`ThreadPoolExecutor.map`", "mutate shared state"],
         ["`FfmpegSink`", "one daemon writer", "ffmpeg stdin", "block fusion; it drops instead"],
         ["`TileServer`, `MjpegSink`", "HTTP server threads", "client sockets",
          "hold the canvas lock across a network write"]],
        widths=[1.15, 1.25, 1.35, 1.65], size=8.8)

    doc.callout("info", "Threads, not processes", [
        "Every expensive call in the loop - `cv2.imdecode`, `cv2.warpPerspective`, "
        "`cv2.resize`, NumPy element-wise operations - **releases the GIL**, so these threads "
        "achieve genuine parallelism.",
        "Processes would be strictly worse: the canvas is 375 MB across its four buffers "
        "(94 MB colour, 125 MB weight, 31 MB owner, 125 MB timestamp at 5600 x 5600), and "
        "pickling any part of that across a pipe every tick would cost more than the work "
        "being distributed.",
    ])

    doc.h3("The mailbox is one deep, and that is deliberate")
    doc.p("`LatestSlot` holds exactly one frame per stream. A new arrival overwrites "
          "whatever was there and increments a superseded counter. It is **not** a queue.")
    doc.bullets([
        "A queue under sustained overload grows without bound, and every frame drawn from it "
        "is older than the last. Latency debt accumulates until the display is minutes behind "
        "the aircraft while appearing to work normally.",
        "A one-deep mailbox discards the *older* frame. Latency stays bounded at one tick "
        "regardless of load, and the superseded counter makes the overload visible as a "
        "number rather than as a slowly worsening feeling.",
    ])

    doc.h2("3.2  Wire Protocol")
    doc.p("MJPEG carried in UDP datagrams. Little-endian throughout. Each datagram is a "
          "36-byte header, a 64-byte telemetry block, then a slice of the JPEG.")

    doc.figure("fig_packet.png",
               "**Figure 3.2 - Datagram layout.** The telemetry block repeats in every "
               "fragment, not only the first.", width=6.2)

    doc.h3("Header, 36 bytes")
    doc.table(
        ["Off", "Size", "Type", "Field", "Meaning"],
        [["0", "4", "u32", "`magic`", "0x31564155, reads as 'UAV1'. Rejects foreign traffic on the port."],
         ["4", "1", "u8", "`version`", "Protocol version, currently 1"],
         ["5", "1", "u8", "`uav_id`", "0 to 3"],
         ["6", "1", "u8", "`sensor_id`", "0 = EO, 1 = IR"],
         ["7", "1", "u8", "`flags`", "bit0 LRF valid, bit1 AGL valid, bit2 telemetry present"],
         ["8", "4", "u32", "`frame_id`", "Monotonic per (uav, sensor). Detects loss and reordering."],
         ["12", "8", "u64", "`t_capture_us`", "Unity capture time in microseconds. The sync key."],
         ["20", "2", "u16", "`frag_index`", "0-based index of this fragment"],
         ["22", "2", "u16", "`frag_count`", "Total fragments in this frame"],
         ["24", "2", "u16", "`frag_len`", "JPEG bytes carried in **this** datagram"],
         ["26", "4", "u32", "`total_len`", "JPEG bytes in the complete frame"],
         ["30", "2", "u16", "`telem_len`", "Always 64"],
         ["32", "4", "u32", "`frag_offset`", "Byte offset of this slice within the JPEG"]],
        widths=[0.32, 0.32, 0.38, 0.95, 3.53], size=8.4, align=["r", "r", "m", None, None])

    doc.callout("warn", "frag_offset exists because of a real defect", [
        "The first implementation omitted it and had the receiver reconstruct each slice's "
        "position by multiplying `frag_index` by an assumed chunk size. That works only while "
        "every fragment is exactly the same length, which stops being true the moment the MTU "
        "changes or a final short fragment appears. The offset is now transmitted explicitly, "
        "so reassembly never infers geometry it was not told.",
    ])

    doc.h3("Telemetry, 64 bytes, present in every fragment")
    doc.table(
        ["Off", "Size", "Type", "Field", "Meaning"],
        [["0", "12", "f32 x3", "`pos_unity`", "Camera position, **raw Unity world coordinates**, metres"],
         ["12", "16", "f32 x4", "`quat_world_cam`", "Camera orientation x, y, z, w, **raw Unity**"],
         ["28", "4", "u16 x2", "`img_w`, `img_h`", "Image dimensions in pixels"],
         ["32", "16", "f32 x4", "`fx`, `fy`, `cx`, `cy`", "Intrinsics in pixels at the current zoom"],
         ["48", "16", "f32 x4", "`lrf_slant_m`, `agl_m`, `hfov_deg`, `zoom`",
          "Laser slant range along the boresight, height above ground, horizontal field of "
          "view for cross-check, and zoom factor"]],
        widths=[0.32, 0.36, 0.5, 1.35, 2.97], size=8.4, align=["r", "r", "m", None, None])

    doc.callout("info", "Why telemetry repeats in every fragment", [
        "It costs about 64 B x 25 fragments = 1.6 KB per frame, roughly 5 % overhead, and buys "
        "the property that matters: **pose and pixels are atomic**. Lose fragment 0 on a real "
        "radio link and the remaining fragments still carry the full pose.",
        "Nothing in the system can ever pair frame N's pose with frame N+1's image, because "
        "they never travelled separately. The alternative - a telemetry stream alongside a "
        "video stream, correlated by timestamp - is the standard design and the standard "
        "source of subtle registration error.",
    ])

    doc.h3("Fragmentation and loss policy")
    doc.table(
        ["Parameter", "Value", "Reason"],
        [["MTU", "1400 B", "Loopback tolerates ~64 KB, but a protocol only exercised on "
                           "loopback hides fragmentation bugs that appear on the first real link"],
         ["Payload per fragment", "1300 B", "1400 minus 36 header minus 64 telemetry"],
         ["Frame deadline", "150 ms", "A frame missing any fragment is dropped whole at this age"],
         ["In-flight frames", "2 per stream", "Bounds memory, so loss cannot grow the heap"],
         ["Partial JPEG", "never decoded", "A truncated JPEG is not a partial image, it is noise"],
         ["EO ports", "5001-5004", "One socket per stream gives per-aircraft OS buffering"],
         ["IR ports", "5011-5014", "Wired, dormant until `--ir`"]],
        widths=[1.15, 0.85, 3.4], size=8.8)

    doc.p("Typical load: 1280 x 720 at quality 65 gives about 31 KB per frame, so 25 "
          "fragments. Four aircraft at 10 Hz is roughly 1000 datagrams per second. Section "
          "5.5 covers what happens when those datagrams start disappearing.")

    doc.h2("3.3  Why Background Daemons Prevent Buffer Overflow")
    doc.p("A UDP socket's receive buffer is fixed. If the application does not call "
          "`recvfrom` fast enough, the kernel discards arriving datagrams silently - no "
          "error, no exception, just missing fragments and frames that never complete.")
    doc.p("The failure is easy to create: do the georeferencing work on the same thread that "
          "reads the socket. Warping takes 12-30 ms; during that window nothing is draining "
          "the buffer, and at 1000 datagrams per second the buffer fills.")

    doc.bullets([
        "**Separation.** Receiver threads do nothing but `recvfrom`, reassemble, decode, and "
        "publish to a mailbox. They never touch the canvas and never block on the fusion loop.",
        "**Enlarged buffers.** Each socket requests an 8 MB receive buffer, roughly 6000 "
        "datagrams of headroom against a scheduling hiccup.",
        "**Bounded work per datagram.** Reassembly is a dictionary insert and a memory copy. "
        "JPEG decode releases the GIL, so other receivers keep draining during it.",
        "**Daemon threads.** All are `daemon=True`, so Ctrl+C terminates the process without "
        "waiting on a blocking `recvfrom`.",
        "**Failures degrade to counters.** A malformed packet, an undecodable JPEG, or a "
        "socket error increments a counter and drops one frame. A receiver thread that died "
        "silently would be indistinguishable from an aircraft going quiet.",
    ])
    doc.pagebreak()


# ------------------------------------------------------------------------------------
# 4. Unity setup
# ------------------------------------------------------------------------------------
def section4(doc: Doc):
    doc.h1("Unity Simulation Setup", "4")

    doc.h2("4.1  Install the Scripts")
    doc.p("Three C# files. The third adds a Unity menu that performs the rest of the setup.")

    doc.table(
        ["File", "Copy to", "Purpose"],
        [["`UavStreamer.cs`", "`Assets/Scripts/Mosaicking/`",
          "Captures the camera, reads pose and range, fragments and transmits"],
         ["`UavRandomFlight.cs`", "`Assets/Scripts/Mosaicking/`",
          "Autonomous lawnmower flight, gimbal wander, optical zoom modulation"],
         ["`MosaickingSetup.cs`", "`Assets/Editor/`",
          "Editor menu. **Must sit directly inside a folder named `Editor`** or Unity "
          "compiles it into the runtime assembly and the build fails"]],
        widths=[1.15, 1.35, 3.0], size=8.8)

    doc.steps([
        "In Windows Explorer, open your Unity project folder and find **Assets**.",
        "Inside **Assets**, create **Scripts**, then **Mosaicking** inside that. "
        "Right-click empty space, then **New** then **Folder**.",
        "Back in **Assets**, create a folder named exactly **Editor**.",
        "Copy the three files to the destinations in the table above.",
        "Switch to Unity and wait for the spinner in the bottom-right corner to stop.",
        "Open **Window** then **General** then **Console**. It must show no red errors "
        "before you continue.",
    ])

    doc.h2("4.2  Camera Configuration")
    doc.p("The mapping camera is **separate** from the existing seeker camera. They have "
          "opposing requirements and cannot be the same object.")

    doc.table(
        ["", "Seeker camera (existing)", "Mapping camera (added)"],
        [["Purpose", "Operator view, target tracking", "Ground coverage for the mosaic"],
         ["Orientation", "Forward or operator-slewed", "Nadir, within a small wander cone"],
         ["Field of view", "Whatever the operator selects", "Modulated 16 to 50 degrees vertical"],
         ["Rendering", "Continuous, to a display", "On demand only, at the capture rate"],
         ["Incidence", "Frequently past 65 degrees off nadir", "Held under the gate by design"],
         ["Consequence", "Unusable for mapping - most frames rejected",
          "Every frame usable"]],
        widths=[0.85, 1.9, 1.9], size=8.8)

    doc.p("`Tools` then `Mosaicking` then `Set Up 4-UAV Streaming` creates the mapping "
          "camera as a child of each aircraft, points it down, attaches both scripts, and "
          "assigns identifiers 0 to 3 with ports 5001 to 5004. `Revert 4-UAV Streaming` "
          "removes everything it added. `Check Scene (read-only)` reports what it finds and "
          "changes nothing - run it first.")

    doc.callout("warn", "Universal Render Pipeline requires on-demand capture", [
        "Under URP, calling `Camera.Render()` directly is unsupported and returns black "
        "frames. The streamer instead enables the camera, yields on `WaitForEndOfFrame`, "
        "calls `ReadPixels`, then disables the camera again. The mapping camera is therefore "
        "**disabled between captures** and costs nothing on frames where nothing is sent.",
        "If your project uses the Built-in Render Pipeline instead, this path still works; it "
        "is simply not the cheapest option there.",
    ])

    doc.h2("4.3  UavStreamer Parameters")
    doc.table(
        ["Inspector field", "Default", "Notes"],
        [["**Uav Id**", "0, 1, 2, 3", "Must be unique per aircraft. Selects the port."],
         ["**Eo Camera**", "auto", "The mapping camera. Auto-assigned by the setup menu."],
         ["**Ir Camera**", "empty", "Optional. Backend ignores it unless started with `--ir`."],
         ["**Host**", "`127.0.0.1`", "Ground-station address. Change for a radio link."],
         ["**Eo Port Base**", "5001", "Port used is `eoPortBase + uavId`."],
         ["**Ir Port Base**", "5011", "Same arithmetic for the infrared stream."],
         ["**Mtu**", "1400", "Must match the backend. Do not raise it for a real link."],
         ["**Capture Width / Height**", "1280 x 720", "Reduce for a constrained link."],
         ["**Jpeg Quality**", "65", "See the measured trade-off below."],
         ["**Target Hz**", "10", "Transmission rate. 3 is appropriate over a radio."],
         ["**Stagger**", "true", "Offsets each aircraft's capture phase so all four do not "
                                 "transmit in the same millisecond."],
         ["**Max Range Metres**", "12000", "Laser rangefinder maximum. Beyond it the flag "
                                           "clears and the backend falls to tier 2."],
         ["**Log Stats**", "true", "Prints transmitted frames and bytes to the Unity Console."]],
        widths=[1.5, 0.85, 3.15], size=8.6)

    doc.h3("JPEG quality, measured on this terrain")
    doc.table(
        ["Quality", "Frame", "4 aircraft at 10 Hz", "Mean pixel difference vs q90"],
        [["50", "100 KB", "32.7 Mbit/s", "3.66 / 255"],
         ["**65**", "**~120 KB**", "**~39 Mbit/s**", "**~3.2 / 255**"],
         ["80", "170 KB", "55.7 Mbit/s", "2.56 / 255"],
         ["90", "247 KB", "80.9 Mbit/s", "reference"]],
        widths=[0.7, 0.8, 1.5, 1.9], size=8.8, align=["r", "r", "r", "r"])

    doc.p("Quality 65 is the default. It costs a third less traffic than 80 for a mean pixel "
          "difference below one part in 255, and the canvas resamples to 2.5 m/px in any "
          "case, so the additional bits were being discarded downstream.")

    doc.h2("4.4  Swarm Flight Dynamics")
    doc.p("`UavRandomFlight.cs` flies a lawnmower survey: parallel lanes, one aircraft per "
          "lane, with continuous gimbal wander and optical zoom modulation so that no two "
          "frames share geometry.")

    doc.table(
        ["Inspector field", "Value", "Effect"],
        [["**Mode**", "`Lawnmower`", "Also `Racetrack` and `Wander`. Lawnmower is how real "
                                     "surveys are flown."],
         ["**Unique Seed**", "-1", "-1 derives the seed from `uavId`, so each aircraft takes a "
                                   "different lane and a different slice of the noise field."],
         ["**Survey Half Extent**", "4000 m", "An 8 x 8 km survey box. Must stay comfortably "
                                              "inside the backend's `--extent 7000`."],
         ["**Lane Spacing**", "2000 m", "Width of one aircraft's lane."],
         ["**Lane Count**", "4", "One per aircraft."],
         ["**Passes Per Lane**", "3", "Parallel passes before the lane repeats."],
         ["**Altitude Min / Max**", "4050 / 4350", "**Absolute Unity Y**, not height above "
                                                   "ground. See the callout below."],
         ["**Max Climb Rate**", "14 m/s", "Limits vertical acceleration so pose stays smooth."],
         ["**Speed Min / Max**", "45 / 80 m/s", "Ground speed range."],
         ["**Max Turn Rate**", "9 deg/s", "Turn limit at the lane ends."],
         ["**Max Bank / Pitch**", "25 / 8 deg", "Airframe attitude, visual only - the gimbal "
                                                "is stabilised separately."],
         ["**Gimbal Wander Deg**", "8 deg", "How far off straight-down the gimbal drifts. Well "
                                            "inside the 65 degree rejection gate."],
         ["**Fov Min / Max Deg**", "16 / 50", "Vertical field of view. **Smaller is zoomed "
                                              "in.** This is the optical zoom modulation."]],
        widths=[1.35, 1.0, 3.15], size=8.6)

    doc.h3("Resulting geometry, computed")
    doc.table(
        ["Vertical FOV", "Horizontal FOV", "Footprint", "Aircraft GSD"],
        [["16 deg (zoomed in)", "28.1 deg", "1649 x 928 m", "1.29 m/px"],
         ["25 deg", "43.0 deg", "2601 x 1463 m", "2.03 m/px"],
         ["35 deg", "58.5 deg", "3700 x 2081 m", "2.89 m/px"],
         ["50 deg (zoomed out)", "79.3 deg", "5471 x 3078 m", "4.27 m/px"]],
        widths=[1.3, 1.2, 1.4, 1.2], size=8.8, align=[None, "r", "r", "r"])

    doc.p("Aircraft ground sample distance spans 1.29 to 4.27 m/px, which brackets the "
          "canvas resolution of 2.5 m/px. That is why 2.5 was chosen: zoomed-in frames "
          "contribute genuine detail, and zoomed-out frames are downsampled rather than "
          "upsampled.")

    doc.callout("info", "The 351 m gap is deliberate", [
        "Lane spacing is 2000 m and the narrowest footprint is 1649 m, so at maximum zoom "
        "there is a **351 m strip that no camera observes**. This is the required "
        "no-overlap case, exercised on every run.",
        "The mosaic still places every frame correctly. There is simply unimaged ground "
        "between lanes, rendered black, because nothing looked at it. Reduce `laneSpacing` "
        "below 1649 to close the gap at all zoom levels.",
    ])

    doc.callout("warn", "Altitude: why 4050-4350 rather than the 50-200 m in the brief", [
        "`altitudeMin` and `altitudeMax` are **absolute Unity Y**, not height above ground. "
        "Terrain under the survey box spans Y 283 to 1766 with a mean of 904, so 4050-4350 "
        "puts the aircraft roughly **3150-3450 m above ground**.",
        "Flying at 50-200 m AGL over this 8 km box is geometrically impractical, and the "
        "table below is why. The area, not the altitude, is the binding constraint - if your "
        "survey area shrinks to a few hundred metres, 50-200 m AGL becomes the correct choice "
        "and the pipeline needs no change beyond these two fields and `--extent`.",
    ])

    doc.table(
        ["Height above ground", "Footprint range", "Aircraft GSD", "Lanes for 8 km",
         "Flight time, 4 aircraft"],
        [["**3300 m (delivered)**", "1649 - 5471 m", "1.29 - 4.27 m/px", "5", "0.05 h"],
         ["200 m", "100 - 332 m", "0.08 - 0.26 m/px", "81", "0.75 h"],
         ["50 m", "25 - 83 m", "0.02 - 0.06 m/px", "321", "2.97 h"]],
        widths=[1.35, 1.05, 1.1, 0.8, 1.05], size=8.6,
        align=[None, "r", "r", "r", "r"])

    doc.p("Two things break at low altitude. Coverage time rises by a factor of 60, and "
          "aircraft resolution reaches 0.02 m/px against a 2.5 m/px canvas, so roughly 99 % "
          "of every transmitted pixel is discarded during compositing. Low-altitude survey is "
          "a valid mission; it needs a canvas GSD near 0.05 m and a correspondingly smaller "
          "area of interest, not merely a lower altitude.")

    doc.formula([
        "footprint_width  =  2 · AGL · (W/H) · tan(vFOV / 2)",
        "aircraft_GSD     =  footprint_width / W",
        "",
        "// To retarget: choose the GSD you need, solve for AGL, then set",
        "// altitudeMin/Max = AGL + mean_terrain_elevation.",
    ], "**Formula 4.1** - Retargeting altitude for a different survey area. W and H are "
       "capture width and height in pixels.")
    doc.pagebreak()
