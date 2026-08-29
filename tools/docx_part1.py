"""Sections 1 and 2 of the technical documentation."""

from __future__ import annotations

from make_docx import Doc


# ------------------------------------------------------------------------------------
# 1. System overview
# ------------------------------------------------------------------------------------
def section1(doc: Doc):
    doc.h1("System Overview and Core Methodology", "1")

    doc.h2("1.1  Executive Summary")
    doc.p("Four UAVs survey a shared area. Each carries an electro-optical camera on a "
          "stabilised gimbal, a laser rangefinder, and a telemetry source giving position "
          "and attitude. The operational requirement is a single, live, metrically correct "
          "picture of the ground, on which a target seen by any aircraft reads out at true "
          "world coordinates.")
    doc.p("The constraint that decides the design: **the four fields of view may never "
          "overlap**. Aircraft fly separate lanes. At maximum optical zoom the ground "
          "footprint is 1649 m across while lane spacing is 2000 m, leaving a 351 m strip "
          "that no camera sees. Any method that registers images against each other has "
          "nothing to work with in that condition.")
    doc.p("This pipeline registers every frame against the **world**, not against its "
          "neighbours. Camera pose, intrinsics and laser range are sufficient to compute "
          "which patch of ground each pixel observed. Registration becomes arithmetic on "
          "telemetry rather than a search over pixels. Two frames that never touch still "
          "land in correct relative position, because both were placed by the same "
          "arithmetic against the same world frame.")

    doc.table(
        ["Requirement", "Delivered", "Evidence"],
        [["Non-overlapping fields of view", "Supported by construction",
          "Registration never reads a second frame"],
         ["Real-time, 5-10 Hz", "10 Hz, 30.3 ms median tick",
          "Per-stage p50/p90/p99 instrumentation"],
         ["Deterministic mathematics only", "No feature detector in the spatial path",
          "`georef.py` imports no feature module"],
         ["Metric accuracy", "NCC 0.9961, peak at 0 m offset",
          "Synthetic ground-truth comparison"],
         ["Asynchronous daemon architecture", "4 receiver threads + 4-thread warp pool",
          "All release the GIL"],
         ["Explicit handedness resolution", "One module, unit-tested",
          "`coords.py`, golden nadir case"]],
        widths=[1.5, 1.7, 1.9], size=9.0)

    doc.figure("doc_video.png",
               "**Figure 1.1 - Live output.** Four aircraft over real terrain, roughly two "
               "minutes of coverage. The four blocks are individual aircraft lanes. Black is "
               "ground not yet imaged. Numerals mark current aircraft positions.",
               width=4.4)

    doc.h2("1.2  Why Feature-Based Stitching Fails Here")
    doc.p("Classical mosaicking estimates a homography H between two images from point "
          "correspondences: a detector (SIFT, ORB), a descriptor matcher, and RANSAC to "
          "reject outliers. The failure in this application is structural, not a matter of "
          "tuning.")

    doc.formula([
        "H has 8 degrees of freedom  ⇒  a solution needs ≥ 4 correspondences",
        "A correspondence requires the same ground point visible in both images",
        "Disjoint footprints  ⇒  the set of shared ground points is empty",
        "⇒  0 correspondences  ⇒  H is undefined, not merely inaccurate",
    ], "**Formula 1.1** - A rank deficiency, not a noise problem. No detector, descriptor or "
       "robust estimator changes an empty correspondence set.")

    doc.p("Three further problems apply even where images do overlap:")
    doc.bullets([
        "**Drift.** Relative homographies compose. Chaining N frames multiplies N estimates "
        "and error accumulates without bound. Correcting it needs loop closure or bundle "
        "adjustment, neither of which is real-time at fleet scale.",
        "**No absolute anchor.** A feature-matched mosaic is internally consistent but "
        "floats. It cannot report latitude and longitude without external control points.",
        "**Shallow-angle collapse.** The SkyPin study benchmarks matchers at -30 deg to "
        "-60 deg pitch and reports that even the best of them struggles. Oblique gimbal "
        "geometry is the normal case here, not the exception.",
    ])

    doc.callout("info", "MGRAPH is the informative counter-example", [
        "MGRAPH (Ruiz et al., RA-L 2018) is a well-executed graph-based mosaicking system: "
        "ORB features, brute-force Hamming matching, RANSAC, 4-DoF similarity edges, periodic "
        "non-linear optimisation. Its graph edges **exist only where images match**. On a "
        "disjoint four-aircraft survey the graph is four isolated components with no known "
        "relative transform between them.",
        "Useful reframing: direct georeferencing is MGRAPH collapsed to a **star graph**. "
        "There is one virtual reference vertex - the world - and every absolute homography "
        "comes from telemetry instead of a chain of relative edges. Because no edge is a "
        "chained estimate, drift cannot accumulate, so graph fusion and non-linear "
        "refinement are structurally unnecessary rather than merely skipped.",
        "One result from MGRAPH is retained: its Eq. 3 zero-overlap distance "
        "D_max = 2·√(F_w² + F_h²), used here not for registration but as a cheap "
        "predictor of which aircraft pairs will contend for canvas pixels.",
    ])

    doc.h2("1.3  The Direct-Georeferencing Paradigm")
    doc.p("Every frame carries the five quantities that fix its geometry: camera position, "
          "camera orientation, focal length, image size, and range to ground. From those, "
          "each pixel defines a ray in world space, and the intersection of that ray with "
          "the ground surface is the world point the pixel observed.")

    doc.table(
        ["Step", "Operation", "Cost"],
        [["1", "Build K from focal length and image size at the current zoom", "negligible"],
         ["2", "Convert the Unity quaternion to an ENU-to-camera rotation matrix", "negligible"],
         ["3", "Solve the ground plane from the laser range (3-tier cascade, 2.4)", "negligible"],
         ["4", "Cast the **four image corners** to that plane", "4 ray/plane solves"],
         ["5", "`cv2.getPerspectiveTransform` on the four corner pairs", "one 8x8 solve"],
         ["6", "`cv2.warpPerspective` into the canvas region of interest", "dominant, 12-30 ms"],
         ["7", "Composite under the adaptive max-weight rule (2.5)", "2-5 ms"]],
        widths=[0.35, 3.75, 1.05], size=9.0)

    doc.p("Only four rays are cast per frame. The homography derived from those four corner "
          "correspondences is exact for a planar scene, and `warpPerspective` resamples "
          "every interior pixel through it.")

    doc.callout("info", "Why one homography per frame is exact, not an approximation", [
        "GROMS partitions an orthorectification into blocks, measures elevation variance per "
        "block, and applies a single fast remap to flat blocks while reserving per-pixel "
        "correction for rugged ones. A laser-derived ground plane makes **every block flat by "
        "definition**, so one homography per frame is the degenerate-exact case of GROMS's "
        "own fast path rather than a shortcut around it.",
        "A second consequence is worth stating. Hinzmann reports his backward-projection path "
        "as 25-100x slower than his forward homography and treats the two as a quality/speed "
        "trade. That gap exists because his backward path ray-traces against a digital "
        "surface model. On a plane, the backward mapping **is** the inverse homography, and "
        "`cv2.warpPerspective` already samples backward internally. The result is "
        "backward-projection quality at forward-projection cost; the published trade-off does "
        "not apply to this configuration.",
    ])

    doc.figure("compare.png",
               "**Figure 1.2 - Registration validation.** Left: synthetic ground-truth world. "
               "Right: the mosaic fused from four aircraft using telemetry only. Grid "
               "intersections coincide. Normalised cross-correlation 0.9961 with the "
               "correlation peak at exactly 0 m offset, so there is no systematic bias in any "
               "direction.", width=6.3)
    doc.pagebreak()


# ------------------------------------------------------------------------------------
# 2. Mathematical blueprint
# ------------------------------------------------------------------------------------
def section2(doc: Doc):
    doc.h1("Mathematical Blueprint and Coordinate Frames", "2")

    doc.callout("warn", "Frame convention: ENU throughout, never NED", [
        "The world frame is **ENU** - X East, Y North, Z Up - and Z increases upward. NED "
        "does not appear anywhere in the implementation. Every formula in this section, "
        "every sign, and every gate assumes ENU. Substituting NED silently inverts the sign "
        "of the descent test in 2.3 and the mosaic collapses.",
    ])

    doc.h2("2.1  The Four Frames")
    doc.table(
        ["Frame", "Axes", "Handedness", "Origin"],
        [["Unity world", "X right, Y **up**, Z forward", "**Left**", "Unity scene origin"],
         ["Unity camera", "X right, Y **up**, Z along the optical axis", "**Left**", "camera"],
         ["CV camera", "X right, Y **down**, Z along the optical axis", "Right", "camera"],
         ["Local ENU", "X East, Y North, Z Up", "Right", "geodetic anchor"]],
        widths=[0.95, 2.5, 0.85, 1.4], size=9.0)

    doc.p("Two fixed matrices bridge them. Both are involutions (S² = F² = I) and both have "
          "determinant -1, which is exactly what converts left-handed to right-handed.")

    doc.formula([
        "        [ 1  0  0 ]                    [ 1   0  0 ]",
        "  S  =  [ 0  0  1 ]              F  =  [ 0  -1  0 ]",
        "        [ 0  1  0 ]                    [ 0   0  1 ]",
        "",
        "         ENU  <-  Unity world           Unity camera  <-  CV camera",
        "",
        "//  S maps  E = x_u,  N = z_u,  U = y_u.      F flips the camera Y axis.",
        "//  det(S) = -1   and   det(F) = -1   -- both asserted in tests/test_coords.py",
    ], "**Formula 2.1** - The handedness bridge. Defined once, in `uavmosaic/coords.py`.")

    doc.figure("fig_frames.png",
               "**Figure 2.1 - The three frames and the two matrices that connect them.** "
               "Unity is left-handed with Y up; ENU and the CV camera frame are right-handed. "
               "The composition of two determinant -1 matrices with a proper rotation yields a "
               "proper rotation.")

    doc.h2("2.2  The Rotation Chain")
    doc.p("Unity reports q, the camera orientation in Unity world space. Let R_u = R(q) be "
          "Unity's own quaternion-to-matrix conversion, which is self-consistent within its "
          "left-handed frame and maps Unity-camera to Unity-world. Then:")

    doc.formula([
        "R_E←C  =  S · R_u · F",
        "",
        "//  det = (-1)(+1)(-1) = +1, a proper rotation, as required",
        "//  Camera position:   C = S · p_unity",
    ], "**Formula 2.2** - The only rotation that matters. Everything downstream consumes "
       "R_E←C and C.")

    doc.h3("Golden test case, hand-derived")
    doc.p("A nadir camera whose image-up direction points along Unity +Z. This case is "
          "asserted exactly in `tests/test_coords.py::test_nadir_north_up`:")

    doc.formula([
        "          [ 1  0   0 ]                          [ 1   0   0 ]",
        "  R_u  =  [ 0  0  -1 ]        R_E<-C = S R_u F = [ 0  -1   0 ]",
        "          [ 0  1   0 ]                          [ 0   0  -1 ]",
        "",
        "//  X_cv  (image right)  ->  ( 1,  0,  0) = East    correct",
        "//  Y_cv  (image down)   ->  ( 0, -1,  0) = South   correct: image y grows southward",
        "//  Z_cv  (optical axis) ->  ( 0,  0, -1) = Down    correct",
    ], "**Formula 2.3** - If this case is wrong, every mosaic is mirrored or rotated. It is "
       "checked on every test run, along with orthonormality and det = +1 for randomised "
       "quaternions.")

    doc.callout("info", "One design rule removes an entire class of bug", [
        "The Unity C# sender performs **zero** coordinate conversion. It transmits raw Unity "
        "floats. Every left-to-right-handed operation lives in one Python module, `coords.py`, "
        "behind unit tests. Correia et al. spend five pages on frame chains precisely because "
        "that conversion is where projects break; splitting it across two languages and two "
        "codebases is how it becomes unfindable.",
    ])

    doc.h2("2.3  Intrinsics and Ray/Plane Intersection")
    doc.p("Intrinsics are recomputed per frame, because optical zoom changes focal length "
          "in flight (Correia Eq. 32-33):")

    doc.formula([
        "  f_x = f[mm] * W[px] / SensorWidth[mm]              [ f_x   0    c_x ]",
        "  f_y = f[mm] * H[px] / SensorHeight[mm]       K  =  [  0   f_y   c_y ]",
        "                                                    [  0    0     1  ]",
        "",
        "//  If Unity reports horizontal FOV instead:  f_x = (W/2) / tan(FOV_h / 2)",
        "//  Both are transmitted, so the backend cross-checks and warns on disagreement.",
    ], "**Formula 2.4** - K is rebuilt every frame. EO and IR carry separate blocks.")

    doc.p("For each of the four image corners (u, v):")

    doc.formula([
        "d_c  =  K⁻¹ [u, v, 1]ᵀ                       ray in the CV camera frame",
        "d_w  =  R_E←C · d_c                          ray in ENU",
        "",
        "reject the frame if  d_w.z ≥ -ε              the ray does not descend",
        "cosθ  =  -d_w.z / ‖d_w‖                      incidence angle",
        "reject the frame if  θ > θ_max = 65°",
        "",
        "λ  =  (z_plane - C.z) / d_w.z               λ > 0 since C.z > z_plane, d_w.z < 0",
        "G  =  C + λ · d_w                            the ground point, in ENU",
    ], "**Formula 2.5** - Ray/plane intersection. Three independent sources state this "
       "identically: Hinzmann Alg. 2, Correia Eq. 60-61, and SkyPin Eq. 4.")

    doc.figure("fig_raypl.png",
               "**Figure 2.2 - Ray/plane intersection and the ground-plane cascade.** Four "
               "corner rays are intersected with the assumed plane; the resulting quadrilateral "
               "becomes the homography target. The true terrain, shown beneath, is what the "
               "plane approximates - the source of the error analysed in 6.1.")

    doc.p("ENU is then converted to canvas pixels by a fixed affine transform. This is the "
          "reason the area of interest is fixed rather than growing: the mapping never "
          "changes, so no frame's placement depends on when it arrived.")

    doc.formula([
        "x_px  =  (E - E_min) / GSD",
        "y_px  =  (N_max - N) / GSD              // N is flipped: image y grows southward",
        "",
        "H  =  cv2.getPerspectiveTransform(src_corners, dst_corners_in_ROI)",
        "warped  =  cv2.warpPerspective(frame, H, (roi_w, roi_h))",
    ], "**Formula 2.6** - ENU to canvas. The warp is done into a region of interest, never "
       "into the full 5600 x 5600 canvas.")

    doc.callout("warn", "Guards on the footprint", [
        "**Descent test.** A ray with d_w.z >= 0 points at or above the horizon and has no "
        "ground intersection. The frame is rejected.",
        "**Incidence gate.** Rejected past 65 deg off nadir, following Map2DFusion's "
        "axis.dot(downLook) < 0.4 threshold. On the delivered configuration the worst "
        "measured corner was 48.5 deg, so nothing is rejected in normal flight.",
        "**Extent clamp.** A footprint larger than 20x the nadir diagonal indicates a "
        "near-horizon ray. The frame is retried using only the **lower half** of the image - "
        "the ground-facing part under a tilted gimbal - and dropped if that also fails.",
    ])

    doc.h2("2.4  Three-Tier Ground-Plane Cascade")
    doc.p("The boresight direction in world space is the third column of R_E←C. The first "
          "rule that yields a valid range wins:")

    doc.formula([
        "d_bore  =  R_E←C · [0, 0, 1]ᵀ",
        "",
        "1.  slant range valid AND d_bore.z < -sin(10°):   z_plane = (C + r · d_bore).z",
        "2.  elif AGL valid:                                z_plane = C.z - AGL",
        "3.  else:                                          z_plane = z_default",
    ], "**Formula 2.7** - Implemented in `georef.solve_ground_plane`.")

    doc.table(
        ["Tier", "Source", "Correct when", "Failure mode"],
        [["1", "Laser slant range along the boresight",
          "Always, including an oblique gimbal - it takes the elevation of the point the "
          "boresight actually strikes",
          "Returns nothing over water or beyond maximum range"],
         ["2", "Nadir AGL probe",
          "Exact at nadir",
          "Degrades over sloped terrain, because it measures beneath the aircraft rather "
          "than where the camera looks"],
         ["3", "Configured area elevation (`--plane-z`)",
          "A last resort that keeps the frame usable",
          "Fixed value; error equals local relief"]],
        widths=[0.3, 1.15, 2.25, 1.9], size=8.8)

    doc.p("Each aircraft solves its own plane, every frame. There is no single global plane. "
          "The `d_bore.z < -sin(10°)` condition on tier 1 prevents a near-horizontal "
          "boresight from projecting the range to an absurd elevation. The tier actually "
          "used is logged and shown in the engineering overlay.")

    doc.h2("2.5  Adaptive Weight and the Compositing Rule")
    doc.p("Where two aircraft see the same ground, one pixel must win. The weight is the "
          "product of three terms:")

    doc.formula([
        "w(u,v)  =  w_radial(u,v)  ·  w_incidence(u,v)  ·  w_gsd",
    ], "**Formula 2.8** - Map2DFusion's paper describes an adaptive weight over height, view "
       "angle and pixel location, but its released code ships only the radial term. The other "
       "two are implemented here.")

    doc.table(
        ["Term", "Form", "Rationale", "Cost"],
        [["w_radial", "(1 - r/r_max)², floored at 1e-5",
          "Lens falloff and residual distortion are worst at the frame border",
          "Cached once per image size"],
         ["w_incidence", "max(0, cosθ(u,v))²",
          "Hinzmann's closest-to-nadir rule, made continuous rather than a hard choice",
          "Evaluated on a 1/16-scale grid, then resized"],
         ["w_gsd", "gsd_ref / gsd_frame",
          "A lower-flying or zoomed-in aircraft resolves more detail and should win",
          "One scalar per frame"]],
        widths=[0.8, 1.4, 2.6, 1.2], size=8.8)

    doc.p("Compositing is deliberately single-threaded. The rule is a read-modify-write on "
          "shared pixels, and two threads writing overlapping regions would race. It is also "
          "cheap, so there is nothing to gain from parallelising it.")

    doc.code(
        "same_owner = (OWNER_roi == uav_id)\n"
        "update     = valid & (same_owner | (w_new > WEIGHT_roi))\n"
        "\n"
        "CANVAS_roi[update] = warped[update]\n"
        "WEIGHT_roi[update] = w_new[update]\n"
        "OWNER_roi[update]  = uav_id\n"
        "STAMP_roi[update]  = t_now",
        "**Listing 2.1** - The canvas update rule, from `uavmosaic/canvas.py`. In production "
        "the masked assignment is `cv2.copyTo`, which measured 0.28 ms against 22.88 ms for "
        "`np.copyto(where=mask[:,:,None])` - an 80x difference on the same operation.")

    doc.callout("stop", "The same-owner clause is essential and is absent from every "
                        "reference implementation", [
        "Without it, an aircraft's own stale high-weight pixels block its fresh frames the "
        "moment it drifts to a marginally worse viewing angle. The mosaic **freezes beneath "
        "the aircraft** while it is still transmitting - a failure that looks like a dropped "
        "link but is not one.",
        "The rule: an aircraft always overwrites itself. Maximum weight arbitrates only "
        "*between* different aircraft.",
    ])

    doc.h3("Seam feathering")
    doc.p("A hard weight comparison produces a visible brightness step where two frames "
          "meet, because the same ground photographed at different zoom or angle returns "
          "slightly different exposure. The composite cross-fades over a band proportional "
          "to the peak weight:")

    doc.code(
        "alpha = clip((w_new - w_cur) / (feather * w_peak) + 0.5, 0, 1) * valid\n"
        "alpha[valid & (w_cur <= 0)]      = 1.0    # never blend against unwritten canvas\n"
        "alpha[valid & (owner == uav_id)] = 1.0    # never blend against one's own stale pixels\n"
        "cv2.blendLinear(warped, dst, alpha, 1 - alpha, dst=dst)",
        "**Listing 2.2** - Division-free alpha ramp. Default `--feather 0.3`. Measured effect: "
        "the photometric step across a seam falls from 108.5 to 53.6 levels out of 255, for "
        "about 1 ms per tick.")

    doc.callout("warn", "Both guard lines were found by measurement, not by inspection", [
        "The first version blended against the unwritten black canvas at the edge of coverage. "
        "End-to-end correlation against ground truth fell from 0.995 to 0.885 - the mosaic was "
        "measurably worse while looking smoother. Both `putmask` lines above exist to prevent "
        "that, and the end-to-end test now asserts the correlation floor.",
    ])
    doc.pagebreak()
