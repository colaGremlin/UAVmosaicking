# Real-Time 4-UAV Direct-Georeferencing Video Mosaicking

Fuses four independent UAV video feeds into one live, metrically-correct canvas where a target
seen by any aircraft reads out at exact world coordinates — **without any feature matching**,
and therefore **without requiring the fields of view to overlap**.

Every transform derives from telemetry: camera pose, intrinsics, and laser-range-finder
measurement. Registration is arithmetic on the pose, not a search over pixels.

```
UNITY (4 drones)                  PYTHON BACKEND                      MISSION PLANNER
UavStreamer.cs ──UDP:5001-4──►  4x daemon receivers                   H.264 / UDP:5600
  EO + IR + LRF                   ↓ reassemble, decode (GIL-free)     ◄──────────────
  pose sampled in the             ↓ FusionEngine @ 10 Hz
  SAME frame as the render        ↓   project → warp (4-way parallel)
  telemetry in EVERY datagram     ↓   composite, max-weight
                                  ↓ HUD → ffmpeg subprocess
```

## Status

Working end to end and validated against ground truth.

| Measure | Result |
|---|---|
| Fusion tick (4 UAVs, 4000×4000 canvas) | **p50 30.3 ms · p90 42.1 ms · p99 48.7 ms** (~3× headroom at 10 Hz) |
| Budget overruns at 10 Hz | **0 of 179 ticks** |
| Mosaic vs ground-truth world | **NCC 0.9961**, correlation peak exactly at 0 m offset |
| Frames fused / rejected / outside AOI | 663 / 0 / 0 |
| Tests | **150 passing** |

Stage breakdown (mean ms): `georef+warp` 21.4 · `composite` 8.1 · `snapshot` 0.01.
Getting here took three measured optimisations, each recorded in the code where it applies:
`cv2.copyTo` over numpy masked assignment (22.9 → 0.28 ms), caching the pose-independent ray
norm (9.6 → 1.0 ms), and subsampling the HUD statistics (`np.unique` at 99 ms → ~2 ms).

## Quick start

```bash
pip install numpy opencv-python pytest      # ffmpeg on PATH for video out

python tools/sim_sender.py --duration 30 &  # stands in for Unity
python -m uavmosaic.app --stats             # fuse + stream to Mission Planner

ffplay -fflags nobuffer -flags low_delay -i udp://127.0.0.1:5600   # or just watch it
```

Headless, saving the mosaic:

```bash
python -m uavmosaic.app --duration 20 --no-encoder --save mosaic.png --stats
```

## Unity setup

Attach `tools/unity/UavStreamer.cs` to each drone, assign the EO camera, set `uavId` to 0–3.
That is the whole integration.

> The C# does **zero** coordinate conversion — it sends raw Unity left-handed values. All
> LH→RH handling lives in one unit-tested module (`uavmosaic/coords.py`). Converting in both
> places would apply the transform twice and produce a mosaic that looks *almost* right, which
> is the worst possible failure mode. `tests/test_unity_wire.py` pins the byte layout the C#
> hardcodes.

## How it works

**Ray → plane.** Each frame corner is back-projected (`K⁻¹[u,v,1]`), rotated into ENU, and
intersected with a ground plane whose height comes from the LRF:

```
λ = (z_plane − z_cam) / d_world.z        G = C + λ·d_world
```

Three independent sources state this identically — Hinzmann Alg. 2, Correia Eq. 60-61, SkyPin
Eq. 4 — and two of the supplied codebases implement it.

**Ground plane, 3-tier cascade.** `lrf_slant` (intersect the boresight at the measured range —
correct for an oblique gimbal) → `agl` → configured default. The tier used is carried on every
target fix, because a fix from an assumed plane deserves less trust than a ranged one.

**One homography per frame is exact, not a shortcut.** GROMS applies a single remap to *flat*
DEM blocks and per-pixel correction only to rugged ones. The LRF gives one plane per frame, so
every block is flat by construction. And `cv2.warpPerspective` samples backward internally, so
this gets backward-projection quality at forward-projection cost — Hinzmann's 25–100× penalty
for backward mapping exists only because his backward path ray-traces a DSM.

**Compositing.** `w = w_radial · w_incidence · w_gsd`, then per pixel:

```python
update = valid & ((owner == uav_id) | (w_new > weight))
```

The `same_owner` clause appears in none of the reference implementations. Without it the mosaic
**freezes under a moving aircraft**: as a UAV banks, its own older higher-weight pixels outrank
its fresh ones. A UAV always overwrites itself; max-weight arbitrates only *between* UAVs.

**Target read-out** back-projects the single ray — it does not interpolate between footprint
corners. Perspective projection is projective, bilinear interpolation is not, and the gap is
not academic:

| Gimbal dip from horizon | Bilinear error |
|---|---|
| 90° (nadir) | 0.00 m |
| 70° | 10.90 m |
| 60° | 21.21 m |
| 30° | 282.48 m |

## Wire protocol

Telemetry and JPEG travel in the **same datagram**, and the 64 B telemetry block repeats in
*every* fragment. That costs ~5% overhead and buys atomicity: nothing downstream can ever pair
the wrong pose with the right image, even under fragment loss.

```
HEADER 36 B   magic 'UAV1' | version | uav_id | sensor_id | flags | frame_id
              t_capture_us | frag_index | frag_count | frag_len | total_len
              telem_len | frag_offset      ← explicit, so the receiver never has to
                                             reverse-engineer the sender's chunking
TELEM  64 B   pos_unity[3] | quat_world_cam[4] | img_w,h | fx,fy,cx,cy
              lrf_slant_m | agl_m | hfov_deg | zoom        (all RAW Unity values)
PAYLOAD       JPEG slice
```

MTU is capped at 1400 B rather than the ~64 KB loopback allows — a protocol that only ever saw
single-datagram frames in simulation would hide the fragmentation bugs that bite on a real link.
A frame is dropped whole if any fragment is missing: a partial JPEG decodes to plausible
garbage, which is worse than nothing.

## Layout

```
uavmosaic/
  coords.py     LH↔RH, ENU↔canvas, ENU↔WGS-84    ← the ONLY place handedness is touched
  camera.py     intrinsics, rebuilt per frame for variable zoom
  georef.py     LRF plane cascade, ray→plane, footprint, gates
  weights.py    radial × incidence × gsd
  canvas.py     fixed buffers + max-weight composite
  fusion.py     tick loop, parallel warps, per-stage p50/p90/p99
  ingest.py     daemon receivers, 1-deep mailboxes
  protocol.py   pack/parse, bounded reassembly
  encoder.py    ffmpeg subprocess → H.264/UDP
  targets.py    exact pixel → world read-out
  hud.py        footprints, staleness, scale bar
  synth.py      ground-truth world + camera renderer (validation)
  app.py        wiring + CLI
tools/
  sim_sender.py         four synthetic feeds over real UDP
  unity/UavStreamer.cs  the Unity sender
```

## Validation

`tests/test_end_to_end.py` renders a known world from four poses, ships it over real UDP
sockets carrying raw Unity values, fuses it, and compares the mosaic **against the source
world**. A sign, axis, scale or ROI error anywhere breaks the correlation. A companion test
shifts the mosaic 12 m and asserts the check *fails*, so the guard cannot pass vacuously.

The geometry itself is pinned independently by closed-form footprints (`h·tan(fov/2)`), a
pixel→ground→pixel round trip exact to **7e-13 px**, and homography-vs-direct-projection
agreement at every interior pixel.

```bash
python -m pytest tests/ -q
```

## Known limits

- **Flat-plane parallax.** An object of height `h` displaces by ≈ `h·tanθ`; a 10 m building at
  30° off-nadir smears ~5.8 m at 200 m altitude. Mitigated by the incidence gate and
  `w_incidence`; genuinely fixed only by a DSM, which is out of scope. GROMS documents exactly
  this failure against Map2DFusion.
- **Accuracy is bounded by telemetry, not by this code.** In Unity the pose is ground truth. On
  real hardware expect the 1–3 m class (Map2DFusion 3.07/5.71 m; GROMS 1.11/0.64 m MAE).
- **IR is protocol-live but dormant.** `--ir` switches on compositing; ports 5011–5014.
- **Hard seams** under differing exposure between aircraft. Unity's identical virtual cameras
  largely avoid this; multi-band blending (GROMS Eq. 8-11) is the designed-in quality toggle.
- **No loop closure or bundle adjustment**, by design — every frame is independently anchored to
  the world, so there is no accumulated drift to correct.

## Sources

`aerial_mapper` (Hinzmann et al.) — λ scale, corner→warp pattern, best-viewing-angle rule.
`Map2DFusion` (Bu et al.) — plane-local frame, incidence gate, radial weight, max-weight fusion.
`Argus` — cleanest Python reference for `R_world←cam`, ROI-local warp, footprint sanity clamp.
`MGRAPH` (Ruiz et al.) — the deliberate counter-example; direct georeferencing is MGRAPH
collapsed to a star graph whose absolute homographies come from telemetry, so graph fusion and
non-linear optimisation are structurally unnecessary rather than skipped.
`GROMS` (Yao et al.) — the block-wise orthorectification argument, and the runtime budget anchor.
`Correia et al.` (Sensors 2022) — the frame-chain spine and `K` from focal length / sensor size.
`SkyPin` (Drones 2026) — third statement of the ray-plane formula; oblique-view error expectations.
