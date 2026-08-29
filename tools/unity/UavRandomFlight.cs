// UavRandomFlight.cs -- survey flight for one UAV.
//
// TARGET: Unity 2022.3 LTS, Universal Render Pipeline. No packages required.
//
// THREE PATTERNS
//   Lawnmower (default)  parallel straight legs with turns at the ends, each aircraft
//                        working its own lane of a shared survey box. This is what real
//                        mapping aircraft fly, and it is the right choice for mosaicking:
//                        constant heading and a steady look angle are exactly what
//                        flat-ground projection wants. A random walk re-covers ground it has
//                        already seen and leaves permanent gaps elsewhere.
//   Racetrack            an elongated loop around a point, for persistent stare rather than
//                        area search. Maps a ring, not an area.
//   Wander               smooth Perlin drift. Organic-looking, poor coverage.
//
// THE NO-OVERLAP CASE IS DELIBERATE
// Direct georeferencing does not need the cameras to see the same ground, and the setup here
// proves it rather than assuming it. Lane spacing is wider than the zoomed-in footprint, so
// when the cameras are zoomed in the four footprints sit in their own lanes with a clear gap
// between neighbours -- no shared imagery at all -- and the mosaic still assembles correctly.
// Zoom out and they overlap heavily. Both regimes occur in every sortie.
//
// WHY THE GIMBAL IS SEPARATE FROM THE AIRFRAME
// A camera bolted to a banking aircraft swings with the bank, and a hard turn would push the
// view past the backend's incidence gate. Real surveillance aircraft carry a stabilised
// gimbal for exactly this reason. This script rolls the airframe for looks and aims the
// gimbal independently at the ground.

using System.Collections.Generic;
using UnityEngine;

public enum UavFlightMode { Lawnmower, Racetrack, Wander }

[AddComponentMenu("UAV Mosaicking/UAV Random Flight")]
[DisallowMultipleComponent]
public class UavRandomFlight : MonoBehaviour
{
    // ---------------------------------------------------------------- inspector

    [Header("Identity")]
    [Tooltip("Which lane this aircraft takes, and which slice of the noise field it uses. "
           + "Leave at -1 to take the Uav Id from the UavStreamer on this object.")]
    public int uniqueSeed = -1;

    [Header("Pattern")]
    public UavFlightMode mode = UavFlightMode.Lawnmower;

    [Header("Survey area (metres, centred on the world origin)")]
    [Tooltip("Half-width of the shared survey box. 4000 gives an 8 x 8 km area.")]
    public float surveyHalfExtent = 4000f;
    [Tooltip("Width of one aircraft's lane. Must exceed the zoomed-in footprint or the four "
           + "footprints can never separate, and the no-overlap case is lost.")]
    public float laneSpacing = 2000f;
    [Tooltip("Aircraft sharing the box. Lanes are laid out symmetrically about the centre.")]
    public int laneCount = 4;
    [Tooltip("Parallel passes within a lane before it repeats. More passes fill the lane more "
           + "evenly and vary the spacing between neighbours, which exercises both the "
           + "overlapping and the non-overlapping case.")]
    [Range(1, 8)] public int passesPerLane = 3;
    [Tooltip("How close the aircraft must get before it turns for the next leg.")]
    public float waypointRadius = 200f;

    [Header("Altitude -- ABSOLUTE Unity Y, not height above ground")]
    [Tooltip("Terrain under the 8 km box spans Y 283-1766 (mean 904), so 4050-4350 puts the "
           + "aircraft about 3300 m above it. Registration over relief improves as this rises; "
           + "image detail improves as it falls.")]
    public float altitudeMin = 4050f;
    public float altitudeMax = 4350f;
    [Range(0.005f, 0.5f)] public float altitudeNoiseRate = 0.03f;
    public float maxClimbRate = 14f;

    [Header("Speed (metres per second)")]
    public float speedMin = 45f;
    public float speedMax = 80f;
    [Range(0.005f, 0.5f)] public float speedNoiseRate = 0.04f;

    [Header("Turning")]
    public float maxTurnRateDeg = 9f;
    [Tooltip("Wander mode only -- how quickly the heading drifts.")]
    [Range(0.005f, 0.5f)] public float headingNoiseRate = 0.04f;

    [Header("Airframe attitude (visual only -- the gimbal is stabilised separately)")]
    public float maxBankDeg = 25f;
    public float maxPitchDeg = 8f;
    [Range(0.2f, 8f)] public float attitudeSmoothing = 1.5f;

    [Header("Sensor gimbal")]
    [Tooltip("Empty GameObject holding the mapping camera. Left empty, the script looks for a "
           + "child named 'MapGimbal', then 'Gimbal'.")]
    public Transform sensorGimbal;
    [Tooltip("How far off straight-down the gimbal may wander. The backend rejects frames past "
           + "65 degrees off nadir, so stay well under that.")]
    [Range(0f, 45f)] public float gimbalWanderDeg = 8f;
    [Range(0.005f, 0.5f)] public float gimbalNoiseRate = 0.02f;

    [Header("Optical zoom (Unity vertical field of view, degrees)")]
    [Tooltip("SMALLER = zoomed IN = narrower footprint. The minimum must be narrow enough that "
           + "the footprint fits inside a lane, or the aircraft never stop overlapping.")]
    public float fovMinDeg = 16f;
    public float fovMaxDeg = 50f;
    [Range(0.002f, 0.2f)] public float zoomNoiseRate = 0.02f;
    public Camera zoomCamera;

    [Header("Diagnostics (read-only)")]
    [SerializeField] private float currentSpeed;
    [SerializeField] private float currentAltitude;
    [SerializeField] private float currentHeadingDeg;
    [SerializeField] private float currentFovDeg;
    [SerializeField] private float footprintWidth;
    [SerializeField] private int waypointIndex;
    [SerializeField] private string coverageNote = "";

    // ---------------------------------------------------------------- state

    private readonly List<Vector2> _route = new List<Vector2>();
    private int _wp;
    private float _seedX, _seedY;
    private float _heading;      // radians, 0 = +Z
    private float _bank, _pitch;
    private int _id;

    // ---------------------------------------------------------------- lifecycle

    void Start()
    {
        _id = ResolveId();
        _seedX = 137.13f * _id + 11.7f;
        _seedY = 71.31f * _id + 3.9f;

        if (sensorGimbal == null)
        {
            Transform t = transform.Find("MapGimbal") ?? transform.Find("Gimbal");
            if (t != null) sensorGimbal = t;
        }
        if (zoomCamera == null)
        {
            UavStreamer s = GetComponent<UavStreamer>();
            if (s != null) zoomCamera = s.eoCamera;
        }

        SanitiseRanges();
        BuildRoute();

        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null && !rb.isKinematic)
        {
            rb.isKinematic = true;
            Debug.Log($"[UavRandomFlight {_id}] Rigidbody set kinematic so scripted flight is "
                    + "not fought by gravity.", this);
        }

        if (_route.Count > 0)
        {
            Vector2 start = _route[0];
            transform.position = new Vector3(start.x, Mathf.Lerp(altitudeMin, altitudeMax, 0.5f), start.y);
            Vector2 next = _route[Mathf.Min(1, _route.Count - 1)] - start;
            _heading = Mathf.Atan2(next.x, next.y);
            _wp = Mathf.Min(1, _route.Count - 1);
        }
        else
        {
            _heading = transform.eulerAngles.y * Mathf.Deg2Rad;
        }

        currentAltitude = transform.position.y;
        ReportCoverage();
    }

    private int ResolveId()
    {
        if (uniqueSeed >= 0) return uniqueSeed;
        UavStreamer s = GetComponent<UavStreamer>();
        if (s != null) return s.uavId;
        return Mathf.Abs(GetInstanceID()) % 4;
    }

    private void SanitiseRanges()
    {
        if (altitudeMax < altitudeMin) { float t = altitudeMin; altitudeMin = altitudeMax; altitudeMax = t; }
        if (speedMax < speedMin) { float t = speedMin; speedMin = speedMax; speedMax = t; }
        if (fovMaxDeg < fovMinDeg) { float t = fovMinDeg; fovMinDeg = fovMaxDeg; fovMaxDeg = t; }
        fovMinDeg = Mathf.Clamp(fovMinDeg, 1f, 170f);
        fovMaxDeg = Mathf.Clamp(fovMaxDeg, 1f, 170f);
        surveyHalfExtent = Mathf.Max(surveyHalfExtent, 100f);
        laneCount = Mathf.Max(laneCount, 1);
        laneSpacing = Mathf.Max(laneSpacing, 50f);
    }

    /// <summary>Reports whether the chosen geometry actually produces a no-overlap case.</summary>
    private void ReportCoverage()
    {
        if (zoomCamera == null) return;
        float agl = Mathf.Lerp(altitudeMin, altitudeMax, 0.5f) - 904f;   // mean ground in this scene
        float narrow = FootprintWidth(agl, fovMinDeg);
        float wide = FootprintWidth(agl, fovMaxDeg);
        float step = laneSpacing / Mathf.Max(passesPerLane, 1);
        coverageNote = narrow < laneSpacing
            ? $"zoomed in {narrow:F0} m < lane {laneSpacing:F0} m -> neighbours separate"
            : $"WARNING zoomed-in footprint {narrow:F0} m fills the {laneSpacing:F0} m lane -- "
            + "they will always overlap";
        if (_id == 0)
        {
            Debug.Log($"[UavRandomFlight] survey {surveyHalfExtent * 2f / 1000f:F1} km box, "
                    + $"{laneCount} lanes of {laneSpacing:F0} m, {passesPerLane} passes "
                    + $"({step:F0} m apart). Footprint {narrow:F0}-{wide:F0} m. {coverageNote}", this);
        }
    }

    private float FootprintWidth(float agl, float vfovDeg)
    {
        if (zoomCamera == null) return 0f;
        float aspect = zoomCamera.pixelWidth > 0 && zoomCamera.pixelHeight > 0
            ? (float)zoomCamera.pixelWidth / zoomCamera.pixelHeight : 16f / 9f;
        float halfV = Mathf.Deg2Rad * vfovDeg * 0.5f;
        return 2f * agl * Mathf.Tan(halfV) * aspect;
    }

    // ---------------------------------------------------------------- route

    private void BuildRoute()
    {
        _route.Clear();
        if (mode == UavFlightMode.Wander) return;

        float laneCentre = (_id % Mathf.Max(laneCount, 1) - (laneCount - 1) * 0.5f) * laneSpacing;

        if (mode == UavFlightMode.Racetrack)
        {
            // Elongated loop, each aircraft phase-shifted a quarter turn around it.
            float a = surveyHalfExtent * 0.75f, b = laneSpacing * 0.9f;
            const int SEGMENTS = 24;
            for (int i = 0; i < SEGMENTS; i++)
            {
                float th = (i / (float)SEGMENTS + _id * 0.25f) * Mathf.PI * 2f;
                _route.Add(new Vector2(laneCentre + Mathf.Cos(th) * b, Mathf.Sin(th) * a));
            }
            return;
        }

        // Lawnmower: parallel legs down the lane, alternating direction, stepping sideways.
        int passes = Mathf.Max(passesPerLane, 1);
        float step = laneSpacing / passes;
        float laneMin = laneCentre - laneSpacing * 0.5f;
        for (int k = 0; k < passes; k++)
        {
            float x = laneMin + (k + 0.5f) * step;
            float far = surveyHalfExtent;
            if (k % 2 == 0) { _route.Add(new Vector2(x, -far)); _route.Add(new Vector2(x, far)); }
            else            { _route.Add(new Vector2(x, far));  _route.Add(new Vector2(x, -far)); }
        }
    }

    // ---------------------------------------------------------------- flight

    void Update()
    {
        float dt = Time.deltaTime;
        if (dt <= 0f) return;
        float t = Time.time;

        currentSpeed = Mathf.Lerp(speedMin, speedMax, Noise(t * speedNoiseRate, 0f));

        float desired;
        if (mode == UavFlightMode.Wander || _route.Count == 0)
        {
            desired = _heading + (Noise(t * headingNoiseRate, 17f) * 2f - 1f) * 0.6f;
            Vector3 p0 = transform.position;
            float edge = surveyHalfExtent - 400f;
            if (Mathf.Max(Mathf.Abs(p0.x), Mathf.Abs(p0.z)) > edge)
                desired = Mathf.Atan2(-p0.x, -p0.z);   // steer home
        }
        else
        {
            Vector2 here = new Vector2(transform.position.x, transform.position.z);
            Vector2 target = _route[_wp];
            if ((target - here).sqrMagnitude < waypointRadius * waypointRadius)
            {
                _wp = (_wp + 1) % _route.Count;
                target = _route[_wp];
            }
            Vector2 d = target - here;
            desired = Mathf.Atan2(d.x, d.y);
        }
        waypointIndex = _wp;

        // Turn toward the desired heading, rate-limited. This is what produces straight legs
        // with rounded turns instead of instant snapping.
        float err = Mathf.DeltaAngle(_heading * Mathf.Rad2Deg, desired * Mathf.Rad2Deg) * Mathf.Deg2Rad;
        float maxTurn = maxTurnRateDeg * Mathf.Deg2Rad * dt;
        float turn = Mathf.Clamp(err, -maxTurn, maxTurn);
        _heading += turn;
        currentHeadingDeg = Mathf.Repeat(_heading * Mathf.Rad2Deg, 360f);

        float targetAlt = Mathf.Lerp(altitudeMin, altitudeMax, Noise(t * altitudeNoiseRate, 43f));
        Vector3 p = transform.position;
        float climb = Mathf.Clamp(targetAlt - p.y, -maxClimbRate * dt, maxClimbRate * dt);

        Vector3 fwd = new Vector3(Mathf.Sin(_heading), 0f, Mathf.Cos(_heading));
        Vector3 next = p + fwd * currentSpeed * dt;
        next.y = p.y + climb;

        float lim = surveyHalfExtent + laneSpacing;   // generous: turns overshoot the box edge
        next.x = Mathf.Clamp(next.x, -lim, lim);
        next.z = Mathf.Clamp(next.z, -lim, lim);
        next.y = Mathf.Clamp(next.y, altitudeMin, altitudeMax);
        transform.position = next;
        currentAltitude = next.y;

        float bankTarget = -(turn / Mathf.Max(maxTurn, 1e-5f)) * maxBankDeg;
        float pitchTarget = -(climb / Mathf.Max(maxClimbRate * dt, 1e-5f)) * maxPitchDeg;
        float k = 1f - Mathf.Exp(-attitudeSmoothing * dt);
        _bank = Mathf.Lerp(_bank, bankTarget, k);
        _pitch = Mathf.Lerp(_pitch, pitchTarget, k);
        transform.rotation = Quaternion.Euler(_pitch, currentHeadingDeg, _bank);

        AimGimbal(t);
        UpdateZoom(t);
    }

    private void AimGimbal(float t)
    {
        if (sensorGimbal == null) return;
        float wx = (Noise(t * gimbalNoiseRate, 91f) * 2f - 1f) * gimbalWanderDeg;
        float wz = (Noise(t * gimbalNoiseRate, 113f) * 2f - 1f) * gimbalWanderDeg;
        // 90 degrees about X points a Unity camera straight down. Set in WORLD space so the
        // airframe's bank does not leak into the sensor line of sight.
        sensorGimbal.rotation = Quaternion.Euler(90f + wx, currentHeadingDeg, wz);
    }

    private void UpdateZoom(float t)
    {
        if (zoomCamera == null) return;
        currentFovDeg = Mathf.Lerp(fovMinDeg, fovMaxDeg, Noise(t * zoomNoiseRate, 197f));
        zoomCamera.fieldOfView = currentFovDeg;
        footprintWidth = FootprintWidth(transform.position.y - 904f, currentFovDeg);
    }

    // ---------------------------------------------------------------- noise

    // Mathf.PerlinNoise clusters around 0.5 and rarely reaches its extremes, so the raw value
    // would leave most of the commanded altitude and zoom range unused. Stretching about the
    // midpoint and clamping restores the full envelope.
    private float Noise(float x, float channel)
    {
        float raw = Mathf.PerlinNoise(_seedX + x, _seedY + channel);
        return Mathf.Clamp01((raw - 0.5f) * 1.8f + 0.5f);
    }

    // ---------------------------------------------------------------- editor gizmo

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0.2f, 0.9f, 1f, 0.8f);
        float mid = (altitudeMin + altitudeMax) * 0.5f;
        Gizmos.DrawWireCube(new Vector3(0f, mid, 0f),
            new Vector3(surveyHalfExtent * 2f, altitudeMax - altitudeMin, surveyHalfExtent * 2f));

        if (!Application.isPlaying) { BuildRoutePreview(); }
        Gizmos.color = new Color(1f, 0.75f, 0.2f, 0.9f);
        for (int i = 0; i + 1 < _route.Count; i++)
            Gizmos.DrawLine(new Vector3(_route[i].x, mid, _route[i].y),
                            new Vector3(_route[i + 1].x, mid, _route[i + 1].y));
        if (_route.Count > 1)
            Gizmos.DrawLine(new Vector3(_route[_route.Count - 1].x, mid, _route[_route.Count - 1].y),
                            new Vector3(_route[0].x, mid, _route[0].y));
    }

    private void BuildRoutePreview()
    {
        _id = ResolveId();
        SanitiseRanges();
        BuildRoute();
    }
}
