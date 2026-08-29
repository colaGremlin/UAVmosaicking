// UavStreamer.cs -- attach one to each drone in Unity.
//
// Renders the EO camera, JPEG-encodes it, and streams it to the Python mosaicking backend
// over UDP with the pose embedded in EVERY datagram.
//
// TARGET: Unity 2022.3 LTS, Universal Render Pipeline (URP).
//
// WHY THIS IS WRITTEN FOR URP SPECIFICALLY
// In the old Built-in pipeline you could call Camera.Render() to render a camera on demand.
// URP does not support that -- it silently produces nothing, which shows up as a black or
// frozen video feed with no error message. So instead:
//
//   * each sensor camera keeps a Render Texture permanently assigned, and
//   * the Camera component is left DISABLED, so it costs nothing on ordinary frames, and
//   * the capture coroutine enables it for exactly one frame, waits for end-of-frame so URP
//     has finished drawing into the Render Texture, reads the pixels back, then disables it.
//
// That gives on-demand rendering at 10 Hz on a pipeline that has no on-demand render call.
//
// DESIGN NOTE -- read this before "tidying" the coordinate handling.
// This script performs ZERO coordinate conversion. It sends transform.position and
// transform.rotation exactly as Unity reports them, in Unity's left-handed frame. All
// LH -> RH conversion happens in one place on the Python side (uavmosaic/coords.py), which
// is unit-tested against hand-derived golden cases. Converting here as well would apply the
// transform twice, and the resulting mosaic would look *almost* right -- the worst possible
// failure mode. If a mosaic ever comes out mirrored, fix coords.py, not this file.
//
// The pose is sampled at end-of-frame, which is exactly the transform URP rendered with, and
// it travels in the same datagram as the pixels, so nothing downstream can mispair them.

using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

[AddComponentMenu("UAV Mosaicking/UAV Streamer")]
[DisallowMultipleComponent]
public class UavStreamer : MonoBehaviour
{
    // ---------------------------------------------------------------- inspector

    [Header("Identity")]
    [Tooltip("0..3. Must match the backend port map: EO 5001+uavId, IR 5011+uavId.")]
    [Range(0, 15)] public int uavId = 0;

    [Header("Cameras (leave IR empty unless the backend is run with --ir)")]
    [Tooltip("The DOWNWARD-LOOKING sensor camera. NOT your chase/follow camera.")]
    public Camera eoCamera;
    public Camera irCamera;

    [Header("Network")]
    public string host = "127.0.0.1";
    [Tooltip("EO port = eoPortBase + uavId. Backend default is 5001.")]
    public int eoPortBase = 5001;
    [Tooltip("IR port = irPortBase + uavId. Backend default is 5011.")]
    public int irPortBase = 5011;
    [Tooltip("Bytes per datagram including 100 B of header+telemetry. 1400 matches the backend.")]
    public int mtu = 1400;

    [Header("Capture")]
    [Range(1f, 30f)] public float sendHz = 10f;
    public int captureWidth = 1280;
    public int captureHeight = 720;
    [Tooltip("65 is the sweet spot for this pipeline. Measured on real terrain at 1280x720: "
           + "q80 gives 170 KB frames (56 Mbit/s across four aircraft), q60 gives 114 KB "
           + "(37 Mbit/s) for a mean pixel difference under 1 part in 255. The mosaic "
           + "downsamples the imagery anyway, so the extra bits buy nothing you can see.")]
    [Range(1, 100)] public int jpegQuality = 65;
    [Tooltip("Spreads the four UAVs' captures across different frames so they do not all "
           + "stall the GPU on the same one. Leave on.")]
    public bool stagger = true;

    [Header("Sensors")]
    [Tooltip("Layers the laser range finder and AGL probe can see. Set this to your Terrain "
           + "layer ONLY, so the ray cannot hit the aircraft itself.")]
    public LayerMask terrainMask = ~0;
    [Tooltip("Must exceed your altitude. At 2000 m flying height, 12000 is comfortable.")]
    public float maxRangeMetres = 12000f;

    [Header("Diagnostics")]
    public bool logStats = true;
    [SerializeField] private int framesSent;
    [SerializeField] private int packetsSent;
    [SerializeField] private float lastFrameKB;
    [SerializeField] private string lastError = "";

    // ---------------------------------------------------------------- wire constants
    // These mirror uavmosaic/protocol.py exactly, and tests/test_unity_wire.py pins the
    // byte layout. Changing one side without the other breaks the link silently: the backend
    // just counts bad packets while the operator sees a black screen.

    private const uint Magic = 0x31564155;   // 'UAV1' little-endian
    private const byte Version = 1;
    private const int HeaderSize = 36;
    private const int TelemSize = 64;

    private const byte SensorEO = 0;
    private const byte SensorIR = 1;

    private const byte FlagLrfValid = 0x01;
    private const byte FlagAglValid = 0x02;
    private const byte FlagTelemPresent = 0x04;

    // ---------------------------------------------------------------- state

    private UdpClient _udp;
    private IPEndPoint _eoEndpoint, _irEndpoint;
    private RenderTexture _eoRT, _irRT;
    private Texture2D _eoTex, _irTex;
    private readonly uint[] _frameId = new uint[2];
    private byte[] _packet;
    private double _startTime;
    private Coroutine _loop;

    // ---------------------------------------------------------------- lifecycle

    void OnEnable()
    {
        if (eoCamera == null)
        {
            lastError = "No EO camera assigned.";
            Debug.LogError($"[UavStreamer {uavId}] {lastError} Assign the downward-looking "
                         + "sensor camera to the 'Eo Camera' field. Disabling.", this);
            enabled = false;
            return;
        }

        if (captureWidth <= 0 || captureHeight <= 0)
        {
            Debug.LogError($"[UavStreamer {uavId}] captureWidth/Height must be positive.", this);
            enabled = false;
            return;
        }

        WarnAboutClipPlane(eoCamera, "EO");
        if (irCamera != null) WarnAboutClipPlane(irCamera, "IR");

        try
        {
            _udp = new UdpClient();
            _udp.Client.SendBufferSize = 4 << 20;
            IPAddress addr = IPAddress.Parse(host);
            _eoEndpoint = new IPEndPoint(addr, eoPortBase + uavId);
            _irEndpoint = new IPEndPoint(addr, irPortBase + uavId);
        }
        catch (Exception e)
        {
            lastError = "Socket setup failed: " + e.Message;
            Debug.LogError($"[UavStreamer {uavId}] {lastError}", this);
            enabled = false;
            return;
        }

        _eoRT = NewRT();
        _eoTex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        eoCamera.targetTexture = _eoRT;
        eoCamera.enabled = false;   // the coroutine turns it on for one frame at a time

        if (irCamera != null)
        {
            _irRT = NewRT();
            _irTex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
            irCamera.targetTexture = _irRT;
            irCamera.enabled = false;
        }

        _packet = new byte[mtu];
        _startTime = Time.timeAsDouble;
        lastError = "";

        Debug.Log($"[UavStreamer {uavId}] EO -> {_eoEndpoint}"
                + (irCamera != null ? $", IR -> {_irEndpoint}" : "")
                + $" @ {sendHz} Hz, {captureWidth}x{captureHeight}, mtu {mtu}", this);

        _loop = StartCoroutine(CaptureLoop());
    }

    void OnDisable()
    {
        if (_loop != null) { StopCoroutine(_loop); _loop = null; }

        if (eoCamera != null) eoCamera.targetTexture = null;
        if (irCamera != null) irCamera.targetTexture = null;

        if (_udp != null) { _udp.Close(); _udp = null; }
        if (_eoRT != null) { _eoRT.Release(); Destroy(_eoRT); _eoRT = null; }
        if (_irRT != null) { _irRT.Release(); Destroy(_irRT); _irRT = null; }
        if (_eoTex != null) { Destroy(_eoTex); _eoTex = null; }
        if (_irTex != null) { Destroy(_irTex); _irTex = null; }
    }

    private void WarnAboutClipPlane(Camera cam, string label)
    {
        // The single most common cause of an all-black feed: Unity's default far clip plane
        // is 1000 m, so from 1800 m up the ground is simply not drawn.
        float needed = Mathf.Max(transform.position.y, 100f) * 2f;
        if (cam.farClipPlane < needed)
        {
            Debug.LogWarning(
                $"[UavStreamer {uavId}] {label} camera Clipping Planes > Far is "
              + $"{cam.farClipPlane:F0} m but you are flying at {transform.position.y:F0} m. "
              + $"The ground will not render and the feed will be black. Set Far to at least "
              + $"{needed:F0}.", cam);
        }
    }

    private RenderTexture NewRT()
    {
        var rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1;
        rt.Create();
        return rt;
    }

    // ---------------------------------------------------------------- capture loop

    private IEnumerator CaptureLoop()
    {
        // Stagger the four UAVs so they do not all read back from the GPU on the same frame.
        if (stagger && sendHz > 0f)
        {
            float slice = (1f / sendHz) * (uavId % 4) / 4f;
            if (slice > 0f) yield return new WaitForSeconds(slice);
        }

        var endOfFrame = new WaitForEndOfFrame();
        double next = Time.timeAsDouble;

        while (true)
        {
            double period = 1.0 / Mathf.Max(sendHz, 0.1f);
            while (Time.timeAsDouble < next) yield return null;
            next = Time.timeAsDouble + period;

            bool wantIr = irCamera != null;
            eoCamera.enabled = true;
            if (wantIr) irCamera.enabled = true;

            // URP draws the enabled cameras during this frame's render loop, which runs after
            // every Update and LateUpdate. Waiting for end-of-frame guarantees the Render
            // Textures are populated AND that the transform we read below is precisely the one
            // that was rendered -- sampling earlier could pair pixels with a stale pose.
            yield return endOfFrame;

            eoCamera.enabled = false;
            if (wantIr) irCamera.enabled = false;

            GrabAndSend(eoCamera, _eoRT, _eoTex, SensorEO, _eoEndpoint);
            if (wantIr) GrabAndSend(irCamera, _irRT, _irTex, SensorIR, _irEndpoint);
        }
    }

    private void GrabAndSend(Camera cam, RenderTexture rt, Texture2D tex, byte sensorId,
                             IPEndPoint endpoint)
    {
        Vector3 pos = cam.transform.position;
        Quaternion rot = cam.transform.rotation;

        RenderTexture prevActive = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        tex.Apply(false);
        RenderTexture.active = prevActive;

        byte[] jpeg = tex.EncodeToJPG(jpegQuality);
        if (jpeg == null || jpeg.Length == 0)
        {
            lastError = "JPEG encode returned nothing.";
            Debug.LogWarning($"[UavStreamer {uavId}] {lastError}", this);
            return;
        }
        lastFrameKB = jpeg.Length / 1024f;

        // Intrinsics. Unity's Camera.fieldOfView is the VERTICAL fov in degrees; the backend
        // wants fx/fy/cx/cy in pixels. fy = (H/2) / tan(vfov/2). Square pixels => fx == fy,
        // and the horizontal fov then follows from the capture aspect ratio. The backend
        // cross-checks the reported hfov against fx and warns on a mismatch, which catches a
        // physical-camera or aspect misconfiguration on the very first frame.
        float vfovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float fy = (captureHeight * 0.5f) / Mathf.Tan(vfovRad * 0.5f);
        float fx = fy;
        float cx = captureWidth * 0.5f;
        float cy = captureHeight * 0.5f;
        float hfovDeg = 2f * Mathf.Atan((captureWidth * 0.5f) / fx) * Mathf.Rad2Deg;

        // LRF: slant range along the camera boresight (+Z of the camera in Unity).
        float lrf = 0f; bool lrfValid = false;
        RaycastHit hit;
        if (Physics.Raycast(pos, cam.transform.forward, out hit, maxRangeMetres, terrainMask))
        {
            lrf = hit.distance; lrfValid = true;
        }

        // AGL: straight down, independent of where the gimbal happens to be looking.
        float agl = 0f; bool aglValid = false;
        RaycastHit down;
        if (Physics.Raycast(pos, Vector3.down, out down, maxRangeMetres, terrainMask))
        {
            agl = down.distance; aglValid = true;
        }

        byte flags = FlagTelemPresent;
        if (lrfValid) flags |= FlagLrfValid;
        if (aglValid) flags |= FlagAglValid;

        // Reported for telemetry only; the backend derives everything from fx/fy.
        float zoom = 1f;

        byte[] telem = PackTelemetry(pos, rot, fx, fy, cx, cy, lrf, agl, hfovDeg, zoom);
        ulong tCaptureUs = (ulong)((Time.timeAsDouble - _startTime) * 1e6);

        _frameId[sensorId]++;
        SendFragmented(jpeg, telem, flags, sensorId, _frameId[sensorId], tCaptureUs, endpoint);

        framesSent++;
        if (logStats && framesSent % 100 == 0)
        {
            Debug.Log($"[UavStreamer {uavId}] {framesSent} frames, {packetsSent} packets, "
                    + $"last {lastFrameKB:F1} KB, fov {cam.fieldOfView:F1}d, "
                    + $"alt {pos.y:F0} m, lrf {(lrfValid ? lrf.ToString("F0") + " m" : "MISS")}", this);
        }

        if (!lrfValid && !aglValid && framesSent % 50 == 1)
        {
            Debug.LogWarning($"[UavStreamer {uavId}] neither LRF nor AGL ray hit anything. "
                           + "Check Terrain Mask and that your ground has a Collider. The "
                           + "backend will fall back to a flat assumed plane.", this);
        }
    }

    // ---------------------------------------------------------------- protocol

    private byte[] PackTelemetry(Vector3 pos, Quaternion rot, float fx, float fy,
                                 float cx, float cy, float lrf, float agl,
                                 float hfovDeg, float zoom)
    {
        var b = new byte[TelemSize];
        int o = 0;
        // RAW Unity values -- see the design note at the top of this file.
        o = Put(b, o, pos.x); o = Put(b, o, pos.y); o = Put(b, o, pos.z);
        o = Put(b, o, rot.x); o = Put(b, o, rot.y); o = Put(b, o, rot.z); o = Put(b, o, rot.w);
        o = Put(b, o, (ushort)captureWidth); o = Put(b, o, (ushort)captureHeight);
        o = Put(b, o, fx); o = Put(b, o, fy); o = Put(b, o, cx); o = Put(b, o, cy);
        o = Put(b, o, lrf); o = Put(b, o, agl); o = Put(b, o, hfovDeg); o = Put(b, o, zoom);
        return b;
    }

    private void SendFragmented(byte[] jpeg, byte[] telem, byte flags, byte sensorId,
                                uint frameId, ulong tCaptureUs, IPEndPoint endpoint)
    {
        int chunk = mtu - HeaderSize - TelemSize;
        if (chunk <= 0)
        {
            Debug.LogError($"[UavStreamer {uavId}] mtu {mtu} is too small for "
                         + $"{HeaderSize + TelemSize} B of framing.", this);
            return;
        }
        int total = jpeg.Length;
        int n = Mathf.Max(1, (total + chunk - 1) / chunk);
        if (n > ushort.MaxValue)
        {
            Debug.LogError($"[UavStreamer {uavId}] frame needs {n} fragments; lower "
                         + "Capture Width/Height or Jpeg Quality.", this);
            return;
        }

        for (int i = 0; i < n; i++)
        {
            int offset = i * chunk;
            int len = Mathf.Min(chunk, total - offset);
            if (len < 0) len = 0;

            int o = 0;
            o = Put(_packet, o, Magic);
            _packet[o++] = Version;
            _packet[o++] = (byte)uavId;
            _packet[o++] = sensorId;
            _packet[o++] = flags;
            o = Put(_packet, o, frameId);
            o = Put(_packet, o, tCaptureUs);
            o = Put(_packet, o, (ushort)i);
            o = Put(_packet, o, (ushort)n);
            o = Put(_packet, o, (ushort)len);
            o = Put(_packet, o, (uint)total);
            o = Put(_packet, o, (ushort)TelemSize);
            o = Put(_packet, o, (uint)offset);   // explicit byte offset: the receiver never
                                                 // has to reverse-engineer our chunking

            Buffer.BlockCopy(telem, 0, _packet, o, TelemSize);
            o += TelemSize;
            if (len > 0) Buffer.BlockCopy(jpeg, offset, _packet, o, len);

            try
            {
                _udp.Send(_packet, o + len, endpoint);
                packetsSent++;
            }
            catch (SocketException e)
            {
                lastError = "send failed: " + e.Message;
                Debug.LogWarning($"[UavStreamer {uavId}] {lastError}", this);
                return;
            }
        }
    }

    // ---------------------------------------------------------------- little-endian writers
    // BitConverter is little-endian on every platform Unity ships for, but writing the bytes
    // explicitly keeps the wire format correct even on a big-endian target.

    private static int Put(byte[] b, int o, uint v)
    {
        b[o] = (byte)v; b[o + 1] = (byte)(v >> 8);
        b[o + 2] = (byte)(v >> 16); b[o + 3] = (byte)(v >> 24);
        return o + 4;
    }

    private static int Put(byte[] b, int o, ushort v)
    {
        b[o] = (byte)v; b[o + 1] = (byte)(v >> 8);
        return o + 2;
    }

    private static int Put(byte[] b, int o, ulong v)
    {
        for (int i = 0; i < 8; i++) b[o + i] = (byte)(v >> (8 * i));
        return o + 8;
    }

    private static int Put(byte[] b, int o, float v)
    {
        uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(v), 0);
        return Put(b, o, bits);
    }
}
