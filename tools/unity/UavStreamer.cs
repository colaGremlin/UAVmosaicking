// UavStreamer.cs -- attach one to each drone in Unity.
//
// Renders the EO camera, JPEG-encodes it, and streams it to the Python mosaicking backend
// over UDP with the pose embedded in EVERY datagram.
//
// SETUP
//   1. Drop this on the drone GameObject (or on the camera itself).
//   2. Assign `eoCamera`. Leave `irCamera` empty until IR is switched on in the backend.
//   3. Set `uavId` to 0..3, matching the backend ports 5001..5004.
//   4. Press Play. Run the backend:  python -m uavmosaic.app
//
// DESIGN NOTE -- read this before "tidying" the coordinate handling.
// This script performs ZERO coordinate conversion. It sends `transform.position` and
// `transform.rotation` exactly as Unity reports them, in Unity's left-handed frame. All
// LH -> RH conversion happens in one place on the Python side (uavmosaic/coords.py), which
// is unit-tested against hand-derived golden cases. Converting here as well would apply the
// transform twice, and the resulting mosaic would look *almost* right -- the worst possible
// failure mode. If a mosaic ever comes out mirrored, fix coords.py, not this file.
//
// The pose is sampled in the SAME frame as the render, so pose and pixels are always
// consistent, and both travel in the same datagram so nothing downstream can mispair them.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using UnityEngine;
using UnityEngine.Rendering;

[AddComponentMenu("UAV Mosaicking/UAV Streamer")]
public class UavStreamer : MonoBehaviour
{
    // ---------------------------------------------------------------- inspector

    [Header("Identity")]
    [Tooltip("0..3. Must match the backend port map: EO 5001+uavId, IR 5011+uavId.")]
    [Range(0, 3)] public int uavId = 0;

    [Header("Cameras")]
    public Camera eoCamera;
    [Tooltip("Leave empty unless the backend is run with --ir.")]
    public Camera irCamera;

    [Header("Network")]
    public string host = "127.0.0.1";
    public int eoPortBase = 5001;
    public int irPortBase = 5011;
    [Tooltip("Bytes per datagram including the 100 B of header+telemetry. 1400 matches the backend default.")]
    public int mtu = 1400;

    [Header("Capture")]
    [Range(1f, 30f)] public float sendHz = 10f;
    public int captureWidth = 1280;
    public int captureHeight = 720;
    [Range(1, 100)] public int jpegQuality = 80;

    [Header("Sensors")]
    [Tooltip("Layers the laser range finder and AGL probe can see. Exclude the drone itself.")]
    public LayerMask terrainMask = ~0;
    public float maxRangeMetres = 5000f;

    [Header("Diagnostics")]
    public bool logStats = true;
    [SerializeField] private int framesSent;
    [SerializeField] private int packetsSent;
    [SerializeField] private float lastFrameKB;

    // ---------------------------------------------------------------- constants
    // These mirror uavmosaic/protocol.py exactly. Changing one without the other breaks
    // the link silently -- the backend just counts bad packets.

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
    private float _nextSendTime;
    private double _startTime;

    // ---------------------------------------------------------------- lifecycle

    void OnEnable()
    {
        if (eoCamera == null) eoCamera = GetComponentInChildren<Camera>();
        if (eoCamera == null)
        {
            Debug.LogError($"[UavStreamer {uavId}] no EO camera assigned; disabling.");
            enabled = false;
            return;
        }

        _udp = new UdpClient();
        _udp.Client.SendBufferSize = 4 << 20;
        var addr = IPAddress.Parse(host);
        _eoEndpoint = new IPEndPoint(addr, eoPortBase + uavId);
        _irEndpoint = new IPEndPoint(addr, irPortBase + uavId);

        _eoRT = NewRT();
        _eoTex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        if (irCamera != null)
        {
            _irRT = NewRT();
            _irTex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        }

        _packet = new byte[mtu];
        _startTime = Time.timeAsDouble;
        _nextSendTime = 0f;

        Debug.Log($"[UavStreamer {uavId}] EO -> {_eoEndpoint}" +
                  (irCamera != null ? $", IR -> {_irEndpoint}" : "") +
                  $" @ {sendHz} Hz, {captureWidth}x{captureHeight}");
    }

    void OnDisable()
    {
        _udp?.Close(); _udp = null;
        if (_eoRT != null) { _eoRT.Release(); Destroy(_eoRT); _eoRT = null; }
        if (_irRT != null) { _irRT.Release(); Destroy(_irRT); _irRT = null; }
        if (_eoTex != null) { Destroy(_eoTex); _eoTex = null; }
        if (_irTex != null) { Destroy(_irTex); _irTex = null; }
    }

    private RenderTexture NewRT()
    {
        var rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        rt.Create();
        return rt;
    }

    void LateUpdate()
    {
        if (Time.time < _nextSendTime) return;
        _nextSendTime = Time.time + 1f / Mathf.Max(sendHz, 0.1f);

        CaptureAndSend(eoCamera, _eoRT, _eoTex, SensorEO, _eoEndpoint);
        if (irCamera != null)
            CaptureAndSend(irCamera, _irRT, _irTex, SensorIR, _irEndpoint);
    }

    // ---------------------------------------------------------------- capture

    private void CaptureAndSend(Camera cam, RenderTexture rt, Texture2D tex, byte sensorId,
                                IPEndPoint endpoint)
    {
        // Sample the pose in the same frame as the render, BEFORE anything else can move it.
        Vector3 pos = cam.transform.position;
        Quaternion rot = cam.transform.rotation;

        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;
        cam.targetTexture = rt;
        cam.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        tex.Apply(false);
        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;

        byte[] jpeg = tex.EncodeToJPG(jpegQuality);
        lastFrameKB = jpeg.Length / 1024f;

        // Intrinsics. Unity gives a vertical FOV; the backend wants fx/fy/cx/cy in pixels.
        // fy = (H/2) / tan(vfov/2); square pixels => fx == fy, and the horizontal FOV then
        // follows from the aspect ratio. The backend cross-checks hfov against fx and warns
        // if they disagree, which catches a mismatched physical-camera setup immediately.
        float vfovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float fy = (captureHeight * 0.5f) / Mathf.Tan(vfovRad * 0.5f);
        float fx = fy;
        float cx = captureWidth * 0.5f;
        float cy = captureHeight * 0.5f;
        float hfovDeg = 2f * Mathf.Atan((captureWidth * 0.5f) / fx) * Mathf.Rad2Deg;

        // LRF: slant range along the camera boresight (+Z in Unity's camera frame).
        float lrf = 0f; bool lrfValid = false;
        if (Physics.Raycast(pos, cam.transform.forward, out RaycastHit hit, maxRangeMetres, terrainMask))
        {
            lrf = hit.distance; lrfValid = true;
        }

        // AGL: straight down, independent of where the gimbal is looking.
        float agl = 0f; bool aglValid = false;
        if (Physics.Raycast(pos, Vector3.down, out RaycastHit down, maxRangeMetres, terrainMask))
        {
            agl = down.distance; aglValid = true;
        }

        byte flags = FlagTelemPresent;
        if (lrfValid) flags |= FlagLrfValid;
        if (aglValid) flags |= FlagAglValid;

        // Zoom is reported for telemetry only; the backend uses fx/fy, never this.
        float zoom = 1f;

        byte[] telem = PackTelemetry(pos, rot, fx, fy, cx, cy, lrf, agl, hfovDeg, zoom);
        ulong tCaptureUs = (ulong)((Time.timeAsDouble - _startTime) * 1e6);

        _frameId[sensorId]++;
        SendFragmented(jpeg, telem, flags, sensorId, _frameId[sensorId], tCaptureUs, endpoint);

        framesSent++;
        if (logStats && framesSent % 100 == 0)
            Debug.Log($"[UavStreamer {uavId}] {framesSent} frames, {packetsSent} packets, " +
                      $"last {lastFrameKB:F1} KB, lrf={(lrfValid ? lrf.ToString("F1") : "--")} m");
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
            Debug.LogError($"[UavStreamer {uavId}] mtu {mtu} too small for {HeaderSize + TelemSize} B of framing");
            return;
        }
        int total = jpeg.Length;
        int n = Mathf.Max(1, (total + chunk - 1) / chunk);
        if (n > ushort.MaxValue)
        {
            Debug.LogError($"[UavStreamer {uavId}] frame needs {n} fragments; lower the resolution or quality");
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

            try { _udp.Send(_packet, o + len, endpoint); packetsSent++; }
            catch (SocketException e)
            {
                Debug.LogWarning($"[UavStreamer {uavId}] send failed: {e.Message}");
                return;
            }
        }
    }

    // ---------------------------------------------------------------- little-endian writers
    // BitConverter is little-endian on every platform Unity ships for, but writing the bytes
    // explicitly keeps the wire format correct even on a big-endian target.

    private static int Put(byte[] b, int o, uint v)
    {
        b[o] = (byte)v; b[o + 1] = (byte)(v >> 8); b[o + 2] = (byte)(v >> 16); b[o + 3] = (byte)(v >> 24);
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
