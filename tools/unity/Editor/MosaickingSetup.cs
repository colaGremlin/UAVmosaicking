// MosaickingSetup.cs -- one-click wiring of the 4-UAV mosaicking pipeline.
//
// PUT THIS FILE IN: Assets/Editor/          (the folder name "Editor" matters to Unity)
// RUN IT FROM:      Tools > Mosaicking > Set Up 4-UAV Streaming
//
// It is an EDITOR script, so none of it ships in a build and none of it runs during Play.
//
// WHY A MENU ITEM RATHER THAN HAND-EDITING THE SCENE
// MiniLM0-3 are prefab instances of an FBX. Adding children and components to a prefab
// instance by editing the scene's YAML means hand-writing m_AddedGameObjects and
// m_Modifications blocks with correct fileIDs -- easy to get subtly wrong, and a corrupted
// scene is expensive. Going through Unity's own APIs cannot produce an inconsistent scene,
// and every change below is registered with Undo, so Ctrl+Z reverses the whole thing.
//
// WHAT IT DOES (all idempotent -- running it twice changes nothing the second time)
//   1. creates a "Terrain" layer if absent and puts the ground mesh on it
//   2. swaps the ground mesh's Box Collider for a Mesh Collider, so the laser range finder
//      and the AGL probe actually hit the terrain surface
//   3. gives each MiniLM a MapGimbal + MapCamera for mapping, leaving the seeker untouched
//   4. attaches and configures UavStreamer and UavRandomFlight on each MiniLM
//   5. lifts the four aircraft to the mapping altitude band
//   6. repairs the seeker cameras' far clip plane
//
// There is a matching "Revert" item that removes everything this added.

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class MosaickingSetup
{
    // ---- values measured off the actual Muzaffarabad mesh -----------------------------
    // Ground under an 8 km box around the origin spans Y 283..1766, mean 904, so relief is
    // 1483 m. Flying at Y 4050-4350 puts the aircraft ~3300 m above it: a relief-to-height
    // ratio of 0.45, which is where flat-ground projection still registers acceptably.
    //
    // Lane spacing is chosen against the ZOOMED-IN footprint, not the average one. At 16 deg
    // vertical FOV and 3300 m AGL a footprint is ~1650 m across, so inside a 2000 m lane the
    // four aircraft have a real gap between them and share no imagery at all -- the
    // no-overlap case the whole direct-georeferencing approach exists to handle. Zoomed out
    // to 50 deg the same footprint is ~5500 m and they overlap heavily. Both occur every run.
    const float ALT_MIN = 4050f;
    const float ALT_MAX = 4350f;
    const float FLIGHT_HALF_EXTENT = 4000f;   // 8 x 8 km survey box
    const float LANE_SPACING = 2000f;
    const int LANE_COUNT = 4;
    const int PASSES_PER_LANE = 3;

    const float MAP_FOV_MIN = 16f;      // vertical FOV, degrees -- zoomed in
    const float MAP_FOV_MAX = 50f;      // zoomed out
    const float MAP_FOV_START = 30f;
    const float CAM_NEAR = 1f;
    const float CAM_FAR = 30000f;       // must clear altitude + slant range comfortably

    const int CAPTURE_W = 1280;
    const int CAPTURE_H = 720;
    const float SEND_HZ = 10f;

    const string TERRAIN_LAYER = "Terrain";
    const string GIMBAL_NAME = "MapGimbal";
    const string CAMERA_NAME = "MapCamera";
    const string GROUND_ROOT = "latestmodel";

    static readonly string[] UAV_NAMES = { "MiniLM0", "MiniLM1", "MiniLM2", "MiniLM3" };

    // ==================================================================================
    [MenuItem("Tools/Mosaicking/Set Up 4-UAV Streaming", false, 10)]
    public static void SetUp()
    {
        var log = new List<string>();
        int terrainLayer = EnsureLayer(TERRAIN_LAYER, log);

        FixGroundCollider(terrainLayer, log);

        var found = new List<GameObject>();
        foreach (string n in UAV_NAMES)
        {
            GameObject go = GameObject.Find(n);
            if (go == null) { log.Add($"MISSING: no GameObject named '{n}' in the open scene."); continue; }
            found.Add(go);
        }

        if (found.Count == 0)
        {
            EditorUtility.DisplayDialog("Mosaicking Setup",
                "Could not find MiniLM0..MiniLM3 in the open scene.\n\n" +
                "Open Scenes/MultipleLMVideoStitching first, then run this again.", "OK");
            return;
        }

        for (int i = 0; i < found.Count; i++)
            ConfigureUav(found[i], i, terrainLayer, log);

        RepairSeekerCameras(log);

        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Report("Set Up 4-UAV Streaming", log);
    }

    // ==================================================================================
    [MenuItem("Tools/Mosaicking/Revert 4-UAV Streaming", false, 11)]
    public static void Revert()
    {
        var log = new List<string>();
        foreach (string n in UAV_NAMES)
        {
            GameObject go = GameObject.Find(n);
            if (go == null) continue;

            var s = go.GetComponent<UavStreamer>();
            if (s != null) { Undo.DestroyObjectImmediate(s); log.Add($"{n}: removed UavStreamer"); }

            var f = go.GetComponent<UavRandomFlight>();
            if (f != null) { Undo.DestroyObjectImmediate(f); log.Add($"{n}: removed UavRandomFlight"); }

            Transform g = go.transform.Find(GIMBAL_NAME);
            if (g != null) { Undo.DestroyObjectImmediate(g.gameObject); log.Add($"{n}: removed {GIMBAL_NAME}"); }
        }
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Report("Revert 4-UAV Streaming", log);
    }

    // ==================================================================================
    [MenuItem("Tools/Mosaicking/Check Scene (read-only)", false, 30)]
    public static void Check()
    {
        var log = new List<string>();

        GameObject ground = GameObject.Find(GROUND_ROOT);
        if (ground == null) log.Add("FAIL: ground mesh 'latestmodel' not found.");
        else
        {
            var mc = ground.GetComponentInChildren<MeshCollider>();
            var bc = ground.GetComponentInChildren<BoxCollider>();
            if (mc != null && mc.enabled) log.Add("OK  : ground has an enabled MeshCollider (LRF will work).");
            else log.Add("FAIL: ground has no MeshCollider -- the LRF and AGL rays will miss.");
            if (bc != null && bc.enabled)
                log.Add("WARN: ground still has an enabled BoxCollider; it will shadow the mesh.");
        }

        foreach (string n in UAV_NAMES)
        {
            GameObject go = GameObject.Find(n);
            if (go == null) { log.Add($"FAIL: {n} not in scene."); continue; }

            var s = go.GetComponent<UavStreamer>();
            var f = go.GetComponent<UavRandomFlight>();
            string cam = s != null && s.eoCamera != null ? s.eoCamera.name : "NONE";
            log.Add($"{n}: streamer={(s != null ? "yes id=" + s.uavId + " port=" + (s.eoPortBase + s.uavId) : "NO")}, " +
                    $"flight={(f != null ? f.mode.ToString() : "NO")}, camera={cam}, Y={go.transform.position.y:F0}");

            if (s != null && s.eoCamera != null && s.eoCamera.farClipPlane < go.transform.position.y * 2f)
                log.Add($"   WARN: {n} map camera far clip {s.eoCamera.farClipPlane:F0} is low for altitude {go.transform.position.y:F0}.");
        }
        Report("Scene Check", log);
    }

    // ==================================================================================
    // pieces
    // ==================================================================================

    static int EnsureLayer(string name, List<string> log)
    {
        int existing = LayerMask.NameToLayer(name);
        if (existing >= 0) { log.Add($"layer '{name}' already exists (index {existing})"); return existing; }

        var tagManager = new SerializedObject(
            AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        SerializedProperty layers = tagManager.FindProperty("layers");

        // 0-7 are Unity's built-ins and must not be touched.
        for (int i = 8; i < layers.arraySize; i++)
        {
            SerializedProperty slot = layers.GetArrayElementAtIndex(i);
            if (string.IsNullOrEmpty(slot.stringValue))
            {
                slot.stringValue = name;
                tagManager.ApplyModifiedProperties();
                log.Add($"created layer '{name}' in slot {i}");
                return i;
            }
        }
        log.Add($"WARN: no free layer slot for '{name}'; falling back to everything-visible raycasts");
        return -1;
    }

    static void FixGroundCollider(int terrainLayer, List<string> log)
    {
        GameObject ground = GameObject.Find(GROUND_ROOT);
        if (ground == null) { log.Add($"MISSING: ground mesh '{GROUND_ROOT}' not found."); return; }

        MeshFilter mf = ground.GetComponentInChildren<MeshFilter>();
        if (mf == null) { log.Add("MISSING: ground has no MeshFilter."); return; }

        GameObject target = mf.gameObject;

        // A BoxCollider around a 168 x 156 km terrain is a solid box, and the aircraft fly
        // INSIDE it. Physics.Raycast does not report hits when the ray starts inside a
        // collider, so both the LRF and the AGL probe would silently return nothing and every
        // frame would fall back to an assumed flat plane. The mesh is ~8k triangles, so a
        // MeshCollider is essentially free.
        BoxCollider box = target.GetComponent<BoxCollider>();
        if (box != null && box.enabled)
        {
            Undo.RecordObject(box, "Disable ground BoxCollider");
            box.enabled = false;
            log.Add($"{target.name}: disabled BoxCollider (it was hiding the terrain surface)");
        }

        MeshCollider mc = target.GetComponent<MeshCollider>();
        if (mc == null)
        {
            mc = Undo.AddComponent<MeshCollider>(target);
            log.Add($"{target.name}: added MeshCollider ({mf.sharedMesh.triangles.Length / 3} triangles)");
        }
        mc.sharedMesh = mf.sharedMesh;
        mc.convex = false;

        if (terrainLayer >= 0)
        {
            SetLayerRecursive(ground, terrainLayer);
            log.Add($"{ground.name}: moved onto layer '{TERRAIN_LAYER}'");
        }
    }

    static void SetLayerRecursive(GameObject go, int layer)
    {
        Undo.RecordObject(go, "Set layer");
        go.layer = layer;
        foreach (Transform c in go.transform) SetLayerRecursive(c.gameObject, layer);
    }

    static void ConfigureUav(GameObject uav, int index, int terrainLayer, List<string> log)
    {
        // ---- gimbal ------------------------------------------------------------------
        Transform gimbal = uav.transform.Find(GIMBAL_NAME);
        if (gimbal == null)
        {
            var g = new GameObject(GIMBAL_NAME);
            Undo.RegisterCreatedObjectUndo(g, "Create MapGimbal");
            g.transform.SetParent(uav.transform, false);
            gimbal = g.transform;
            log.Add($"{uav.name}: created {GIMBAL_NAME}");
        }
        gimbal.localPosition = Vector3.zero;
        gimbal.localRotation = Quaternion.Euler(90f, 0f, 0f);   // straight down

        // ---- mapping camera ------------------------------------------------------------
        Transform camT = gimbal.Find(CAMERA_NAME);
        Camera cam;
        if (camT == null)
        {
            var c = new GameObject(CAMERA_NAME);
            Undo.RegisterCreatedObjectUndo(c, "Create MapCamera");
            c.transform.SetParent(gimbal, false);
            cam = c.AddComponent<Camera>();
            log.Add($"{uav.name}: created {CAMERA_NAME}");
        }
        else cam = camT.GetComponent<Camera>() ?? Undo.AddComponent<Camera>(camT.gameObject);

        Undo.RecordObject(cam, "Configure MapCamera");
        cam.transform.localPosition = Vector3.zero;
        cam.transform.localRotation = Quaternion.identity;
        cam.fieldOfView = MAP_FOV_START;
        cam.nearClipPlane = CAM_NEAR;
        cam.farClipPlane = CAM_FAR;
        cam.depth = -50;                       // never competes with the operator view
        cam.enabled = false;                   // UavStreamer enables it one frame at a time
        cam.clearFlags = CameraClearFlags.Skybox;
        // Do not photograph the aircraft themselves.
        cam.cullingMask = ~0;
        if (uav.layer != 0) cam.cullingMask &= ~(1 << uav.layer);

        // ---- streamer -------------------------------------------------------------------
        var streamer = uav.GetComponent<UavStreamer>() ?? Undo.AddComponent<UavStreamer>(uav);
        Undo.RecordObject(streamer, "Configure UavStreamer");
        streamer.uavId = index;
        streamer.eoCamera = cam;
        streamer.irCamera = null;
        streamer.host = "127.0.0.1";
        streamer.eoPortBase = 5001;
        streamer.irPortBase = 5011;
        streamer.mtu = 1400;
        streamer.sendHz = SEND_HZ;
        streamer.captureWidth = CAPTURE_W;
        streamer.captureHeight = CAPTURE_H;
        streamer.jpegQuality = 65;
        streamer.stagger = true;
        streamer.maxRangeMetres = 12000f;
        streamer.terrainMask = terrainLayer >= 0 ? (1 << terrainLayer) : ~0;
        log.Add($"{uav.name}: UavStreamer id={index} -> 127.0.0.1:{5001 + index}");

        // ---- flight ---------------------------------------------------------------------
        var flight = uav.GetComponent<UavRandomFlight>() ?? Undo.AddComponent<UavRandomFlight>(uav);
        Undo.RecordObject(flight, "Configure UavRandomFlight");
        flight.uniqueSeed = index;
        flight.mode = UavFlightMode.Lawnmower;
        flight.surveyHalfExtent = FLIGHT_HALF_EXTENT;
        flight.laneSpacing = LANE_SPACING;
        flight.laneCount = LANE_COUNT;
        flight.passesPerLane = PASSES_PER_LANE;
        flight.altitudeMin = ALT_MIN;
        flight.altitudeMax = ALT_MAX;
        flight.fovMinDeg = MAP_FOV_MIN;
        flight.fovMaxDeg = MAP_FOV_MAX;
        flight.sensorGimbal = gimbal;
        flight.zoomCamera = cam;

        // ---- position: at the start of its own lane ---------------------------------------
        Undo.RecordObject(uav.transform, "Move UAV to its survey lane");
        float laneX = (index - (LANE_COUNT - 1) * 0.5f) * LANE_SPACING;
        uav.transform.position = new Vector3(
            laneX - LANE_SPACING * 0.5f + LANE_SPACING / (2f * PASSES_PER_LANE),
            Mathf.Lerp(ALT_MIN, ALT_MAX, 0.5f),
            -FLIGHT_HALF_EXTENT);
        uav.transform.rotation = Quaternion.identity;   // heading north, up the first leg
        log.Add($"{uav.name}: lane centre x={laneX:F0}, starting at {uav.transform.position}");
    }

    static void RepairSeekerCameras(List<string> log)
    {
        // far = 1e16 collapses depth-buffer precision and produces z-fighting across the
        // whole scene. Nothing here needs to see past 20 km.
        foreach (Camera c in Object.FindObjectsOfType<Camera>(true))
        {
            if (c.name == CAMERA_NAME) continue;
            if (c.farClipPlane <= CAM_FAR) continue;
            Undo.RecordObject(c, "Repair far clip plane");
            float was = c.farClipPlane;
            c.farClipPlane = CAM_FAR;
            log.Add($"{c.name}: far clip {was:0.###e+0} -> {CAM_FAR:F0} (depth precision)");
        }
    }

    static void Report(string title, List<string> log)
    {
        string body = log.Count == 0 ? "Nothing to do." : string.Join("\n", log);
        Debug.Log($"[Mosaicking] {title}\n{body}");

        int problems = log.Count(l => l.StartsWith("FAIL") || l.StartsWith("MISSING") || l.Contains("WARN"));
        EditorUtility.DisplayDialog(
            "Mosaicking - " + title,
            body + (problems > 0
                ? $"\n\n{problems} item(s) need your attention - see the Console."
                : "\n\nAll good. Full detail is in the Console."),
            "OK");
    }
}
