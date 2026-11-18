# ForestAirSim — forest-surveillance drone (UE 4.27 + AirSim + ROS 2 + YOLO)

Windows runs Unreal Engine 4.27 + AirSim + the forest. Linux runs ROS 2,
keyboard teleop, obstacle avoidance, YOLO, 3D localisation and SQLite storage.
The two machines talk over the network via AirSim's RPC server (41451).

## Current status (2026-11-10)

The Linux ROS 2 stack, conversions, safety, avoidance planner, detection
pipeline, storage and deterministic forest generator are implemented and
tested offline. The Unreal environment, real assets and end-to-end flight are
**not** validated: this Linux machine has no UE/AirSim and the asset packs must
be acquired manually. See `TEST-REPORT.md` for exact results and
`ASSET-REPORT.md` for what was verified.

| Area | State |
|---|---|
| NED↔ENU, FRD↔FLU, quaternion/IMU/camera math | Implemented, unit-tested |
| 8 ROS 2 nodes + typed messages + launch | Implemented, builds on ROS 2 Jazzy |
| Keyboard teleop, flight states, safety, teleop timeout | Implemented, logic tested |
| Depth-based obstacle avoidance with hysteresis | Implemented, logic tested |
| RGB + depth + thermal (AirSim Infrared proxy) | Configured; live streams untested |
| YOLO RGB/thermal + association + 3D localisation + SQLite/CSV | Implemented; synthetic pipeline tested |
| Dense forest generator (1400 trees / 7100 foliage) | Implemented, deterministic, tested |
| UE editor importer / wildlife motion / thermal IDs | Windows scripts provided; needs UE/assets |
| Real forest with actual trees, humans, animals | **Not complete — external assets required** |
| End-to-end simulation flight | **Not tested** |

## Layout

```
README.md  ARCHITECTURE.md  GAP-ANALYSIS.md  ASSET-REPORT.md  TEST-REPORT.md
WINDOWS-SETUP.md  LINUX-ROS2-SETUP.md  CAMERA-CONFIGURATION.md
YOLO-SETUP.md  OBSTACLE-AVOIDANCE.md  COORDINATE-TRANSFORMS.md
requirements.txt

ForestAirSim/
  ForestAirSim.uproject  Config/{DefaultGame.ini,asset_map.json}
  Docs/{settings.json,placement_out.csv,vegetation_out.csv,placement_summary.json,
        BUILD-REPORT.md,asset-ledger.md,transform.md,tests.md,windows-ubuntu-steps.md}
  Scripts/{ue_place_actors.py,configure_segmentation_ids.py,
           wildlife_motion.py,ue_thermal_setup.py}
  Shaders/ThermalPostProcess.usf

Unreal/ForestEnvironment/Assets/{Trees,Vegetation,Humans,Animals,Terrain,
                                  Obstacles,Materials,Animations}/

scripts/generate_placement.py       # build-time generator, no sim/ROS
ros2_bridge/forest_interfaces/      # typed detection messages
ros2_bridge/forest_bringup/         # 8 nodes, config, launch, tests
tests/                              # repo + ROS integration tests
archive/forest_flight_test.py       # Windows-only raw AirSim smoke test
```

## Quick start (Linux, offline)

```bash
python3 scripts/generate_placement.py --seed 7
python3 -m pytest tests/ ros2_bridge/forest_bringup/test/ -q
python3 -m json.tool ForestAirSim/Docs/settings.json > /dev/null
python3 -m pytest tests/test_forest.py::test_ros_only_gate -q
# or, equivalently, by grep (word boundary keeps airsim_interfaces allowed):
grep -rnE "^[[:space:]]*(import[[:space:]]+airsim\b|from[[:space:]]+airsim\b)" \
  ros2_bridge/ tests/ scripts/ && echo "GATE FAIL" || echo "ROS-only gate OK"
```

Runs without ROS or a simulator: 67 passed, 3 skipped (3 are ROS integration tests, skipped when no sourced workspace). With a sourced ROS 2 workspace: `python3 -m pytest tests/test_ros_integration.py -m integration -q` → 3 passed.

## Build and run (ROS 2)

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select forest_interfaces forest_bringup
source install/setup.bash
ros2 launch forest_bringup forest.launch.py airsim_host:=192.168.1.100
```

Requires the AirSim ROS 2 wrapper (`airsim_ros_pkgs` + `airsim_interfaces`).
See `LINUX-ROS2-SETUP.md` and `WINDOWS-SETUP.md`.

## Controls

↑/↓/←/→ forward/back/strafe · W/S up/down · A/D yaw · Space hover ·
T takeoff · L land · ESC emergency stop (latched).

## Test commands

```bash
# 1. offline unit tests (no ROS)
python3 -m pytest tests/ ros2_bridge/forest_bringup/test/ -q

# 2. ROS integration (sourced workspace; in-process nodes)
python3 -m pytest tests/test_ros_integration.py -m integration -q

# 3. Windows AirSim smoke test (on Windows, sim running)
python archive\forest_flight_test.py

# 4. complete sim: see TEST-REPORT.md
```

## Important limits

- Thermal uses AirSim `ImageType 7 (Infrared)` with stencil IDs — a documented
  object-ID proxy, **not** radiometry. It is never RGB duplication.
- COCO YOLO weights do not contain deer or wild boar; custom weights are
  required. No detection accuracy is claimed.
- 3D positions come from depth (primary) or a documented ground-plane fallback;
  never from a bounding box alone.
- The forest generator references real assets through `asset_map.json` and
  skips missing ones. It never spawns placeholder cubes.
- No free GitHub repo found that supplies a UE 4.27-ready forest; see
  `ASSET-REPORT.md`.
