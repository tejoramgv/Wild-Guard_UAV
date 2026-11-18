"""Repo-level tests: placement, vegetation, settings, generator and ROS-only gate.

Conversions and the ROS nodes have their own suites under
ros2_bridge/forest_bringup/test/ and tests/.
"""

import csv
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_placement", ROOT / "scripts" / "generate_placement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_placement_out_matches_summary():
    rows = _read_csv(ROOT / "ForestAirSim" / "Docs" / "placement_out.csv")
    veg = _read_csv(ROOT / "ForestAirSim" / "Docs" / "vegetation_out.csv")
    summary = json.loads(
        (ROOT / "ForestAirSim" / "Docs" / "placement_summary.json").read_text())
    assert summary["profile"] == "dense"
    assert summary["tree_count"] >= 500, summary["tree_count"]
    assert len(rows) == summary["total_actors"]
    assert len(veg) == summary["total_vegetation"]
    assert summary["min_spacing_ok"] is True
    assert summary["elevation_min"] < summary["elevation_max"]
    vegetation_classes = {r["class"] for r in veg}
    for cls, info in summary["by_class"].items():
        source = veg if cls in vegetation_classes else rows
        actual = sum(1 for r in source if r["class"] == cls)
        assert actual == info["count"], (cls, actual, info["count"])
        if info["required"] > 0:
            assert info["min_dist"] >= info["required"] - 1e-6, cls


def test_vegetation_out_matches_summary():
    rows = _read_csv(ROOT / "ForestAirSim" / "Docs" / "vegetation_out.csv")
    summary = json.loads(
        (ROOT / "ForestAirSim" / "Docs" / "placement_summary.json").read_text())
    assert len(rows) == summary["total_vegetation"]
    assert summary["total_vegetation"] >= 3000
    classes = {r["class"] for r in rows}
    assert {"grass", "bush", "fern"} <= classes


def test_settings():
    s = json.loads((ROOT / "ForestAirSim" / "Docs" / "settings.json").read_text())
    cameras = s["Vehicles"]["ForestDrone"]["Cameras"]
    cam = cameras["front-center"]
    cs = cam["CaptureSettings"][0]
    assert (cs["Width"], cs["Height"], cs["FOV_Degrees"]) == (640, 480, 60)
    assert cam["Pitch"] == -11.5
    # Thermal camera must be a separate camera requesting AirSim Infrared (7),
    # not a duplicate Scene/RGB capture.
    thermal = cameras["thermal"]
    assert thermal["CaptureSettings"][0]["ImageType"] == 7
    assert thermal["Pitch"] == cam["Pitch"]


def test_generator_is_deterministic():
    gen = _load_generator()
    rows_a, veg_a, summary_a = gen.generate(gen.DENSE, 7)
    rows_b, veg_b, summary_b = gen.generate(gen.DENSE, 7)
    assert rows_a == rows_b and veg_a == veg_b and summary_a == summary_b
    assert len(rows_a) >= 1000
    assert summary_a["min_spacing_ok"] is True


def test_legacy_profile_preserves_original_counts():
    gen = _load_generator()
    rows, _veg, summary = gen.generate(gen.LEGACY, 7)
    assert len(rows) == 126
    expected = {"oak": 22, "pine": 30, "rock": 12, "human": 25, "deer": 8,
                "boar": 4, "elephant": 3, "monkey": 8, "bird": 8, "fox": 3,
                "rabbit": 3}
    for cls, count in expected.items():
        assert sum(1 for r in rows if r["class"] == cls) == count, cls
    assert summary["min_spacing_ok"] is True


def test_cross_category_spacing_and_clearances():
    """Dense placement must avoid cross-category interpenetration.

    Regression: earlier layouts could satisfy per-species summaries while placing
    cross-species tree pairs far closer than their min distances or spawning
    ground creatures inside trunk footprints.
    """
    rows = _read_csv(ROOT / "ForestAirSim" / "Docs" / "placement_out.csv")
    veg = _read_csv(ROOT / "ForestAirSim" / "Docs" / "vegetation_out.csv")
    summary = json.loads(
        (ROOT / "ForestAirSim" / "Docs" / "placement_summary.json").read_text())

    tree_classes = {"oak", "pine", "birch", "tropical"}
    trees = [r for r in rows if r["class"] in tree_classes]
    ground = [r for r in rows if r["class"] in
              {"human", "deer", "boar", "elephant", "monkey", "fox", "rabbit"}]
    shrubs = [r for r in veg if r["class"] in {"bush", "fern"}]

    req = {c: summary["by_class"][c]["required"] for c in tree_classes}
    for c in tree_classes:
        assert summary["by_class"][c]["ok"] is True

    # Cross-species tree spacing floor (generator enforces 0.85 * min_dist).
    for i, a in enumerate(trees):
        ax, ay = float(a["x"]), float(a["y"])
        ra = req[a["class"]] * 0.85
        for b in trees[i + 1:]:
            if a["class"] == b["class"]:
                continue
            d = math.hypot(ax - float(b["x"]), ay - float(b["y"]))
            need = max(ra, req[b["class"]] * 0.85) - 1e-9
            assert d >= need, (a["id"], b["id"], d, need)

    # Ground creatures must clear physical trunk footprints.
    trunk = {"oak": 0.5, "pine": 0.35, "birch": 0.3, "tropical": 0.4}
    for c in ground:
        cx, cy = float(c["x"]), float(c["y"])
        for t in trees:
            d = math.hypot(cx - float(t["x"]), cy - float(t["y"]))
            assert d >= trunk[t["class"]] - 1e-9, (c["id"], t["id"], d)

    # Bushes/ferns must clear trunk footprints.
    for s in shrubs:
        sx, sy = float(s["x"]), float(s["y"])
        for t in trees:
            d = math.hypot(sx - float(t["x"]), sy - float(t["y"]))
            assert d >= 0.7 - 1e-9, (s["id"], t["id"], d)

    # Trees must remain outside open bowls and flight corridors.
    for t in trees:
        x, y = float(t["x"]), float(t["y"])
        for (cx, cy), rad in summary["open_bowls"]:
            assert math.hypot(x - cx, y - cy) >= rad - 1e-9, (t["id"], "bowl")
        for (x1, y1), (x2, y2), w in summary["corridors"]:
            dx, dy = x2 - x1, y2 - y1
            l2 = dx * dx + dy * dy
            tt = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / l2)) if l2 else 0.0
            dist = math.hypot(x - (x1 + tt * dx), y - (y1 + tt * dy))
            assert dist >= w / 2.0 - 1e-9, (t["id"], "corridor", dist)


def test_ros_only_gate():
    """No raw AirSim API imports in runtime paths.

    scripts/ is build-time code and must also stay AirSim-free; the only
    exempt location is ForestAirSim/Scripts (Windows-only editor/API tooling)
    plus the quarantined archive/.
    """
    import ast
    runtime_py = (list((ROOT / "ros2_bridge").rglob("*.py"))
                  + list((ROOT / "scripts").rglob("*.py"))
                  + [Path(__file__)])

    def uses_airsim(path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "airsim" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "airsim":
                    return True
        return False

    assert [str(p) for p in runtime_py if uses_airsim(p)] == []
    assert (ROOT / "archive" / "forest_flight_test.py").exists()
