"""Deterministic procedural forest layout generator.

Build-time only: this script never talks to Unreal, AirSim or ROS. It writes:

  ForestAirSim/Docs/placement_out.csv       actor rows (ground-referenced)
  ForestAirSim/Docs/placement_summary.json  counts + spacing verification
  ForestAirSim/Docs/vegetation_out.csv      high-count foliage (instanced)
  ForestAirSim/Scripts/ue_place_actors.py   Unreal Editor Python spawner

Profiles:
  dense   (default) realistic forest: ~1400 trees + ~7000 foliage items
  legacy            original 126-actor budget

Examples:
  python3 scripts/generate_placement.py --seed 7
  python3 scripts/generate_placement.py --profile legacy --seed 7
  python3 scripts/generate_placement.py --config my_forest.json --seed 42
"""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "ForestAirSim" / "Docs" / "placement_out.csv"
OUT_VEG_CSV = ROOT / "ForestAirSim" / "Docs" / "vegetation_out.csv"
OUT_JSON = ROOT / "ForestAirSim" / "Docs" / "placement_summary.json"
UE_PY = ROOT / "ForestAirSim" / "Scripts" / "ue_place_actors.py"

FIELDS = ["id", "class", "species", "x", "y", "z", "yaw", "scale",
          "height_m", "zone", "asset_key", "behavior", "speed", "path_id",
          "group_id"]
VEG_FIELDS = ["id", "class", "x", "y", "z", "yaw", "scale", "zone", "asset_key"]


@dataclass
class Species:
    name: str
    count: int
    min_dist: float
    height_min: float
    height_max: float
    zone: str = "mixed"
    ref_height: float = 10.0
    x_range: tuple = (-1.0, 1.0)   # normalized -1..1 in world half-extents
    y_range: tuple = (-1.0, 1.0)
    asset_key: str = ""

    def key(self):
        return self.asset_key or self.name


@dataclass
class Profile:
    name: str
    world: tuple
    trees: list
    vegetation: list
    obstacles: list
    creatures: list
    open_bowls: list
    corridors: list
    density: float = 1.0


def _tree(name, count, min_d, hmin, hmax, **kw):
    return Species(name, count, min_d, hmin, hmax, **kw)


DENSE = Profile(
    name="dense",
    world=(180.0, 180.0),
    trees=[
        _tree("oak", 420, 3.2, 12.0, 20.0, zone="mixed", x_range=(-1, 1), y_range=(-1, 1)),
        _tree("pine", 520, 2.8, 14.0, 24.0, zone="north", x_range=(-1, 1), y_range=(-1, 0.25)),
        _tree("birch", 300, 2.6, 10.0, 16.0, zone="south-east", x_range=(-0.4, 1), y_range=(-1, 0.2)),
        _tree("tropical", 160, 3.0, 12.0, 22.0, zone="west", x_range=(-1, -0.35), y_range=(-0.7, 0.7)),
    ],
    vegetation=[
        Species("grass", 5000, 0.0, 0.2, 0.5, asset_key="grass"),
        Species("bush", 1200, 1.5, 0.5, 1.2, asset_key="bush"),
        Species("fern", 900, 0.8, 0.3, 0.6, asset_key="fern"),
    ],
    obstacles=[
        Species("rock", 90, 4.0, 0.4, 1.5, ref_height=1.0),
        Species("fallen_log", 45, 5.0, 1.0, 3.0, ref_height=1.0),
    ],
    creatures=[
        Species("human", 40, 0.6, 1.6, 1.85, ref_height=1.75),
        Species("deer", 25, 3.0, 1.2, 1.5, ref_height=1.35),
        Species("boar", 15, 3.0, 0.7, 0.9, ref_height=0.8),
        Species("elephant", 5, 6.0, 2.8, 3.2, ref_height=3.0),
        Species("monkey", 30, 2.0, 0.4, 0.7, ref_height=0.55),
        Species("bird", 25, 4.0, 0.2, 0.4, ref_height=0.3),
        Species("fox", 10, 4.0, 0.4, 0.55, ref_height=0.5),
        Species("rabbit", 15, 4.0, 0.25, 0.35, ref_height=0.3),
    ],
    open_bowls=[((0, 0), 10.0), ((55, -45), 9.0), ((-60, 50), 8.0)],
    corridors=[((0, -90), (0, 90), 5.0), ((-90, 10), (90, 10), 4.0)],
)

LEGACY = Profile(
    name="legacy",
    world=(110.0, 90.0),
    trees=[_tree("oak", 22, 4.0, 8.0, 12.0),
           _tree("pine", 30, 3.0, 10.0, 15.0)],
    vegetation=[Species("grass", 800, 0.0, 0.2, 0.5, asset_key="grass"),
                Species("bush", 120, 1.5, 0.5, 1.2, asset_key="bush")],
    obstacles=[Species("rock", 12, 5.0, 0.4, 1.5, ref_height=1.0)],
    creatures=[Species("human", 25, 0.6, 1.6, 1.85, ref_height=1.75),
               Species("deer", 8, 3.0, 1.2, 1.5, ref_height=1.35),
               Species("boar", 4, 3.0, 0.7, 0.9, ref_height=0.8),
               Species("elephant", 3, 6.0, 2.8, 3.2, ref_height=3.0),
               Species("monkey", 8, 2.0, 0.4, 0.7, ref_height=0.55),
               Species("bird", 8, 4.0, 0.2, 0.4, ref_height=0.3),
               Species("fox", 3, 5.0, 0.4, 0.55, ref_height=0.5),
               Species("rabbit", 3, 4.0, 0.25, 0.35, ref_height=0.3)],
    open_bowls=[((0, 0), 8.0), ((30, -20), 8.0)],
    corridors=[],
)

PROFILES = {"dense": DENSE, "legacy": LEGACY}


def terrain_height(x, y):
    """Smooth deterministic height field (metres).

    The Unreal spawner line-traces the real landscape instead; this value is a
    build-time reference used for elevation statistics and CSV fallback.
    """
    return (2.2 * math.sin(x / 23.0) + 1.6 * math.cos(y / 19.0)
            + 0.9 * math.sin((x + y) / 11.0) + 0.35 * math.cos((x - y) / 7.0))


def _hash_unit(text):
    import hashlib
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(0xFFFFFFFFFFFF)


class OccupancyGrid:
    """Spatial hash for variable-radius clearance checks."""

    def __init__(self, cell_size=2.0):
        self.cell = max(float(cell_size), 0.5)
        self.items = {}  # (cx, cy) -> [(x, y, r)]
        self.max_r = 0.0

    def clear(self, x, y, r):
        span = int(math.ceil((r + self.max_r) / self.cell)) + 1
        cx = int(math.floor(x / self.cell))
        cy = int(math.floor(y / self.cell))
        for ix in range(cx - span, cx + span + 1):
            for iy in range(cy - span, cy + span + 1):
                for px, py, pr in self.items.get((ix, iy), ()):
                    if math.hypot(x - px, y - py) < max(r, pr) - 1e-9:
                        return False
        return True

    def add(self, x, y, r):
        cx = int(math.floor(x / self.cell))
        cy = int(math.floor(y / self.cell))
        self.items.setdefault((cx, cy), []).append((x, y, r))
        if r > self.max_r:
            self.max_r = r


class Placer:
    def __init__(self, rng, world, open_bowls, corridors, density_mask=True):
        self.rng = rng
        self.w = world[0]
        self.h = world[1]
        self.open_bowls = open_bowls
        self.corridors = corridors
        self.density_mask = density_mask

    def in_open(self, x, y, r=0.0):
        for (cx, cy), radius in self.open_bowls:
            if math.hypot(x - cx, y - cy) < radius + r:
                return True
        return False

    def in_corridor(self, x, y, r=0.0):
        for (x1, y1), (x2, y2), width in self.corridors:
            if _point_segment_distance(x, y, x1, y1, x2, y2) < width / 2.0 + r:
                return True
        return False

    def is_free(self, x, y, r=0.0):
        return not self.in_open(x, y, r) and not self.in_corridor(x, y, r)

    def density(self, x, y):
        if not self.density_mask:
            return 1.0
        value = (0.5 + 0.5 * math.sin(x * 0.075) * math.cos(y * 0.065)
                 + 0.25 * math.sin((x + y) * 0.031))
        return max(0.18, min(1.0, value))

    def scatter(self, n, min_dist, bounds, avoid_structures=False,
                checks=None, registers=None):
        """Blue-noise-ish scatter with global occupancy checks.

        ``checks`` / ``registers`` are lists of (OccupancyGrid, radius). A
        candidate must be clear in every check grid and is added to every
        register grid on acceptance.
        """
        checks = checks or []
        registers = registers or []
        x0, x1, y0, y1 = bounds
        cell = max(min_dist, 0.25)
        grid = {}
        points = []
        attempts = 0
        max_attempts = max(20000, n * 400)
        while len(points) < n and attempts < max_attempts:
            attempts += 1
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            if avoid_structures and not self.is_free(x, y):
                continue
            if avoid_structures and self.rng.random() > self.density(x, y):
                continue
            if not self._spacing_ok(grid, x, y, min_dist, cell):
                continue
            if any(not occ.clear(x, y, r) for occ, r in checks):
                continue
            points.append((x, y))
            grid.setdefault((int(x / cell), int(y / cell)), []).append((x, y))
            for occ, r in registers:
                occ.add(x, y, r)
        if len(points) < n:
            raise RuntimeError(
                f"could only place {len(points)}/{n} points with min_d={min_dist} "
                f"in {bounds}; reduce density or spacing")
        return points

    @staticmethod
    def _spacing_ok(grid, x, y, min_dist, cell):
        cx, cy = int(x / cell), int(y / cell)
        for ix in range(cx - 1, cx + 2):
            for iy in range(cy - 1, cy + 2):
                for px, py in grid.get((ix, iy), ()):
                    if math.hypot(x - px, y - py) < min_dist:
                        return False
        return True

    def scatter_groups(self, n, min_dist, bounds, group_sizes, avoid_structures=False,
                       checks=None, registers=None):
        """Place small groups with global spacing enforcement."""
        checks = checks or []
        registers = registers or []
        x0, x1, y0, y1 = bounds
        cell = max(min_dist, 0.25)
        grid = {}
        members = []
        group_id = 0

        def can_place(x, y):
            if not self._spacing_ok(grid, x, y, min_dist, cell):
                return False
            for occ, r in checks:
                if not occ.clear(x, y, r):
                    return False
            return True

        def do_place(x, y):
            grid.setdefault((int(x / cell), int(y / cell)), []).append((x, y))
            for occ, r in registers:
                occ.add(x, y, r)

        attempts = 0
        max_attempts = max(20000, n * 500)
        while len(members) < n and attempts < max_attempts:
            attempts += 1
            cx, cy = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            if avoid_structures and not self.is_free(cx, cy):
                continue
            if avoid_structures and self.rng.random() > self.density(cx, cy):
                continue
            if not can_place(cx, cy):
                continue
            group_id += 1
            size = min(self.rng.choice(group_sizes), n - len(members))
            do_place(cx, cy)
            placed = [(cx, cy)]
            inner = 0
            while len(placed) < size and inner < 200:
                inner += 1
                radius = self.rng.uniform(min_dist, min_dist + 2.5)
                angle = self.rng.uniform(0, 2 * math.pi)
                x = min(x1, max(x0, cx + radius * math.cos(angle)))
                y = min(y1, max(y0, cy + radius * math.sin(angle)))
                if avoid_structures and not self.is_free(x, y):
                    continue
                if can_place(x, y):
                    do_place(x, y)
                    placed.append((x, y))
            for x, y in placed:
                members.append((x, y, group_id))
        if len(members) < n:
            raise RuntimeError(
                f"could only place {len(members)}/{n} grouped points with "
                f"min_d={min_dist} in {bounds}")
        return members[:n]


def _point_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _behavior(cls, index, total):
    if cls == "human":
        return ["static", "static", "wander", "path", "wander"][index % 5]
    if cls in ("deer", "monkey", "bird"):
        return ["static", "wander", "path", "wander"][index % 4]
    if cls in ("boar", "fox"):
        return "wander" if index % 2 else "static"
    if cls == "elephant":
        return "path" if index % 2 == 0 else "wander"
    if cls == "rabbit":
        return "static" if index % 3 else "wander"
    return "static"


def _speed(behavior, unit):
    if behavior == "static":
        return 0.0
    if behavior == "wander":
        return round(0.25 + 0.55 * unit, 3)
    return round(0.6 + 0.9 * unit, 3)


def _bounds_for(spec, world):
    hx, hy = world[0] / 2.0, world[1] / 2.0
    return (spec.x_range[0] * hx, spec.x_range[1] * hx,
            spec.y_range[0] * hy, spec.y_range[1] * hy)


def generate(profile, seed, config_overrides=None):
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
    rng = random.Random(seed)
    placer = Placer(rng, profile.world, profile.open_bowls, profile.corridors)
    rows = []
    veg_rows = []
    elevation = []
    min_spacing = {}

    def add_row(cls, x, y, spec, index, group_id=0, species=None, z=None,
                behavior="static", speed=0.0, path_id=""):
        height = spec.height_min + (spec.height_max - spec.height_min) * rng.random()
        ground = terrain_height(x, y) if z is None else z
        row = {
            "id": f"{cls}-{index:04d}",
            "class": cls,
            "species": species or cls,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(ground + (height / 2.0 if cls in ("bird",) else 0.0), 3),
            "yaw": round(rng.uniform(0, 360), 1),
            "scale": round(height / spec.ref_height, 4),
            "height_m": round(height, 3),
            "zone": spec.zone,
            "asset_key": spec.key(),
            "behavior": behavior,
            "speed": speed,
            "path_id": path_id,
            "group_id": group_id,
        }
        rows.append(row)
        elevation.append(terrain_height(x, y))

    # --- global occupancy grids (enforce cross-category spacing) ---
    # Cross-species tree spacing: prevents trunk/canopy interpenetration while
    # keeping dense profiles feasible (relaxed factor vs full min_dist).
    tree_spacing_occ = OccupancyGrid(cell_size=3.0)
    # Physical trunk/obstacle footprint for creature and shrub clearance.
    tree_trunk_occ = OccupancyGrid(cell_size=2.0)
    # Creatures: bodies must clear trees/obstacles and each other.
    creature_occ = OccupancyGrid(cell_size=2.0)

    # Conservative physical footprint radii (m).
    TREE_TRUNK = {"oak": 0.5, "pine": 0.35, "birch": 0.3, "tropical": 0.4}
    OBSTACLE_FOOTPRINT = {"rock": 0.9, "fallen_log": 0.7}

    # Trees first: enforce same-species min_dist (local grid in scatter) and a
    # global cross-species spacing floor; register physical trunks separately.
    for spec in profile.trees:
        bounds = _bounds_for(spec, profile.world)
        spacing_r = spec.min_dist * 0.85
        trunk_r = TREE_TRUNK.get(spec.name, 0.4)
        points = placer.scatter(spec.count, spec.min_dist, bounds,
                                avoid_structures=True,
                                checks=[(tree_spacing_occ, spacing_r)],
                                registers=[(tree_spacing_occ, spacing_r),
                                           (tree_trunk_occ, trunk_r)])
        min_spacing[spec.name] = _min_pair_distance(points, spec.min_dist)
        for i, (x, y) in enumerate(points, 1):
            add_row(spec.name, x, y, spec, i)

    # Rocks and fallen logs: keep own spacing and stay clear of tree trunks.
    for spec in profile.obstacles:
        bounds = _bounds_for(spec, profile.world)
        footprint = OBSTACLE_FOOTPRINT.get(spec.name, 0.8)
        points = placer.scatter(spec.count, spec.min_dist, bounds,
                                avoid_structures=True,
                                checks=[(tree_trunk_occ, footprint)],
                                registers=[(tree_trunk_occ, footprint)])
        min_spacing[spec.name] = _min_pair_distance(points, spec.min_dist)
        for i, (x, y) in enumerate(points, 1):
            add_row(spec.name, x, y, spec, i)

    # Vegetation (separate CSV; grass/bush/fern are instanced at import)
    for spec in profile.vegetation:
        bounds = _bounds_for(spec, profile.world)
        avoid = spec.name in ("bush", "fern")
        checks = []
        if spec.name in ("bush", "fern"):
            # keep shrubs clear of tree trunks/obstacles
            checks = [(tree_trunk_occ, 0.7)]
        points = placer.scatter(spec.count, spec.min_dist, bounds,
                                avoid_structures=avoid,
                                checks=checks)
        min_spacing[spec.name] = (_min_pair_distance(points, spec.min_dist)
                                  if spec.min_dist else 0.0)
        for i, (x, y) in enumerate(points, 1):
            height = spec.height_min + (spec.height_max - spec.height_min) * rng.random()
            veg_rows.append({
                "id": f"{spec.name}-{i:05d}",
                "class": spec.name,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(terrain_height(x, y), 3),
                "yaw": round(rng.uniform(0, 360), 1),
                "scale": round(height / max(spec.height_max, 0.1), 3),
                "zone": spec.zone,
                "asset_key": spec.key(),
            })

    # Creatures: groups for social species, singles for large animals.
    group_sizes = {"human": [1, 2, 2, 3, 4], "deer": [1, 2, 3],
                   "boar": [1, 2], "monkey": [2, 3, 4], "fox": [1, 2],
                   "rabbit": [1, 2], "bird": [1, 2, 3], "elephant": [1]}
    for spec in profile.creatures:
        bounds = _bounds_for(spec, profile.world)
        # Ground creatures must not spawn inside a tree/obstacle footprint and
        # should not overlap other creatures. Birds may perch/fly above, but
        # keep their base position clear of trunks.
        body_r = max(spec.min_dist / 2.0, 0.4)
        if spec.name == "bird":
            cchecks = [(tree_trunk_occ, 0.4)]
        else:
            cchecks = [(tree_trunk_occ, body_r), (creature_occ, body_r)]
        cregs = [(creature_occ, body_r)]
        if spec.name in group_sizes:
            placed = placer.scatter_groups(spec.count, spec.min_dist, bounds,
                                          group_sizes[spec.name],
                                          avoid_structures=spec.name == "elephant",
                                          checks=cchecks, registers=cregs)
            min_spacing[spec.name] = _min_pair_distance(
                [(x, y) for x, y, _ in placed], spec.min_dist)
            for i, (x, y, gid) in enumerate(placed, 1):
                behavior = _behavior(spec.name, i, len(placed))
                unit = _hash_unit(f"{seed}:{spec.name}:{i}")
                speed = _speed(behavior, unit)
                path_id = f"path_{'abc'[i % 3]}" if behavior == "path" else ""
                z = None
                if spec.name == "bird":
                    if i % 2 == 1:
                        z = terrain_height(x, y) + rng.uniform(5.0, 10.0)
                        behavior, speed, path_id = "wander_air", round(0.8 + unit, 3), ""
                    else:
                        behavior, speed, path_id = "perched", 0.0, ""
                add_row(spec.name, x, y, spec, i, group_id=gid,
                        z=z, behavior=behavior, speed=speed, path_id=path_id)
        else:
            points = placer.scatter(spec.count, spec.min_dist, bounds,
                                    avoid_structures=False,
                                    checks=cchecks, registers=cregs)
            min_spacing[spec.name] = _min_pair_distance(points, spec.min_dist)
            for i, (x, y) in enumerate(points, 1):
                behavior = _behavior(spec.name, i, len(points))
                add_row(spec.name, x, y, spec, i, behavior=behavior,
                        speed=_speed(behavior, _hash_unit(f"{seed}:{spec.name}:{i}")),
                        path_id=f"path_{'abc'[i % 3]}" if behavior == "path" else "")

    spacing_ok = all(
        value >= _required(profile, name) - 1e-6
        for name, value in min_spacing.items())
    all_specs = (profile.trees + profile.vegetation + profile.obstacles
                 + profile.creatures)
    for spec in all_specs:
        if spec.name in min_spacing and min_spacing[spec.name] < spec.min_dist - 1e-6:
            spacing_ok = False

    tree_count = sum(1 for r in rows if r["class"] in
                     [t.name for t in profile.trees])
    summary = {
        "profile": profile.name,
        "seed": seed,
        "world": list(profile.world),
        "total_actors": len(rows),
        "total_vegetation": len(veg_rows),
        "tree_count": tree_count,
        "by_class": {},
        "min_spacing_ok": bool(spacing_ok),
        "min_dist": {k: round(v, 3) for k, v in min_spacing.items()},
        "elevation_min": round(min(elevation), 3) if elevation else 0.0,
        "elevation_max": round(max(elevation), 3) if elevation else 0.0,
        "open_bowls": [[list(c), r] for c, r in profile.open_bowls],
        "corridors": [[list(a), list(b), w] for a, b, w in profile.corridors],
    }
    for spec in all_specs:
        if spec in profile.vegetation:
            count = sum(1 for r in veg_rows if r["class"] == spec.name)
        else:
            count = sum(1 for r in rows if r["class"] == spec.name)
        if count:
            summary["by_class"][spec.name] = {
                "count": count,
                "min_dist": round(min_spacing.get(spec.name, 0.0), 3),
                "required": spec.min_dist,
                "ok": min_spacing.get(spec.name, 0.0) >= spec.min_dist - 1e-6,
            }
    return rows, veg_rows, summary


def _required(profile, name):
    for spec in (profile.trees + profile.vegetation + profile.obstacles
                 + profile.creatures):
        if spec.name == name:
            return spec.min_dist
    return 0.0


def _min_pair_distance(points, min_dist=1.0):
    if len(points) < 2:
        return float("inf")
    cell = max(float(min_dist), 0.5)
    grid = {}
    best = float("inf")
    for x, y in points:
        cx, cy = int(x / cell), int(y / cell)
        for ix in range(cx - 1, cx + 2):
            for iy in range(cy - 1, cy + 2):
                for px, py in grid.get((ix, iy), ()):
                    d = math.hypot(x - px, y - py)
                    if d < best:
                        best = d
        grid.setdefault((cx, cy), []).append((x, y))
    return best


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


UE_SPAWNER = '''"""Run inside Unreal Editor 4.27: spawn real forest assets from the CSV.

Generated by scripts/generate_placement.py. No placeholder cubes are used: if
an asset path is missing or cannot be loaded, the row is skipped and reported.
Configure ForestAirSim/Config/asset_map.json before running.
"""
import csv
import json
import os
import unreal

PLACEMENT = r"{placement}"
VEGETATION = r"{vegetation}"
ASSET_MAP = r"{asset_map}"
REPORT = r"{report}"

def load_map():
    with open(ASSET_MAP) as handle:
        return json.load(handle)

def asset_for(mapping, key):
    entry = mapping.get("classes", {{}}).get(key)
    if not entry:
        return None
    path = entry.get("path", "")
    if not path:
        return None
    return entry

def load_uobject(path):
    if path.startswith("/Game/") and not path.endswith("_C"):
        bp_class = unreal.EditorAssetLibrary.load_blueprint_class(path)
        if bp_class:
            return bp_class
    return unreal.EditorAssetLibrary.load_asset(path)

def ground_z(x, y):
    """Line-trace the landscape; CSV elevation is the fallback."""
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
        start = unreal.Vector(x * 100.0, y * 100.0, 100000.0)
        end = unreal.Vector(x * 100.0, y * 100.0, -100000.0)
        hit = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            True, [], unreal.DrawDebugTrace.NONE, True)
        if hit and hit[0]:
            return hit[1].z / 100.0
    except Exception as exc:
        unreal.log_warning("terrain trace unavailable, using CSV z: %s" % exc)
    return None

def spawn(entry, x, y, z, yaw, scale):
    location = unreal.Vector(x * 100.0, y * 100.0, z * 100.0 + entry.get("z_offset_cm", 0.0))
    rotation = unreal.Rotator(0.0, 0.0, yaw)
    asset = load_uobject(entry["path"])
    if asset is None:
        return None
    if entry.get("spawn_as", "static_mesh") == "actor":
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(asset, location, rotation)
        if actor:
            actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
        return actor
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(asset, location, rotation)
    if actor:
        actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    return actor

def spawn_instanced(entry, instances):
    """Best-effort InstancedStaticMeshComponent group for foliage."""
    try:
        mesh = load_uobject(entry["path"])
        if mesh is None:
            return 0
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.Actor, unreal.Vector(0, 0, 0))
        component = unreal.new_object(unreal.InstancedStaticMeshComponent, actor)
        component.set_static_mesh(mesh)
        component.set_mobility(unreal.ComponentMobility.STATIC)
        actor.add_instance_component(component, "Foliage")
        for (x, y, z, yaw, scale) in instances:
            transform = unreal.Transform(
                location=unreal.Vector(x * 100.0, y * 100.0, z * 100.0),
                rotation=unreal.Rotator(0.0, 0.0, yaw),
                scale=unreal.Vector(scale, scale, scale))
            component.add_instance(transform, True)
        return len(instances)
    except Exception as exc:
        unreal.log_warning("instancing failed, spawn individually: %s" % exc)
        return 0

def main():
    mapping = load_map()
    stats = {{"spawned": 0, "missing_asset": 0, "missing_entries": {{}}, "instanced": 0}}
    with open(PLACEMENT) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        entry = asset_for(mapping, row["asset_key"])
        if entry is None:
            key = row["asset_key"]
            stats["missing_entries"][key] = stats["missing_entries"].get(key, 0) + 1
            stats["missing_asset"] += 1
            continue
        x, y = float(row["x"]), float(row["y"])
        traced = ground_z(x, y)
        z = traced if traced is not None else float(row["z"])
        actor = spawn(entry, x, y, z, float(row["yaw"]), float(row["scale"]))
        if actor is None:
            stats["missing_asset"] += 1
            continue
        actor.set_actor_label(row["id"])
        # The id tag lets AirSim find/set the object pose by name later
        # (AirSim FindActor matches tags or the internal object name).
        actor.set_editor_property(
            "tags", [row["id"], row["class"], row["behavior"], row["group_id"]])
        stats["spawned"] += 1

    instances = {{}}
    with open(VEGETATION) as handle:
        for row in csv.DictReader(handle):
            entry = asset_for(mapping, row["asset_key"])
            if entry is None:
                key = row["asset_key"]
                stats["missing_entries"][key] = stats["missing_entries"].get(key, 0) + 1
                continue
            x, y = float(row["x"]), float(row["y"])
            traced = ground_z(x, y)
            z = traced if traced is not None else float(row["z"])
            instances.setdefault(row["asset_key"], []).append(
                (x, y, z, float(row["yaw"]), float(row["scale"])))
    for key, batch in instances.items():
        entry = asset_for(mapping, key)
        done = spawn_instanced(entry, batch)
        stats["instanced"] += done
        if done == 0:
            for (x, y, z, yaw, scale) in batch:
                if spawn(entry, x, y, z, yaw, scale):
                    stats["spawned"] += 1

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as handle:
        json.dump(stats, handle, indent=2)
    unreal.log("Forest import report: " + json.dumps(stats))
    if stats["missing_asset"]:
        unreal.log_error(
            "Missing assets: fill ForestAirSim/Config/asset_map.json. "
            "No placeholder geometry was spawned.")

main()
'''


def write_ue_script(profile):
    UE_PY.parent.mkdir(parents=True, exist_ok=True)
    UE_PY.write_text(UE_SPAWNER.format(
        placement=OUT_CSV.as_posix(),
        vegetation=OUT_VEG_CSV.as_posix(),
        asset_map=(ROOT / "ForestAirSim" / "Config" / "asset_map.json").as_posix(),
        report=(ROOT / "ForestAirSim" / "Docs" / "import-report.json").as_posix()))


def load_config(path):
    data = json.loads(Path(path).read_text())
    overrides = {}
    if "world" in data:
        overrides["world"] = tuple(data["world"])
    for key in ("open_bowls", "corridors"):
        if key in data:
            overrides[key] = data[key]
    for key in ("trees", "vegetation", "obstacles", "creatures"):
        if key in data:
            current = {s.name: s for s in getattr(PROFILES[data.get("base", "dense")], key)}
            specs = []
            for item in data[key]:
                base = current.get(item["name"])
                if base is None:
                    base = Species(item["name"], 0, 1.0, 1.0, 1.0)
                for field_name, value in item.items():
                    if field_name == "class":
                        continue
                    setattr(base, field_name, value)
                specs.append(base)
            overrides[key] = specs
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="dense")
    parser.add_argument("--config", help="JSON overrides for the selected profile")
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--veg-out", default=str(OUT_VEG_CSV))
    parser.add_argument("--summary-out", default=str(OUT_JSON))
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    overrides = load_config(args.config) if args.config else None
    rows, veg_rows, summary = generate(profile, args.seed, overrides)

    out_csv = Path(args.out)
    out_veg = Path(args.veg_out)
    out_json = Path(args.summary_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, FIELDS, rows)
    write_csv(out_veg, VEG_FIELDS, veg_rows)
    out_json.write_text(json.dumps(summary, indent=2))
    if out_csv == OUT_CSV and out_veg == OUT_VEG_CSV:
        write_ue_script(profile)
    print(json.dumps({k: summary[k] for k in
                      ("profile", "total_actors", "total_vegetation",
                       "tree_count", "min_spacing_ok", "elevation_min",
                       "elevation_max")}, indent=2))
    print(f"wrote {out_csv} ({len(rows)} rows)")
    print(f"wrote {out_veg} ({len(veg_rows)} rows)")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
