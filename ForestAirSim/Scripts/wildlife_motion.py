"""Windows-only: deterministic motion for humans and wildlife.

Actors are found by AirSim, which matches an actor tag equal to the object name
(simSetObjectPose -> FindActor checks tags or the internal name). ue_place_actors.py
tags every actor with its CSV id, so the ids below resolve.

    pip install airsim
    python ForestAirSim/Scripts/wildlife_motion.py --host 192.168.1.100

Movement styles are derived deterministically from the id hash: circles,
figure-eights and back-and-forth paths, at the speed stored in placement_out.csv.
Pose updates translate/rotate the actor; skeletal animation playback depends on
the asset's AnimBlueprint and is NOT guaranteed by this script.
"""
import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import airsim

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "ForestAirSim" / "Docs" / "placement_out.csv"

MOVING = {"wander", "path", "wander_air"}


def unit(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--csv", default=str(CSV_PATH))
    args = parser.parse_args()

    client = airsim.MultirotorClient(ip=args.host)
    client.confirmConnection()

    targets = []
    with open(args.csv) as handle:
        for row in csv.DictReader(handle):
            if row["behavior"] not in MOVING or float(row["speed"]) <= 0.0:
                continue
            try:
                pose = client.simGetObjectPose(row["id"])
            except Exception as exc:
                print(f"skip {row['id']}: {exc}")
                continue
            if math.isnan(pose.position.x_val):
                print(f"skip {row['id']}: not found")
                continue
            targets.append({
                "id": row["id"],
                "origin": (pose.position.x_val, pose.position.y_val, pose.position.z_val),
                "orientation": pose.orientation,
                "speed": float(row["speed"]),
                "behavior": row["behavior"],
                "phase": unit(row["id"]) * 2.0 * math.pi,
                "radius": 1.5 + 3.0 * unit(row["id"] + "r"),
                "axis": unit(row["id"] + "a") * 2.0 * math.pi,
            })
    print(f"moving {len(targets)} actors at {args.hz} Hz")

    period = 1.0 / args.hz
    start = time.time()
    try:
        while True:
            t = time.time() - start
            for target in targets:
                ox, oy, oz = target["origin"]
                speed, phase = target["speed"], target["phase"]
                if target["behavior"] == "path":
                    # Back-and-forth along a deterministic axis.
                    axis = target["axis"]
                    span = 6.0 + 6.0 * unit(target["id"] + "s")
                    offset = math.sin(t * speed / 3.0 + phase) * span
                    x = ox + offset * math.cos(axis)
                    y = oy + offset * math.sin(axis)
                    z = oz
                elif target["behavior"] == "wander_air":
                    radius = target["radius"] * 1.5
                    x = ox + radius * math.cos(t * speed + phase)
                    y = oy + radius * math.sin(t * speed + phase)
                    z = oz - 0.5 * math.sin(t * 0.5 + phase)
                else:
                    radius = target["radius"]
                    x = ox + radius * math.cos(t * speed + phase)
                    y = oy + radius * math.sin(t * speed * 0.7 + phase)
                    z = oz
                pose = airsim.Pose(
                    airsim.Vector3r(x, y, z), target["orientation"])
                client.simSetObjectPose(target["id"], pose, True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
