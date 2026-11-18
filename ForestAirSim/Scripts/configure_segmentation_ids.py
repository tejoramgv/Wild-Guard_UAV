"""Windows-only: give warm-blooded actors unique stencil IDs for AirSim Infrared.

Run after the forest import, with the Unreal simulation running and the
settings.json SegmentationSettings set to MeshNamingMethod=StaticMeshName.

    pip install airsim
    python ForestAirSim/Scripts/configure_segmentation_ids.py --host 192.168.1.100

AirSim Infrared is documented as an object-ID -> grey-scale image, not a
radiometric thermal camera (docs/image_apis.md). This makes humans and animals
bright and the environment dark, which is a genuine, explicit thermal proxy.
Assets must be named so the regex matches their static mesh name, e.g.
SK_Human_Male, SK_Deer, SK_Elephant.
"""
import argparse
import json
from pathlib import Path

import airsim

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--map", default=str(ROOT / "ForestAirSim" / "Config" / "asset_map.json"))
    args = parser.parse_args()

    mapping = json.loads(Path(args.map).read_text()).get("segmentation", {})
    client = airsim.MultirotorClient(ip=args.host)
    client.confirmConnection()
    for name, stencil in mapping.items():
        # regex_match is a full match in AirSim, hence the .* wrappers.
        pattern = f".*{name}.*"
        ok = client.simSetSegmentationObjectID(pattern, int(stencil), True)
        print(f"{pattern:24s} -> {stencil:3d}  {'ok' if ok else 'NO MATCH'}")
    # Environment/background intentionally left at 0 (black in Infrared).
    print("done. Thermal camera ImageType=7 publishes /airsim_node/<v>/thermal/Infrared")


if __name__ == "__main__":
    main()
