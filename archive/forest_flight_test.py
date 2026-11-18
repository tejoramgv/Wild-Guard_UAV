"""TEST 1-18 runner (Windows host with sim running). Requires: pip install airsim numpy opencv-python."""
import airsim

HOST, CAM, VEHICLE = "127.0.0.1", "front-center", "ForestDrone"  # set LAN IP when called from Ubuntu
log = lambda m: print(f"[forest-test] {m}", flush=True)

def main():
    c = airsim.MultirotorClient(ip=HOST)
    c.confirmConnection()
    c.enableApiControl(True, vehicle_name=VEHICLE)
    c.armDisarm(True, vehicle_name=VEHICLE)
    log("T3 spawn/ARM ok")
    c.takeoffAsync(vehicle_name=VEHICLE).join()
    log("T4 takeoff ok")
    c.hoverAsync(vehicle_name=VEHICLE).join()
    log("T5 hover ok")
    # T6 fly-through: forward/back/left/right/up/down/yaw
    for vx, vy, vz, dur, name in [(2,0,0,3,"fwd"),(-2,0,0,3,"back"),(0,2,0,2,"right"),(0,-2,0,2,"left"),(0,0,-1,2,"up"),(0,0,1,2,"down")]:
        c.moveByVelocityAsync(vx, vy, vz, dur, vehicle_name=VEHICLE).join()
        log(f"T6/T12 {name} ok col={c.simGetCollisionInfo(vehicle_name=VEHICLE).has_collided}")
    c.rotateToYawAsync(90, vehicle_name=VEHICLE).join()
    log("yaw ok")
    # T7 camera RGB
    resp = c.simGetImages([airsim.ImageRequest(CAM, airsim.ImageType.Scene, False, False)], vehicle_name=VEHICLE)[0]
    assert (resp.width, resp.height) == (640, 480), f"unexpected {resp.width}x{resp.height}"
    import numpy as np
    img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(resp.height, resp.width, 3)
    assert img.mean() > 15, "image too dark - fix daylight"
    log(f"T7 RGB ok {resp.width}x{resp.height} mean={img.mean():.1f} fov~60 pitch~-11.5")
    log("T8-T11 manual: waypoint drone facing tree/human/deer/boar/elephant/monkey/rock and confirm in image (see tests.md).")
    log(f"T13-15 collision API readable has_collided={c.simGetCollisionInfo(vehicle_name=VEHICLE).has_collided}")
    c.landAsync(vehicle_name=VEHICLE).join()
    log("T18 land ok — T16/17 ROS topics verified separately on Ubuntu (see ros2_bridge/)")

if __name__ == "__main__":
    main()
