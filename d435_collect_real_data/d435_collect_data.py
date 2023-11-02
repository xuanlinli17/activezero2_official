from pathlib import Path

import numpy as np
import time
import os
import glob
import pyrealsense2 as rs

from real_robot.sensors.camera import CameraConfig, Camera
from real_robot.utils.realsense import get_connected_rs_devices

def capture(camera: Camera, save_dir: Path, tag: str):
    camera.take_picture()
    np.savez(save_dir / f"{tag}_images.npz", **camera.get_images())
    np.savez(save_dir / f"{tag}_params.npz", **camera.get_params())

if __name__ == "__main__":
    devices = get_connected_rs_devices()
    root_dir = '/hdd/xuanlin/d435_collect_real_data/data/'
    power_list = [0, 60, 120, 180, 240, 300, 360]

    save_dir = Path(root_dir).resolve()
    save_idx = 0
    while all([os.path.exists(f'{str(save_dir)}/{save_idx}_power_{power}_images.npz') for power in power_list]):
        save_idx += 1

    # NOTE: replace device_sn with the one you have. See terminal output
    rs_so_objs = glob.glob('/dev/shm/*rs*')
    for rs_so_obj in rs_so_objs:
        print("Removing", rs_so_obj)
        os.remove(rs_so_obj)
    print(f"*** Generating data {save_idx} ***")
    for power in power_list:
        camera = Camera(
            CameraConfig(
                "camera", '146322076186', config={
                    "Color": (848, 480, 60),
                    "Depth": (848, 480, 60),
                    "Infrared 1": (848, 480, 60),
                    "Infrared 2": (848, 480, 60),
                }, preset="Default",
                depth_option_kwargs={rs.option.laser_power: power}  # Change this to adjust laser power
            ),
        )
        time.sleep(0.1)
        capture(camera, save_dir=save_dir, tag=f"{save_idx}_power_{power}")
        time.sleep(0.1)
        del(camera)
        time.sleep(0.2)
