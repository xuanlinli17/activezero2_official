import argparse
import os
import time
import glob

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Extract temporal IR pattern from temporal real images")
parser.add_argument(
    "-d",
    "--data-folder",
    type=str,
    default='/hdd/xuanlin/d435_collect_real_data/data/',
)
parser.add_argument("-p", "--patch", type=int, default=9)
parser.add_argument("-t", "--threshold", type=float, default=0.005)
args = parser.parse_args()


def get_smoothed_ir_pattern(diff: np.array, ks=9, threshold=0.005):
    diff = np.abs(diff)
    diff_avg = cv2.blur(diff, (ks, ks))
    ir = np.zeros_like(diff)
    ir[diff - diff_avg > threshold] = 1
    return ir


def main():
    prefix = glob.glob(os.path.join(args.data_folder, '*_power_360_images.npz')) 
    num = len(prefix)

    os.makedirs(os.path.join(args.data_folder, "binary_ir_patterns"), exist_ok=True)

    start = time.time()
    for idx in range(num):
        gripper_present = (idx <= 100 and idx != 93) # these data were collected with robot gripper present; we use ground truth gripper mask to remove the binary ir patterns corresponding to the robot gripper
        
        fnames = [os.path.dirname(prefix[idx]) + f"/{idx}" + f"_power_{i}_images.npz" for i in range(0,420,60)]
        if not os.path.exists(fnames[-1]):
            print(f"WARNING: Data index {idx} does not exist")
            continue
        if not os.path.exists(fnames[0]):
            # ir power=0 not present in this data entry
            fnames = fnames[1:]

        flist = [np.load(fname) for fname in fnames]

        for direction in ["ir_l", "ir_r"]:
            img_temp = np.stack([f[direction] for f in flist], axis=2)

            # Get regression fit on temporal images
            print(f"Generating {img_temp.shape[-1]} temporal {direction} pattern {idx}/{num} time: {time.time() - start:.2f}s")
            h, w, d = img_temp.shape
            x = np.linspace(0, d - 1, num=d, dtype=int).reshape(1, 1, -1)
            x = np.repeat(x, h, axis=0)
            x = np.repeat(x, w, axis=1)  # [H, W, D]
            x_avg = np.average(x, axis=-1).reshape(h, w, 1)

            y = img_temp  # [H, W, D]
            y_avg = np.average(y, axis=-1).reshape(h, w, 1)

            numerator = np.sum((y - y_avg) * (x - x_avg), axis=-1)
            denominator = np.sum((x - x_avg) ** 2, axis=-1)  # [H, W]
            slope = numerator / denominator  # [H, W]
            slope = slope[:, :, None]
            intercept = y_avg - slope * x_avg
            img_temp_fit = slope * x + intercept

            # Get IR pattern
            diff = (img_temp_fit[:, :, -1] - img_temp_fit[:, :, 0]) / 255
            # diff = np.abs(diff)
            # Normalize to [0,1]
            diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
            pattern = get_smoothed_ir_pattern(diff, ks=args.patch, threshold=args.threshold)

            if gripper_present:
                this_script_dir = os.path.dirname(__file__)
                gripper_mask = np.load(os.path.join(this_script_dir, f"{direction}_gripper_mask.npy")).astype(np.uint8)
                gripper_mask = cv2.dilate(gripper_mask, np.ones((args.patch + 4, args.patch + 4)))
                min_y = np.min(np.where(gripper_mask > 0)[0])
                # pattern[gripper_mask > 0] = 0
                pattern[min_y:, :] = 0

            # Save extracted IR pattern
            pattern = (pattern * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    args.data_folder, "binary_ir_patterns", f"{idx}_{direction}_real_temporal_ps{args.patch}_t{args.threshold}.png"
                ),
                pattern,
            )


if __name__ == "__main__":
    main()
