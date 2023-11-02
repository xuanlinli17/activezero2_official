import argparse
import os
import time
import glob
import pickle

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

    meta_info = {
        "extrinsic": np.eye(4),
        "extrinsic_l": np.array([[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, 0.00015650546993128955],
                                              [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.014897654764354229],
                                              [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, -1.1531494237715378e-05],
                                              [0.0, 0.0, 0.0, 1.0]]),
        "extrinsic_r": np.array([[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, -0.0003285693528596312],
                                              [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.06504792720079422],
                                              [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, 0.0006658887723460793],
                                              [0.0, 0.0, 0.0, 1.0]]),
        "intrinsic": np.array([[605.12158203125,     0., 424.5927734375], [0.,    604.905517578125, 236.668975830078], [0, 0, 1]]),
        "intrinsic_l": np.array([[430.139801025391,   0., 425.162841796875], [0.,   430.139801025391, 235.276519775391], [0, 0, 1]]),
        "intrinsic_r": np.array([[430.139801025391,   0., 425.162841796875], [0.,   430.139801025391, 235.276519775391], [0, 0, 1]]),
    }

    start = time.time()
    for idx in range(num):
        gripper_present = (idx <= 100 and idx >= 2 and idx != 93) # these data were collected with robot gripper present; we use ground truth gripper mask to remove the binary ir patterns corresponding to the robot gripper
        
        power_list = list(range(0, 420, 60))
        fnames = [os.path.dirname(prefix[idx]) + f"/{idx}" + f"_power_{pwr}_images.npz" for pwr in power_list]
        if not os.path.exists(fnames[-1]):
            print(f"WARNING: Data index {idx} does not exist")
            continue
        if not os.path.exists(fnames[0]):
            # ir power=0 not present in this data entry
            fnames = fnames[1:]
            power_list = power_list[1:]

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
            
            power_list_nonzero = power_list if power_list[0] != 0 else power_list[1:]
            f_offset = 0 if power_list[0] != 0 else 1
            for pwr_id, pwr in enumerate(power_list_nonzero):
                cur_dirname = os.path.join(args.data_folder, "processed", f"{idx}-{pwr_id}")
                os.makedirs(cur_dirname, exist_ok=True)
                cv2.imwrite(
                    os.path.join(
                        cur_dirname, f"{direction}_real_temporal_ps{args.patch}_t{args.threshold}.png"
                    ),
                    pattern,
                )
                cv2.imwrite(
                    os.path.join(
                        cur_dirname, f"{direction}_real.png"
                    ),
                    flist[pwr_id + f_offset][direction],
                )
                with open(os.path.join(cur_dirname, 'meta.pkl'), 'wb') as f:
                    pickle.dump(meta_info, f)



if __name__ == "__main__":
    main()
