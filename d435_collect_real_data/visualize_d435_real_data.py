from matplotlib import pyplot as plt
import numpy as np
import cv2

idx = 173

power_list = [0, 60, 120, 180, 240, 300, 360]

plt.figure(figsize=(30, 30))
for i, power in enumerate(power_list):
    f = np.load(f'/hdd/xuanlin/d435_collect_real_data/data/{idx}_power_{power}_images.npz')
    plt.subplot(len(power_list),3,i * 3 + 1)
    plt.imshow(f['rgb'])
    print("rgb min", f['rgb'].min(), "rgb max", f['rgb'].max())
    plt.subplot(len(power_list),3,i * 3 + 2)
    plt.imshow(cv2.cvtColor(f['ir_l'], cv2.COLOR_GRAY2BGR))
    print("irl min", f['ir_l'].min(), "irl max", f['ir_l'].max())
    plt.subplot(len(power_list),3,i * 3 + 3)
    plt.imshow(cv2.cvtColor(f['ir_r'], cv2.COLOR_GRAY2BGR))
    print("irr min", f['ir_r'].min(), "irr max", f['ir_r'].max())

plt.show()
plt.close()