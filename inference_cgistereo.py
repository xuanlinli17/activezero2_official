from active_zero2.models.cgi_stereo.cgi_stereo import CGI_Stereo
import torch, numpy as np
import cv2
from PIL import Image

# ckpt_path = 'model_oct17.pth'
# ckpt_path = 'model_oct19.pth'
# ckpt_path = 'model_oct19_veryclose.pth'
# ckpt_path = 'model.pth'
ckpt_path = 'model_oct23_balanced.pth'
img_resize = (424, 240)
# img_L_path = '/home/xuanlin/Downloads/capture_close/L0_Infrared.png'
# img_R_path = '/home/xuanlin/Downloads/capture_close/R0_Infrared.png'
# img_L_path = '/home/xuanlin/Downloads/capture_close/L1_360max_Infrared.png'
# img_R_path = '/home/xuanlin/Downloads/capture_close/R1_360max_Infrared.png'
# img_L_path = '/home/xuanlin/Downloads/modified-messy-table-dataset-test/data/238-5/0128_irL_kuafu_half.png'
# img_R_path = '/home/xuanlin/Downloads/modified-messy-table-dataset-test/data/238-5/0128_irR_kuafu_half.png'
img_L_path = '/home/xuanlin/Downloads/capture/60cm_irl.png'
img_R_path = '/home/xuanlin/Downloads/capture/60cm_irr.png'
device = 'cuda:0'
disp_conf_topk = 2
disp_conf_thres = 0.8 # 0.95
MAX_DISP = 256



img_L = np.array(Image.open(img_L_path).convert(mode="L")) / 255 # [480, 848]
img_R = np.array(Image.open(img_R_path).convert(mode="L")) / 255
assert len(img_L.shape) == len(img_R.shape) == 2, f"Image shape {img_L.shape} {img_R.shape} not supported"
orig_h, orig_w = img_L.shape
img_L = cv2.resize(img_L, img_resize, interpolation=cv2.INTER_CUBIC) # shape img_resize
img_R = cv2.resize(img_R, img_resize, interpolation=cv2.INTER_CUBIC)

model = CGI_Stereo(
    maxdisp=MAX_DISP,
)
model.load_state_dict(torch.load(ckpt_path)['model'])
model = model.to(device)

img_L = torch.from_numpy(img_L).float().to(device)[None, None, ...] # [1, 1, *img_resize]
img_R = torch.from_numpy(img_R).float().to(device)[None, None, ...] # [1, 1, *img_resize]

import time
with torch.no_grad():
    tt = time.time()
    pred_dict = model({'img_l': img_L / 1.5, 'img_r': img_R * 2.0})
    torch.cuda.synchronize()
    print(time.time() - tt)
    pred_dict = model({'img_l': img_L, 'img_r': img_R})
    torch.cuda.synchronize()
    print(time.time() - tt)
for k in pred_dict:
    pred_dict[k] = pred_dict[k].detach().cpu().numpy()
img_L, img_R = img_L.squeeze().cpu().numpy(), img_R.squeeze().cpu().numpy()    
    
disparity = pred_dict['pred_orig'] # [1, H, W]
disparity = disparity.squeeze() # [H, W]
disparity_probs = pred_dict['cost_prob'].squeeze() # [1, disp//4, H, W]
top_disparity_prob_idx = np.argpartition(-disparity_probs, disp_conf_topk, axis=0)[:disp_conf_topk, :, :]
disparity_confidence = np.take_along_axis(disparity_probs, top_disparity_prob_idx, axis=0).sum(axis=0) # [H, W]
disparity_conf_mask = disparity_confidence > disp_conf_thres

focal_length = 430.139801025391 * img_resize[0] / orig_w
baseline = np.linalg.norm(
    np.array([0.000156505, -0.01489765, -1.15314942e-05])
    - np.array([-0.000328569, -0.06504793, 0.000665888])
)

depth = focal_length * baseline / (disparity + 1e-5)
# filter out depth
depth[~disparity_conf_mask] = 0.0

# visualize matching img
from matplotlib import pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

matching_img = np.zeros([img_L.shape[0] * 2, img_L.shape[1] * 2])
matching_img[:img_L.shape[0], :img_L.shape[1]] = img_L
matching_img[:img_L.shape[0], :img_L.shape[1]][~disparity_conf_mask] = 0
matching_img[img_L.shape[0]:, img_L.shape[1]:] = img_R
plt.imshow(matching_img)
for i in range(10, img_L.shape[0]-10, 20):
    for j in range(10, img_L.shape[1]-10, 20):
        if not disparity_conf_mask[i, j]:
            continue
        rand_color = colors[np.random.randint(len(colors))]
        plt.plot([j, j - disparity[i, j] + img_L.shape[1]], [i, i + img_L.shape[0]], 'o', color=rand_color)
        plt.plot([j, j - disparity[i, j] + img_L.shape[1]], [i, i + img_L.shape[0]], linewidth=1.0, color=rand_color)
        
# plt.imshow(depth)
plt.show()
np.save('/home/xuanlin/Downloads/test_depth_0_Infrared_new.npy', depth)
# Image.save('/home/xuanlin/Downloads/test_depth.png', depth)
print("asdf")