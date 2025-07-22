import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

from depthanything_interface import get_model, rescale_pred

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_PATH = "/scratchdata/nyu_depth_crop/train/bedroom_0004/rgb_00000.jpg"
DEPTH_PATH = "/scratchdata/nyu_depth_crop/train/bedroom_0004/sync_depth_00000.png"

rgb = cv2.imread(IMG_PATH)

plt.imsave("rgb.png", rgb)

depth = Image.open(DEPTH_PATH)
depth = np.array(depth, dtype=np.float32)
depth /= 1000.0  # Convert mm to meters

plt.imsave("depth.png", depth, cmap='gray')

model = get_model()
model.to("cuda:0")
model.eval()

pred = model.infer_image(rgb) # HxW depth map in meters in numpy

print(pred.max())

plt.imsave("depth_anything_v2.png", pred, cmap='gray')

ratio = depth / pred
ratio[np.isnan(ratio)] = 0
print(ratio[ratio!=0].max(), ratio[ratio!=0].min(), ratio[ratio!=0].mean())

avg = ratio[ratio!=0].mean()
ratio[ratio != 0] = avg  # Fill in the non-zero values with the average

plt.imsave("ratio.png", ratio, cmap='gray')

from tnnr_apgl import tnnr_apgl

ratio_corrected = tnnr_apgl(ratio, ratio != 0, R=20, l=0.005, eps=0.01)

print(ratio_corrected.max(), ratio_corrected.min(), ratio_corrected.mean())
plt.imsave("ratio_corrected.png", ratio_corrected, cmap='gray')
