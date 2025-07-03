import os
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEPTH_INPAINTING = "/NYUv2Revisit/depthInpainting/build/depthInpainting"
DEPTH_IMG_PATH = "/scratchdata/processed/long/depth/10.png"
MASK_IMG_PATH = "mask.png"
OUTPUT_IMG_PATH = "out.png"

# Generate mask
depth = Image.open(DEPTH_IMG_PATH)
depth = np.array(depth, dtype=np.float32)
print(depth.max(), depth.min())

mask = depth != 0
print(mask.max(), mask.min())
plt.imsave("mask.png", mask, cmap='gray')

subprocess.run([DEPTH_INPAINTING, "L", DEPTH_IMG_PATH, MASK_IMG_PATH, OUTPUT_IMG_PATH])

# Color output
out = Image.open(OUTPUT_IMG_PATH)
out = np.array(out)
plt.imsave("out_color.png", out)

