import os
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

DEPTH_INPAINTING = "/NYUv2Revisit/depthInpainting/build/depthInpainting"
DEPTH_IMG_PATH = "/scratchdata/processed/alcove2/depth/10.png"
MASK_IMG_PATH = os.path.join(FILE_DIR, "mask.png")
OUTPUT_IMG_PATH = os.path.join(FILE_DIR, "out.png")

# Generate mask
depth = Image.open(DEPTH_IMG_PATH)
depth = np.array(depth, dtype=np.float32)
print(depth.max(), depth.min())

mask = depth != 0
print(mask.max(), mask.min())
plt.imsave(MASK_IMG_PATH, mask, cmap='gray')

subprocess.run([DEPTH_INPAINTING, "L", DEPTH_IMG_PATH, MASK_IMG_PATH, OUTPUT_IMG_PATH])

# Color output
out = Image.open(OUTPUT_IMG_PATH)
out = np.array(out)
plt.imsave(os.path.join(FILE_DIR,"out_color.png"), out)

