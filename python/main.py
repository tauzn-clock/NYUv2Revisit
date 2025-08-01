import numpy as np 
from PIL import Image
from tnnr_apgl import tnnr_apgl
from psnr import psnr
import matplotlib.pyplot as plt

COLOR_IMG_PATH = "/scratchdata/processed/alcove2/rgb/0.png"
DEPTH_IMG_PATH = "/scratchdata/processed/alcove2/depth/0.png"

depth = Image.open(DEPTH_IMG_PATH)
depth = np.array(depth, dtype=np.float32)

print("Max depth: ", depth.max())

mask = depth != 0

reconstructed = tnnr_apgl(depth, mask, R=20, l=0.005, eps=0.01)
print("Max reconstructed: ", reconstructed.max(), "Min reconstructed: ", reconstructed.min())
reconstructed = np.clip(reconstructed, 0, reconstructed.max())
plt.imsave("reconstructed_depth.png", reconstructed, cmap='gray')


