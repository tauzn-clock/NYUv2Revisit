import numpy as np
from PIL import Image
import csv
import os

ORIGIN_DIR = "/scratchdata/InformationOptimisation"
DEST_DIR = "/scratchdata/InformationOptimisationCrop"

for INDEX in range(1449):
    rgb_path = "rgb/" + str(INDEX) + ".png"
    depth_path = "depth/" + str(INDEX) + ".png"
    
    rgb = Image.open(os.path.join(ORIGIN_DIR, rgb_path))
    rgb = rgb.crop((43, 45, 608, 472))
    save_path = os.path.join(DEST_DIR, rgb_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rgb.save(os.path.join(DEST_DIR, rgb_path))
    
    depth = Image.open(os.path.join(ORIGIN_DIR, depth_path))
    depth = depth.crop((43, 45, 608, 472))
    save_path = os.path.join(DEST_DIR, depth_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth.save(os.path.join(DEST_DIR, depth_path))
    
   