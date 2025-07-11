import numpy as np
from PIL import Image
import csv
import os

ORIGIN_DIR = "/scratchdata/nyu_depth_v2/sync"
DEST_DIR = "/scratchdata/nyu_depth_crop/train"

CSV_FILE = "/NYUv2Revisit/crop_nyu/nyudepthv2_train_files_with_gt_dense.txt"

files = []

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        files.append(row[0].split())

for f in files:

    rgb_path = f[0]
    depth_path = f[1]
    mask_path = f[1][:-15] + "normal_" + f[1][-9:]
    
    rgb = Image.open(os.path.join(ORIGIN_DIR, rgb_path))
    rgb.crop((43, 45, 608, 472))
    
    save_path = os.path.join(DEST_DIR, rgb_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rgb.save(os.path.join(DEST_DIR, rgb_path))
    
    depth = Image.open(os.path.join(ORIGIN_DIR, depth_path))
    depth.crop((43, 45, 608, 472))
    save_path = os.path.join(DEST_DIR, depth_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth.save(os.path.join(DEST_DIR, depth_path))
    
    normal = Image.open(os.path.join(ORIGIN_DIR, mask_path))
    normal.crop((43, 45, 608, 472))
    save_path = os.path.join(DEST_DIR, mask_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    normal.save(os.path.join(DEST_DIR, mask_path))