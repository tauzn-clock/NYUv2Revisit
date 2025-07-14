import os
import sys

sys.path.append("/NYUv2Revisit/python")

import subprocess
import numpy as np
from PIL import Image
import csv
import tqdm
import time
from threading import Thread, Lock
from matplotlib import pyplot as plt
from tnnr_apgl import tnnr_apgl

CSV_FILE = "/NYUv2Revisit/crop_nyu/nyudepthv2_train_files_with_gt_dense.txt"
ORIGIN_DIR = "/scratchdata/nyu_depth_crop/train"
DEST_DIR = "/scratchdata/nyu_depth_tnnr_apgl/train"

files = []

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        files.append(row[0].split())

DEPTH_INPAINTING = "/NYUv2Revisit/depthInpainting/build/depthInpainting"
MASK_IMG_PATH = "0" #"mask.png"

def worker(data_slice, pbar, lock):
    for INDEX in data_slice:
        DEPTH_IMG_PATH = os.path.join(ORIGIN_DIR, files[INDEX][1])
        OUTPUT_IMG_PATH = os.path.join(DEST_DIR, files[INDEX][1])
        os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)

        # Generate mask
        depth = Image.open(DEPTH_IMG_PATH)
        depth = np.array(depth, dtype=np.float32)
        print(depth.max(), depth.min())

        #subprocess.run([DEPTH_INPAINTING, "TNNR_APGL", DEPTH_IMG_PATH, MASK_IMG_PATH, OUTPUT_IMG_PATH])
        mask = depth > 0
        output = tnnr_apgl(depth, mask, R=20, l=0.005, eps=0.01)
        output = np.clip(output, 0, output.max())
        output = Image.fromarray(output.astype(np.uint16))
        output.save(OUTPUT_IMG_PATH)
        
        with lock:
            pbar.update(1)
    

data = list(range(len(files)))
num_threads = 4
chunk_size = len(data) // num_threads

lock = Lock()
pbar = tqdm.tqdm(total=len(data))
threads = []

for i in range(num_threads):
    start = i * chunk_size
    end = None if i == num_threads - 1 else (i + 1) * chunk_size
    t = Thread(target=worker, args=(data[start:end], pbar, lock))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

pbar.close()
    