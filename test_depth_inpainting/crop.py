from PIL import Image

DEPTH_IMG_PATH = "/scratchdata/InformationOptimisation/depth/0.png"
OUTPUT_IMG_PATH = "crop.png"

depth = Image.open(DEPTH_IMG_PATH)
depth = depth.crop((43, 45, 608, 472))  # Crop
depth.save(OUTPUT_IMG_PATH)