import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Test Grayscale

img = Image.open("/scratchdata/processed/alcove2/depth/0.png")
gray = np.array(img)

gray = gray[50:-50, 50:-50]  # Crop the image to remove borders

plt.imsave("gray.png", gray, cmap='gray')

# Histogram of the grayscale image
plt.hist(gray.ravel(), bins=(gray.max() - gray.min())//20, range=(gray.min(), gray.max()), color='black')
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("histogram_gray.png")

# Mask
mask = gray != 0
plt.imsave("mask.png", mask, cmap='gray')

X = gray * mask
plt.imsave("masked_gray.png", X, cmap='gray')

# SVD X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Reconstruct using the first k singular values
k = 50
X_reconstructed = np.dot(U[:, :k] * S[:k], Vt[:k, :])
print(X_reconstructed.max(), X_reconstructed.min())
plt.imsave("reconstructed_gray.png", X_reconstructed, cmap='gray')

X_reconstructed = X_reconstructed.astype(np.int16)
print(X_reconstructed.max(), X_reconstructed.min())
# Histogram of the reconstructed image
plt.hist(X_reconstructed.ravel(), bins=(X_reconstructed.max() - X_reconstructed.min())//20, range=(X_reconstructed.min(), X_reconstructed.max()), color='black')
plt.title("Histogram of Reconstructed Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("histogram_reconstructed.png")