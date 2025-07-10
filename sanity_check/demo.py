import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Test Grayscale

img = Image.open("/scratchdata/processed/alcove2/rgb/0.png")
gray = img.convert("L")
gray = np.array(gray)

plt.imsave("gray.png", gray, cmap='gray')

# Histogram of the grayscale image
plt.hist(gray.ravel(), bins=(gray.max() - gray.min()), range=(gray.min(), gray.max()), color='black')
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("histogram_gray.png")

# Mask
mask = np.ones_like(gray, dtype=np.uint8)
mask[100:200, 100:200] = 0  # Example mask region
# Random mask
mask = np.random.randint(0, 20, size=gray.shape, dtype=np.uint8)
mask = mask != 0
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
plt.hist(X_reconstructed.ravel(), bins=(X_reconstructed.max() - X_reconstructed.min()), range=(X_reconstructed.min(), X_reconstructed.max()), color='black')
plt.title("Histogram of Reconstructed Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("histogram_reconstructed.png")