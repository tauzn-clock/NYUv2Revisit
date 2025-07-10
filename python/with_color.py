# See if color affects
rgb = Image.open(COLOR_IMG_PATH)
rgb = np.array(rgb, dtype=np.float32)

H, W, _ = rgb.shape

# Combien depth and rgb to get H x (W x 4) tensor
combined_rgb = np.concatenate((rgb, depth[..., np.newaxis]), axis=-1)
combined_mask = np.concatenate((np.ones_like(rgb), mask[..., np.newaxis]), axis=-1)
print("Combined RGB shape: ", combined_rgb.shape)
print("Combined Mask shape: ", combined_mask.shape) 

# Reshape to W x (H x 4)
X_2 = combined_rgb.swapaxes(0, 1).reshape(W, -1)
print("Combined RGB reshaped: ", X_2.shape)
mask_2 = combined_mask.swapaxes(0, 1).reshape(W, -1)
print("Combined Mask reshaped: ", mask_2.shape)

recombined_2 = tnnr_apgl(X_2, mask_2, R=20, l=0.01)
reconstructed_2 = recombined_2.reshape(W, H, 4)[:, :, 3]  # Get the depth channel
print("Max reconstructed: ", reconstructed_2.max(), "Min reconstructed: ", reconstructed_2.min())
print(reconstructed_2.shape)
reconstructed_2 = np.clip(reconstructed_2, 0, reconstructed_2.max())
plt.imsave("reconstructed_depth_rgb_transposed.png", reconstructed_2.T, cmap='gray')

# Reshape to H x (W x 4)
X_1 = combined_rgb.reshape(H, -1)
print("Combined RGB reshaped: ", X_1.shape)
mask_1 = combined_mask.reshape(H, -1)
print("Combined Mask reshaped: ", mask_1.shape)

recombined_1 = tnnr_apgl(X_1, mask_1, R=20, l=0.01)
reconstructed_1 = recombined_1.reshape(H, W, 4)[:, :, 3]  # Get the depth channel
print("Max reconstructed: ", reconstructed_1.max(), "Min reconstructed: ", reconstructed_1.min())
print(reconstructed_1.shape)
reconstructed_1 = np.clip(reconstructed_1, 0, reconstructed_1.max())
plt.imsave("reconstructed_depth_rgb.png", reconstructed_1, cmap='gray')
