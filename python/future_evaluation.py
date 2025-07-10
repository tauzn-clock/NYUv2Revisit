
N = 50
R = np.linspace(1, 50, 20).astype(int)
store_psnr = np.zeros_like(R, dtype=np.float32)

for r in range(len(R)):
    reconstructed = tnnr_apgl(depth, mask, R=R[r], l=0.01)
    print("Max reconstructed: ", reconstructed.max(), "Min reconstructed: ", reconstructed.min())
    reconstructed = np.clip(reconstructed, 0, reconstructed.max())
    store_psnr[r] = psnr(depth, reconstructed, mask)
    print("PSNR for rank {}: {}".format(R[r], store_psnr[r]))
    
plt.plot(R, store_psnr, marker='o')
plt.xlabel('Rank (R)')
plt.ylabel('PSNR')
plt.title('PSNR vs Rank')
plt.grid()  
plt.savefig("psnr_vs_rank.png")