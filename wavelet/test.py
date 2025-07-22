import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt

# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2

print(LL.shape, LH.shape, HL.shape, HH.shape)

fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs[0, 0].imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
axs[0, 0].set_title(titles[0], fontsize=10)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 1].imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
axs[0, 1].set_title(titles[1], fontsize=10)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[1, 0].imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
axs[1, 0].set_title(titles[2], fontsize=10)
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 1].imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
axs[1, 1].set_title(titles[3], fontsize=10)
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])

fig.tight_layout()

plt.savefig("wavelet_transform.png")

undersample_rate = .5
n = original.shape[0] * original.shape[1]
original_undersampled = ( original.reshape(-1) \
    * np.random.permutation( 
        np.concatenate( 
            (np.ones( int( n * undersample_rate ) ), 
             np.zeros( int( n * ( 1-undersample_rate )) )) 
        ) 
    ) 
                        ).reshape(512,512)

def flat_wavelet_transform2(x, method='bior1.3'):
    """For a 2D image x, take the wavelet """
    coeffs = pywt.wavedec2( x, method )
    output = coeffs[0].reshape(-1)
    for tups in coeffs[1:]:
        for c in tups:
            output = np.concatenate((output, c.reshape(-1)))
    return output

def inverse_flat_wavelet_transform2(X,  shape, method='bior1.3'):
    shapes = pywt.wavedecn_shapes( shape , method)
    nx = shapes[0][0]
    ny = shapes[0][1]
    n = nx * ny
    coeffs = [X[:n].reshape(nx,ny) ]
    for i, d in enumerate(shapes[1:]):
        vals=list(d.values())
        nx = vals[0][0]
        ny = vals[0][1]
        coeffs.append( (X[ n : n + nx * ny].reshape( nx, ny ), 
                        X[ n + nx * ny : n + 2 * nx * ny ].reshape( nx, ny ), 
                        X[ n + 2 * nx * ny : n + 3 * nx * ny ].reshape( nx, ny ))  )
        n += 3 * nx * ny
    return pywt.waverec2(coeffs, method)



methods = ['haar','coif1','coif2','coif3','bior1.1','bior1.3','bior3.1','bior3.3','rbio1.1','rbio1.3','rbio3.1','rbio3.3']
def distance(x,y):
    return sum(abs(x.reshape(-1)-y.reshape(-1)))

# undersampled noisy signal in image-space and let this be first order Xhat
y = original_undersampled

# Repeat steps 1-4 until change is below a threshold
eps = 1e-2
lam = 100
lam_decay = 0.995
minlam = 1

err2=[]


lam = 100


def soft_thresh(x, lam):
    if not (isinstance(x[0], complex)):
        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam) 
    else:
        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam) 



xhat = y.copy()
for i in range(80):
    method = 'sym3'
    xhat_old = xhat
    Xhat_old = flat_wavelet_transform2(xhat, method)
    Xhat = soft_thresh(Xhat_old, lam)
    xhat = inverse_flat_wavelet_transform2(Xhat, (512,512), method)
    xhat[y!=0] = y[y!=0]   


    xhat = xhat.astype(int)
    xhat[xhat<0] = 0
    xhat[xhat>255] = 255
    err2.append(distance(original, xhat))
    lam *= lam_decay 
    
plt.imsave("wavelet_reconstructed.png", xhat, cmap='gray')