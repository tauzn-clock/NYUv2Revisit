import numpy as np

def psnr(ori, pred, mask):    
    error = (ori - pred) ** 2
    mse = np.mean(error[mask])
    
    if mse == 0:
        return float('inf')
    else:
        max_pixel = ori.max()
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr_value