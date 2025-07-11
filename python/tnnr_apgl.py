import numpy as np
from tqdm import tqdm

def tnnr_apgl(ori, mask, R, l):
    # SVD decomposition
    
    A, S, Bt = np.linalg.svd(ori, full_matrices=False)
    A = A[:, :R]
    Bt = Bt[:R, :]
    AB = A @ Bt
    
    # The value of t does not affect convergence, 
    # what matters more is the sequence of t, that is needed to satisfy convergence
    t = 1.0
    X = ori.copy()
    Y = AB.copy()
    
    pbar = tqdm(range(1000), desc="APGL Iterations")
    for i in pbar:
        # Update X
        Xlast = X.copy()
        temp = Y + t * (AB - l * (Y - ori) * mask)
        u, sigma, vt = np.linalg.svd(temp, full_matrices=False)
        sigma = np.maximum(sigma - t, 0)
        X = u @ np.diag(sigma) @ vt
        
        # Update t
        tlast = t
        t = (1 + np.sqrt(1 + 4 * tlast**2)) / 2
        
        # Update Y
        Y = X + (tlast - 1) / t * (X - Xlast)
        
        # Check Frobenius norm of the difference
        X_diff = np.linalg.norm(X - Xlast, 'fro')
        
        # Check objective function value
        nuclear_norm = np.sum(sigma)
        trace = np.trace(A.T @ X @ Bt.T)
        fro_norm = np.linalg.norm((X - ori) * mask, 'fro')
        
        objective_value = nuclear_norm + trace + (l/2) * fro_norm**2
        pbar.set_postfix({
            'Frobenius Norm': X_diff,
            'Objective Value': objective_value
        })    
        
    return X