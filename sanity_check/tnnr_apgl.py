import numpy as np

def tnnr_apgl(ori, mask, R, l):
    # SVD decomposition
    
    A, S, Bt = np.linalg.svd(ori, full_matrices=False)
    A = A[:, :R]
    Bt = Bt[:R, :]
    AB = A @ Bt
    
    t = 1.0
    X = ori.copy()
    Y = AB.copy()
    for _ in range(200):
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
        print("X diff", np.linalg.norm(X - Xlast, 'fro'))
        
        # Check objective function value
        nuclear_norm = np.sum(sigma)
        trace = np.trace(A.T @ X @ Bt.T)
        fro_norm = np.linalg.norm((X - ori) * mask, 'fro')
        
        objective_value = nuclear_norm + trace + (l/2) * fro_norm**2
        print("Objective value:", objective_value)
        
    print("t", t)
    
    return X