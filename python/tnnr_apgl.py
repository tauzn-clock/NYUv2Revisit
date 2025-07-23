import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_sort(data, name):
    print(data[:40])
    fig, ax = plt.subplots()
    ax.plot(data, marker='o', linestyle='-', color='blue')
    ax.set_title('Sorted Singular Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    
    fig.savefig(name)

def tnnr_apgl(ori, mask, R, l, eps=0.01):
    # SVD decomposition
    
    A, S, Bt = np.linalg.svd(ori, full_matrices=False)
    A = A[:, :R]
    Bt = Bt[:R, :]
    AB = A @ Bt
    
    # The value of t does not affect convergence, 
    # what matters more is the sequence of t, that is needed to satisfy convergence
    t = 1.0
    X = ori.copy()
    Y = ori.copy()
    
    #sigma_r_init = S[R]
    #smallest = 1.0
    
    pbar = tqdm(range(200), desc="APGL Iterations")
    for i in pbar:
        # Update X
        Xlast = X.copy()
        temp = Y + t * (AB - l * (Y - ori) * mask)
        u, sigma, vt = np.linalg.svd(temp, full_matrices=False)
    
        #if sigma[R]/ sigma_r_init - smallest > eps:
        #    break
        #smallest = min(smallest, sigma[R] / sigma_r_init)
    
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
        fro_norm = np.linalg.norm((X - ori)[mask!=0], ord=2)
        
        objective_value = nuclear_norm + trace + (l/2) * fro_norm**2
        pbar.set_postfix({
            'Frobenius Norm': X_diff,
            'Objective Value': objective_value
        })   
        
    return X

def tnnr_apgl_torch(ori, mask, R, l, eps=0.01):
    A, S, Bt = torch.linalg.svd(ori, full_matrices=False)
    A = A[:, :R]
    Bt = Bt[:R, :]
    AB = A @ Bt
    
    t = 1.0
    X = ori.clone()
    Y = ori.clone()
    
    pbar = tqdm(range(200), desc="APGL Iterations")
    for i in pbar:
        Xlast = X.clone()
        temp = Y + t * (AB - l * (Y - ori) * mask)
        u, sigma, vt = torch.linalg.svd(temp, full_matrices=False)
        
        sigma = torch.maximum(sigma - t, torch.zeros_like(sigma))
        X = u @ torch.diag(sigma) @ vt
        
        tlast = t
        t = (1 + (1 + 4 * tlast**2)**0.5) / 2
        
        Y = X + (tlast - 1) / t * (X - Xlast)
        
        X_diff = torch.linalg.norm(X - Xlast, 'fro')
        
        nuclear_norm = torch.sum(sigma)
        trace = torch.trace(A.T @ X @ Bt.T)
        fro_norm = torch.linalg.norm((X - ori) * mask, 'fro')
        
        objective_value = nuclear_norm + trace + (l/2) * fro_norm**2
        pbar.set_postfix({
            'Frobenius Norm': X_diff.item(),
            'Objective Value': objective_value.item()
        })
        
    del A, S, Bt, temp, u, sigma, vt, Xlast, Y, AB  # Free memory
    
    return X