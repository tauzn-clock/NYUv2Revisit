import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from tnnr_apgl import tnnr_apgl

def TV_NORM(X):
    X_LEFT = X[:-1, :] - X[1:, :]
    X_RIGHT = X[1:, :] - X[:-1, :]
    X_UP = X[:, :-1] - X[:, 1:]
    X_DOWN = X[:, 1:] - X[:, :-1]
    
    total = torch.sum(torch.abs(X_LEFT)) + torch.sum(torch.abs(X_RIGHT)) + torch.sum(torch.abs(X_UP)) + torch.sum(torch.abs(X_DOWN))
    total /= 2.0
    
    return total

def iter_U(M, Y, ori, mask, rho, lambda_tv):
    M = M.detach()
    Y = Y.detach()
    
    U = ori * mask
    U.requires_grad = True
    
    def objective_function(U, ori, mask, lambda_tv):
        obj_1 = torch.linalg.vector_norm((U - ori) * mask, ord=2)**2
        obj_2 = TV_NORM(U) * lambda_tv
        obj_3 = (rho / 2) * torch.linalg.vector_norm(U - M + Y, ord=2)**2
        
        return obj_1 + obj_2 + obj_3
    
    # Gradient descent with U to minimize the objective function
    
    lr = 0.01
    
    for i in range(100):
        loss = objective_function(U, ori, mask, lambda_tv)
        loss.backward()
        with torch.no_grad():
            U -= lr * U.grad
            U.grad.zero_()
            
    return U
    """
    U = ori
    U_padded = np.pad(U, ((1, 1), (1, 1)), mode='edge')
    
    U_left = U_padded[:-2, 1:-1] - U_padded[1:-1, 1:-1]
    U_right = U_padded[2:, 1:-1] - U_padded[1:-1, 1:-1]
    U_up = U_padded[1:-1, :-2] - U_padded[1:-1, 1:-1]
    U_down = U_padded[1:-1, 2:] - U_padded[1:-1, 1:-1]
    
    U_xx = U_left + U_right
    U_yy = U_up + U_down
    
    rho = 1.0
    
    def get_val(U, M, Y, ori, mask, rho, lambda_tv):
        U_tmp = np.pad(U, ((1, 1), (1, 1)), mode='edge')
        U_left = U_tmp[:-2, 1:-1] - U_tmp[1:-1, 1:-1]
        U_right = U_tmp[2:, 1:-1] - U_tmp[1:-1, 1:-1]
        U_up = U_tmp[1:-1, :-2] - U_tmp[1:-1, 1:-1]
        U_down = U_tmp[1:-1, 2:] - U_tmp[1:-1, 1:-1]
        
        obj_1 = np.linalg.norm((U-ori) * mask, ord='fro')**2
        obj_2 = abs(np.sum(U_left) + np.sum(U_right) + np.sum(U_up) + np.sum(U_down)) * lambda_tv
        obj_3 = (rho/2) * np.linalg.norm(U - M + Y, ord='fro')**2
        
        return obj_1 + obj_2 + obj_3
    
    grad_1 = 2 * (U - ori) * mask
    grad_2 = (np.sign(U_left) + np.sign(U_right) + np.sign(U_up) + np.sign(U_down)) * lambda_tv
    grad_3 = rho * (U - M + Y)
    
    dt = 1.0
    tau = 0.8
    for _ in range(10):
        print(get_val(U, M, Y, ori, mask, rho, lambda_tv))
        U -= dt * (grad_1 + grad_2 + grad_3)
        dt *= tau
    print("U max: ", U.max(), "U min: ", U.min())
    print(get_val(U, M, Y, ori, mask, rho, lambda_tv))
    """

def iter_M(U, Y, rho, lambda_nuc):
    X = (U + Y)
    A, S, Bt = torch.linalg.svd(X, full_matrices=False)
    S = torch.maximum(S - lambda_nuc/rho, torch.zeros_like(S))
    print(S.max(), S.min())
    M = A @ torch.diag(S) @ Bt
    
    return M
    
def iter_Y(Y, U, M):
    return Y + U - M

def tnnr_tv(ori, mask, rho = 1.0, lambda_tv = 40, lambda_nuc=1.0):
    ori = torch.from_numpy(ori).float()
    mask = torch.from_numpy(mask).float()
    Y = torch.zeros_like(ori, requires_grad=True)
    M = torch.zeros_like(ori, requires_grad=True)

    def objective_function(U, ori, mask, lambda_tv, lambda_nuc):
        obj_1 = torch.linalg.vector_norm((U - ori) * mask, ord=2)**2
        obj_2 = TV_NORM(U) * lambda_tv
        _, S, _ = torch.linalg.svd(U, full_matrices=False)
        obj_3 = lambda_nuc * torch.sum(S)
        
        return obj_1 + obj_2 + obj_3
    
    for _ in range(40):
        U = iter_U(M, Y, ori, mask, rho, lambda_tv)
        M = iter_M(U, Y, rho, lambda_nuc)
        Y = iter_Y(Y, U, M)
        
        print(objective_function(U, ori, mask, lambda_tv, lambda_nuc).item())

    return U
if __name__ == "__main__":
    DEPTH_IMG_PATH = "/scratchdata/processed/alcove2/depth/0.png"

    depth = Image.open(DEPTH_IMG_PATH)
    depth = np.array(depth, dtype=np.float32)
    
    mask = depth != 0
    
    output = tnnr_tv(depth, mask, lambda_tv=1000, lambda_nuc=1000)
    print("Max reconstructed: ", output.max().item(), "Min reconstructed: ", output.min().item())
    
    import matplotlib.pyplot as plt
    plt.imsave("reconstructed_depth_tv.png", output.detach().numpy(), cmap='gray')