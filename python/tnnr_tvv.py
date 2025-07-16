import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

def TVV_NORM(X):
    X_LEFT = X[:-1, :] - X[1:, :]
    X_UP = X[:, :-1] - X[:, 1:]
    X_xx = 2*X[1:-1, :] - X[:-2, :] - X[2:, :]
    X_yy = 2*X[:, 1:-1] - X[:, :-2] - X[:, 2:]

    total = torch.sum(torch.abs(X_xx)) + torch.sum(torch.abs(X_yy)) #+ torch.sum(torch.abs(X_LEFT)) + torch.sum(torch.abs(X_UP))
    #total /= 2.0
    
    return total

def iter_U(M, Y, ori, mask, rho, lambda_tv):
    M = M.detach()
    Y = Y.detach()
    
    U = ori * mask
    U.requires_grad = True
    
    def objective_function(U, ori, mask, lambda_tv):
        obj_1 = torch.linalg.vector_norm((U - ori) * mask, ord=2)**2
        obj_2 = TVV_NORM(U) * lambda_tv
        obj_3 = (rho / 2) * torch.linalg.vector_norm(U - M + Y, ord=2)**2
        
        return obj_1 + obj_2 + obj_3
    
    # Gradient descent with U to minimize the objective function
    
    lr = 0.01
    U_fro_prev = torch.inf
    pbar = tqdm(range(2000))
    
    for i in pbar:
        loss = objective_function(U, ori, mask, lambda_tv)
        loss.backward()
        with torch.no_grad():
            U -= lr * U.grad
            U.grad.zero_()
        
        U_fro = torch.linalg.norm(U, ord='fro').item()
        pbar.set_description(f"Loss: {loss.item():.4f}, U: {U_fro:.4f}")

        if (abs(U_fro - U_fro_prev) < 0.1):
            break
        U_fro_prev = U_fro
        
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
        obj_2 = TVV_NORM(U) * lambda_tv
        _, S, _ = torch.linalg.svd(U, full_matrices=False)
        obj_3 = lambda_nuc * torch.sum(S)
        
        return obj_1 + obj_2 + obj_3
    
    obj_prev = torch.inf
    
    for _ in range(20):
        U = iter_U(M, Y, ori, mask, rho, lambda_tv)
        M = iter_M(U, Y, rho, lambda_nuc)
        Y = iter_Y(Y, U, M)
        
        obj = objective_function(U, ori, mask, lambda_tv, lambda_nuc).item()
        if (abs(obj - obj_prev)/obj_prev < 1e-3):
            break
        print(obj)
    return U
if __name__ == "__main__":
    
    RGB_IMG_PATH = "/scratchdata/nyu_plane_crop/rgb/0.png"
    DEPTH_IMG_PATH = "/scratchdata/nyu_plane_crop/depth/0.png"

    depth = Image.open(DEPTH_IMG_PATH)
    depth = np.array(depth, dtype=np.float32)
    
    mask = depth != 0
    
    rho = 1.0    
    _, S, _ = np.linalg.svd(depth, full_matrices=False)
    lambda_nuc = 0.01 * S[0] / rho
    print("Lambda nuclear: ", lambda_nuc)
    
    lambda_tv = 1e1

    output = tnnr_tv(depth, mask, lambda_tv=lambda_tv, lambda_nuc=lambda_nuc, rho=rho)
    output = output.detach().numpy()
    print("Max reconstructed: ", output.max().item(), "Min reconstructed: ", output.min().item())
    print("Depth:", depth.max(), depth.min())

    import matplotlib.pyplot as plt
    plt.imsave("reconstructed_depth_tv.png", output, cmap='gray')
    plt.imsave("original_depth.png", depth, cmap='gray')
    
    from metrics import evaluateMetrics
    
    evaluateMetrics(depth, output)
    
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from test_depth_inpainting.visualise import img_over_pcd
    from test_depth_inpainting.process_depth import get_3d
    import open3d as o3d
    
    img = Image.open(RGB_IMG_PATH)
    img = np.array(img)

    pts_3d = get_3d(output, [518.8579, 518.8579, 282.5824, 208.7362])
    
    pcd = img_over_pcd(pts_3d, img)

    o3d.visualization.draw_geometries([pcd])
    
    test = output.copy()
    test[depth!=0] = depth[depth!=0]
    
    pts_3d = get_3d(test, [518.8579, 518.8579, 282.5824, 208.7362])
    pcd = img_over_pcd(pts_3d, img)
    o3d.visualization.draw_geometries([pcd])
    
    pts_3d = get_3d(depth, [518.8579, 518.8579, 282.5824, 208.7362])
    pcd = img_over_pcd(pts_3d, img)
    o3d.visualization.draw_geometries([pcd])
    