import math
import sys
import numpy as np
from matplotlib import pylab as plt
from sklearn.decomposition import PCA
import torch
from scipy.io import loadmat
import spectral as spy

# =============================================================================
# 依赖检查与导入
# =============================================================================
try:
    from tensor_function import Patch, getU, kmode_product, train_test_tensor, \
        train_test_zishiying, train_test_tensor_half, train_test_tensor_fold
    from tensor_function_MTLPP import dist_EMD
except ImportError:
    print("错误：未找到自定义张量模块。")
    print("请确保 'tensor_function.py' 和 'tensor_function_MTLPP.py' 在当前工作目录中。")
    sys.exit(1)

class TRPCA:
    def __init__(self):
        # 强制使用 CPU 运行（Batch 优化后 CPU 速度已足够快且稳定）
        self.device = torch.device("cpu")

    def T_product(self, A, B):
        """
        优化的 t-product: 利用 FFT + Batch 矩阵乘法
        """
        A_f = torch.fft.fft(A, dim=2)
        B_f = torch.fft.fft(B, dim=2)

        # 调整维度以适配 PyTorch 的 batch matmul: (Batch, n1, n2)
        A_f_batch = A_f.permute(2, 0, 1) # (n3, n1, n2)
        B_f_batch = B_f.permute(2, 0, 1) # (n3, n2, n4)

        # Batch 矩阵乘法
        C_f_batch = torch.matmul(A_f_batch, B_f_batch) 

        C_f = C_f_batch.permute(1, 2, 0)
        return torch.fft.ifft(C_f, dim=2).real

    def SoftShrink(self, X, tau):
        """软阈值算子"""
        return torch.sgn(X) * torch.relu(torch.abs(X) - tau)

    def SVDShrink(self, X, tau):
        """
        优化的张量 SVD 阈值算子 (Batch SVD)
        """
        n1, n2, n3 = X.shape
        X_f = torch.fft.fft(X, dim=2)
        
        # 调整为 Batch 模式 (n3, n1, n2)
        X_f_batch = X_f.permute(2, 0, 1)

        # Batch SVD
        U, S, Vh = torch.linalg.svd(X_f_batch, full_matrices=False)

        # 奇异值收缩
        S_new = self.SoftShrink(S, tau)

        # 重构 (利用广播机制)
        W_batch = torch.matmul(U, S_new.unsqueeze(-1) * Vh) 

        W_f = W_batch.permute(1, 2, 0)
        return torch.fft.ifft(W_f, dim=2).real

    def block_diagonal_fft(self, X):
        """
        仅用于兼容旧接口 (本优化版本中主要流程已不再依赖此函数)
        """
        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X, dim=2)
        res = torch.zeros((n1 * n3, n3), dtype=torch.complex64)
        for i in range(n3):
            res[i*n1 : (i+1)*n1, i : i+1] = Xf[:, :, i]
        return res

    def getyi_yj(self, X_train_tensor_list, W, D):
        """
        原始的低效构图函数 (保留以备不时之需，但主流程中已被优化代码替代)
        """
        print("警告：正在调用低效的 getyi_yj 函数，这可能会非常慢。建议使用矩阵优化版。")
        l = len(X_train_tensor_list)
        sample_0 = X_train_tensor_list[0]
        if torch.is_tensor(sample_0): sample_0 = sample_0.numpy()
        dim = sample_0.shape[0]
        
        try:
            re = np.zeros((dim, dim), dtype=np.complex64)
            re1 = np.zeros((dim, dim), dtype=np.complex64)
        except MemoryError:
            print("错误：内存不足。")
            sys.exit(1)

        for i in range(l):
            yi = X_train_tensor_list[i]
            if torch.is_tensor(yi): yi = yi.numpy()
            
            term = np.dot(yi, yi.conj().T)
            re1 += term * D[i][i]
            
            for j in range(i + 1, l):
                w_val = W[i][j]
                if w_val != 0:
                    yj = X_train_tensor_list[j]
                    if torch.is_tensor(yj): yj = yj.numpy()
                    
                    diff = yi - yj
                    term_diff = np.dot(diff, diff.conj().T)
                    re += term_diff * w_val * 2
        
        return re, re1

    def ADMM(self, left_mat, right_mat, X, Y):
        """
        全流程 Batch 并行优化的 ADMM 算法
        """
        # 1. 维度与设备准备
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        l, m, n = X.shape  

        # === 核心修改：智能识别输入格式 ===
        # 原矩阵大小 (d*n, d*n) 或 Batch (n, d, d)
        def big_to_batch(big_mat, n_freq):
            if not torch.is_tensor(big_mat):
                big_mat = torch.from_numpy(big_mat).to(self.device).to(torch.complex64)
            dim_big = big_mat.shape[0]
            dim_small = dim_big // n_freq
            batch_mat = torch.zeros((n_freq, dim_small, dim_small), dtype=torch.complex64, device=self.device)
            for i in range(n_freq):
                batch_mat[i] = big_mat[i*dim_small:(i+1)*dim_small, i*dim_small:(i+1)*dim_small]
            return batch_mat

        # 如果输入已经是 3D (n, d, d)，直接使用；否则转换
        if hasattr(left_mat, 'ndim') and left_mat.ndim == 3:
            print("检测到优化后的 Batch 图矩阵，直接使用。")
            left_batch = left_mat.to(self.device)
        else:
            print("检测到传统大矩阵，正在转换为 Batch 模式...")
            left_batch = big_to_batch(left_mat, n)

        if hasattr(right_mat, 'ndim') and right_mat.ndim == 3:
            right_batch = right_mat.to(self.device)
        else:
            right_batch = big_to_batch(right_mat, n)

        c = Y.shape[0]
        r = 30 # 降维后的维度
        rho = 1.1
        mu = 1e-2
        mu_max = 1e10
        max_iters = 100
        lamb = 1
        beita = 1
        gama = 1 

        # 初始化变量
        L = torch.zeros((l, m, n), device=self.device)
        M = torch.zeros((l, m, n), device=self.device)
        E = torch.zeros((r, m, n), device=self.device)
        P = torch.zeros((r, l, n), device=self.device)
        Q = torch.zeros((r, l, n), device=self.device)
        W = torch.zeros((c, r, n), device=self.device)
        G = torch.zeros((c, l, n), device=self.device)
        
        W1 = torch.zeros((r, m, n), device=self.device)
        W2 = torch.zeros((l, m, n), device=self.device)
        W3 = torch.zeros((c, l, n), device=self.device)
        W4 = torch.zeros((r, l, n), device=self.device)

        I_batch = torch.eye(l, device=self.device).unsqueeze(0).expand(n, l, l).to(torch.complex64)

        print(f"ADMM 开始优化: 输入 X={X.shape}, 类别={c}")

        for iters in range(max_iters):
            # 1. Update M
            M_new = self.SVDShrink(L + W2/mu, 1/mu)

            # 2. Update L
            X_f = torch.fft.fft(X, dim=2).permute(2,0,1)
            P_f = torch.fft.fft(P, dim=2).permute(2,0,1)
            M_f = torch.fft.fft(M_new, dim=2).permute(2,0,1)
            E_f = torch.fft.fft(E, dim=2).permute(2,0,1)
            W1_f = torch.fft.fft(W1, dim=2).permute(2,0,1)
            W2_f = torch.fft.fft(W2, dim=2).permute(2,0,1)
            
            Ph = P_f.conj().transpose(1, 2)
            Term1 = torch.matmul(Ph, torch.matmul(P_f, X_f) - E_f + W1_f/mu)
            Term2 = M_f - W2_f/mu
            LHS_inv = torch.linalg.inv(torch.matmul(Ph, P_f) + I_batch)
            L_new_f = torch.matmul(LHS_inv, Term1 + Term2)
            L_new = torch.fft.ifft(L_new_f.permute(1,2,0), dim=2).real

            # 3. Update E
            PX = self.T_product(P, X)
            PL = self.T_product(P, L_new)
            E_new = self.SoftShrink(PX - PL + W1/mu, lamb/mu)

            # 4. Update W, G
            G_f = torch.fft.fft(G, dim=2).permute(2,0,1)
            W3_f = torch.fft.fft(W3, dim=2).permute(2,0,1)
            P_f = torch.fft.fft(P, dim=2).permute(2,0,1)
            
            Target_W = torch.matmul(G_f - W3_f/mu, P_f.conj().transpose(1, 2))
            Uw, _, Vhw = torch.linalg.svd(Target_W, full_matrices=False)
            W_new_f = torch.matmul(Uw, Vhw)
            W_new = torch.fft.ifft(W_new_f.permute(1,2,0), dim=2).real
            
            Y_f = torch.fft.fft(Y, dim=2).permute(2,0,1)
            Xh = X_f.conj().transpose(1, 2)
            W_new_f = torch.fft.fft(W_new, dim=2).permute(2,0,1)
            Target_G = beita * torch.matmul(Y_f, Xh) + (mu/2)*(torch.matmul(W_new_f, P_f) + W3_f/mu)
            Ug, _, Vhg = torch.linalg.svd(Target_G, full_matrices=False)
            G_new = torch.fft.ifft(torch.matmul(Ug, Vhg).permute(1,2,0), dim=2).real

            # 5. Update P
            Eb_f = torch.fft.fft(E_new, dim=2).permute(2,0,1)
            Lb_f = torch.fft.fft(L_new, dim=2).permute(2,0,1)
            Qb_f = torch.fft.fft(Q, dim=2).permute(2,0,1)
            Gb_f = torch.fft.fft(G_new, dim=2).permute(2,0,1)
            W4_f = torch.fft.fft(W4, dim=2).permute(2,0,1)
            
            Diff_XL = X_f - Lb_f
            M1 = torch.matmul(Eb_f - W1_f/mu, Diff_XL.conj().transpose(1,2))
            Wh_f = W_new_f.conj().transpose(1, 2)
            M2 = torch.matmul(Wh_f, Gb_f - W3_f/mu)
            M3 = Qb_f - W4_f/mu
            M_total = M1 + M2 + M3
            Up, _, Vhp = torch.linalg.svd(M_total, full_matrices=False)
            P_new_f = torch.matmul(Up, Vhp)
            P_new = torch.fft.ifft(P_new_f.permute(1,2,0), dim=2).real

            # 6. Update Q
            Pb_f = P_new_f 
            W4b_f = W4_f
            Q_curr = torch.fft.fft(Q, dim=2).permute(2,0,1)
            
            q_lr = 0.01
            q_iters = 5
            lambda_val = 1.0
            
            for _ in range(q_iters):
                QL = torch.matmul(Q_curr, left_batch)
                Diff = Pb_f - Q_curr + W4b_f/mu
                QD = torch.matmul(Q_curr, right_batch)
                
                Grad = 2 * gama * QL - mu * Diff + 2 * lambda_val * QD
                Q_curr = Q_curr - q_lr * Grad
                
                QDQ = torch.matmul(Q_curr, torch.matmul(right_batch, Q_curr.conj().transpose(1,2)))
                tr_val = torch.real(torch.vmap(torch.trace)(QDQ)).sum()
                grad_lambda = tr_val - 1
                lambda_val = lambda_val + q_lr * grad_lambda
            
            Q_new = torch.fft.ifft(Q_curr.permute(1,2,0), dim=2).real

            # 7. Update Multipliers
            term1 = self.T_product(P_new, X) - self.T_product(P_new, L_new) - E_new
            W1 = W1 + mu * term1
            W2 = W2 + mu * (L_new - M_new)
            term3 = self.T_product(W_new, P_new) - G_new
            W3 = W3 + mu * term3
            term4 = P_new - Q_new
            W4 = W4 + mu * term4
            mu = min(rho * mu, mu_max)

            M, L, E, P, W, G, Q = M_new, L_new, E_new, P_new, W_new, G_new, Q_new

            if iters % 10 == 0:
                err = torch.norm(term1)
                print(f"Iter {iters:3d}: P*X Recon Err={err:.4f}, mu={mu:.2e}")

        return M_new, L_new, E_new, P_new, W_new, G_new, Q

def nn_unique(x_train, train_label, x_test, test_label):
    N_train = x_train.shape[1]
    N_test = x_test.shape[1]
    xt_flat = x_train.permute(1, 0, 2).reshape(N_train, -1).detach().cpu().numpy()
    xe_flat = x_test.permute(1, 0, 2).reshape(N_test, -1).detach().cpu().numpy()
    
    from scipy.spatial.distance import cdist
    D = cdist(xe_flat, xt_flat, metric='euclidean')
    id = np.argsort(D, axis=1)
    pred_labels = np.array(train_label)[id[:, 0]]

    test_label = np.array(test_label)
    total_correct = np.sum(pred_labels == test_label)
    precision_OA = total_correct / N_test

    unique_classes = np.unique(train_label)
    class_precision = {}
    for cls in unique_classes:
        indices = np.where(test_label == cls)[0]
        if len(indices) == 0: continue
        correct = np.sum(pred_labels[indices] == cls)
        class_precision[cls] = correct / len(indices)

    precision_AA = np.mean(list(class_precision.values()))
    
    confusion = np.zeros((len(unique_classes)+1, len(unique_classes)+1))
    lbl_map = {c: i for i, c in enumerate(unique_classes)}
    for t, p in zip(test_label, pred_labels):
        if t in lbl_map and p in lbl_map:
            confusion[lbl_map[t], lbl_map[p]] += 1
    total = np.sum(confusion)
    po = np.trace(confusion) / total
    pe = np.sum(np.sum(confusion, 0) * np.sum(confusion, 1)) / (total**2)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0

    return {'OA': precision_OA, 'AA': precision_AA, 'Kappa': kappa}

if __name__ == '__main__':
    # ================= 配置区 =================
    data_path = '/data/LOH/TSPLL_label/dataset/Indian_pines/' 
    mat_file = 'Indian_pines.mat'
    gt_file = 'Indian_pines_gt.mat'
    key_img = 'indian_pines_corrected' 
    key_gt = 'indian_pines_gt'
    idx_file_path = '/data/LOH/TSPLL_label/random_idx/random_idx.npy'
    
    PATCH_SIZE = 9
    w_sq = PATCH_SIZE ** 2
    # =========================================

    print(f"Loading data from {data_path}...")
    try:
        import os
        img_path = os.path.join(data_path, mat_file)
        gt_path = os.path.join(data_path, gt_file)
        image = loadmat(img_path)[key_img]
        label = loadmat(gt_path)[key_gt]
        print(f"Data loaded successfully. Shape: {image.shape}")
    except Exception as e:
        print(f"Error loading data: {e}. Using random data.")
        image = np.random.rand(145, 145, 200)
        label = np.random.randint(0, 17, (145, 145))

    # 1. 归一化
    image = image.astype(float)
    for band in range(image.shape[2]):
        mi = np.min(image[:, :, band])
        ma = np.max(image[:, :, band])
        if ma != mi:
            image[:, :, band] = (image[:, :, band] - mi) / (ma - mi)

    # 2. 添加噪声
    np.random.seed(42)
    noisy_image = np.copy(image)
    selected_bands = np.random.choice(image.shape[2], size=min(30, image.shape[2]), replace=False)
    for i in selected_bands:
        variance = np.random.uniform(0, 0.5)
        noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
        noisy_image[..., i] += noise
        salt = np.random.rand(*image[..., i].shape) < 0.05
        pepper = np.random.rand(*image[..., i].shape) < 0.05
        noisy_image[salt, i] = np.max(image[..., i])
        noisy_image[pepper, i] = np.min(image[..., i])
    image = noisy_image 

    # 3. 数据集划分
    print("Preparing Datasets...")
    try:
        random_idx = np.load(idx_file_path, allow_pickle=True)
        if isinstance(random_idx, np.ndarray) and random_idx.dtype == object: random_idx = random_idx.item()
        if isinstance(random_idx, dict):
            final_idx = []
            for k, v in random_idx.items():
                if isinstance(v, dict) and 'train_indices' in v: final_idx.extend(v['train_indices'])
                elif isinstance(v, (list, np.ndarray)): final_idx.extend(v)
            random_idx = final_idx
        
        ret = train_test_tensor_fold(PATCH_SIZE, random_idx, image, label)
        x_train, train_label_tensor, x_test, test_label_tensor, train_label_list, test_label_list = ret
    except Exception as e:
        print(f"Dataset failed: {e}")
        sys.exit(1)

    ours = TRPCA()
    N_train = x_train.shape[1]

    # 4. 构图 (EMD) - 这一步是物理时间开销，不可避免
    print("Constructing Graph (Using EMD)...")
    try:
        ret_half = train_test_tensor_half(PATCH_SIZE, random_idx, image, label)
        x_train_W = ret_half[0]
        ci = []
        b_dim = x_train_W[0].shape[2]
        for i in range(len(x_train_W)):
            c_matrix = torch.zeros((b_dim, b_dim))
            xt = x_train_W[i]
            if not torch.is_tensor(xt): xt = torch.from_numpy(xt).float()
            ui = torch.mean(xt, dim=(0, 1), keepdim=True)
            ui1 = ui.reshape(b_dim, -1)
            for m in range(PATCH_SIZE):
                for n in range(PATCH_SIZE):
                    xt1 = xt[m, n, :].reshape(b_dim, -1)
                    diff = xt1 - ui1
                    c_matrix = c_matrix + torch.matmul(diff, diff.T)
            c_matrix = c_matrix / (PATCH_SIZE * PATCH_SIZE - 1)
            ci.append(c_matrix)

        w_mat, d_mat = dist_EMD(x_train_W, ci, 10, 1000)
        print("EMD Graph constructed.")
    except Exception as e:
        print(f"Graph failed: {e}. Using Identity.")
        w_mat = np.eye(N_train)
        d_mat = np.eye(N_train)

    # 5. 【矩阵运算极速优化】计算 Laplacian Matrices
    print("Calculating Laplacian Matrices (Vectorized)...")
    
    # 转换到频域并调整维度: (w^2, d, N)
    x_train_f = torch.fft.fft(x_train, dim=2).permute(2, 0, 1).to(torch.complex64)
    
    # 准备图矩阵: (1, N, N) 用于广播
    W_tensor = torch.from_numpy(w_mat).float().to(ours.device).unsqueeze(0)
    D_tensor = torch.from_numpy(d_mat).float().to(ours.device).unsqueeze(0)
    L_tensor = D_tensor - W_tensor
    
    # Batch Matrix Multiplication
    # Right = X * D * X^H
    XD = torch.matmul(x_train_f, D_tensor.type_as(x_train_f)) 
    Right_batch = torch.matmul(XD, x_train_f.conj().transpose(1, 2))
    
    # Left = 2 * X * L * X^H
    XL = torch.matmul(x_train_f, L_tensor.type_as(x_train_f))
    Left_batch = 2 * torch.matmul(XL, x_train_f.conj().transpose(1, 2))
    
    print("Laplacian calculation finished.")

    # 准备标签
    num_classes = 16
    if len(train_label_list) > 0: max_val = int(np.max(train_label_list)); num_classes = max(num_classes, max_val)
    Y_tensor = torch.zeros(num_classes, N_train, w_sq)
    for i in range(N_train):
        l_idx = int(train_label_list[i]) - 1 
        if 0 <= l_idx < num_classes: Y_tensor[l_idx, i, :] = 1

    # 6. ADMM 优化 (直接传入优化后的 Batch Tensor)
    print(">>> Starting ADMM Optimization <<<")
    M, L, E, P, W, G, Q = ours.ADMM(Left_batch, Right_batch, x_train, Y_tensor)
    print("Optimization Finished.")
    
    # 7. 评估
    print("Evaluating...")
    X_train_red = ours.T_product(P, x_train)
    X_test_red = ours.T_product(P, x_test)
    X_train_red_Q = ours.T_product(Q, x_train)
    X_test_red_Q = ours.T_product(Q, x_test)
    
    print("--- P * X Results ---")
    res_px = nn_unique(X_train_red, train_label_list, X_test_red, test_label_list)
    print(f"OA: {res_px['OA']*100:.2f}%  AA: {res_px['AA']*100:.2f}%  Kappa: {res_px['Kappa']*100:.4f}")
    
    print("--- Q * X Results ---")
    res_qx = nn_unique(X_train_red_Q, train_label_list, X_test_red_Q, test_label_list)
    print(f"OA: {res_qx['OA']*100:.2f}%  AA: {res_qx['AA']*100:.2f}%  Kappa: {res_qx['Kappa']*100:.4f}")