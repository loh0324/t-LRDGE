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
        A: (n1, n2, n3)
        B: (n2, n4, n3)
        Return: (n1, n4, n3)
        """
        # 1. FFT 变换到频域
        A_f = torch.fft.fft(A, dim=2)
        B_f = torch.fft.fft(B, dim=2)

        # 2. 调整维度以适配 PyTorch 的 batch matmul: (Batch, n1, n2)
        # 将频域维 n3 移到第一维
        A_f_batch = A_f.permute(2, 0, 1) # (n3, n1, n2)
        B_f_batch = B_f.permute(2, 0, 1) # (n3, n2, n4)

        # 3. Batch 矩阵乘法 (并行计算 n3 个矩阵乘法)
        C_f_batch = torch.matmul(A_f_batch, B_f_batch) # (n3, n1, n4)

        # 4. 还原维度并做 IFFT
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
        # 1. FFT
        X_f = torch.fft.fft(X, dim=2)
        
        # 2. 调整为 Batch 模式 (n3, n1, n2)
        X_f_batch = X_f.permute(2, 0, 1)

        # 3. Batch SVD
        # U: (n3, n1, k), S: (n3, k), Vh: (n3, k, n2)
        U, S, Vh = torch.linalg.svd(X_f_batch, full_matrices=False)

        # 4. 奇异值收缩
        S_new = self.SoftShrink(S, tau)

        # 5. 重构 (利用广播机制构建对角阵乘法)
        # S_new.unsqueeze(-1) 变为 (n3, k, 1)，利用广播实现对角矩阵乘法
        W_batch = torch.matmul(U, S_new.unsqueeze(-1) * Vh) 

        # 6. IFFT
        W_f = W_batch.permute(1, 2, 0)
        return torch.fft.ifft(W_f, dim=2).real

    def block_diagonal_fft(self, X):
        """
        仅用于构图时的兼容函数 (生成用于计算权重的中间大矩阵)
        X: (d, 1, w^2) -> (d*w^2, w^2) 块对角阵
        """
        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X, dim=2)
        res = torch.zeros((n1 * n3, n3), dtype=torch.complex64)
        for i in range(n3):
            res[i*n1 : (i+1)*n1, i : i+1] = Xf[:, :, i]
        return res

    def getyi_yj(self, X_train_tensor_list, W, D):
        """
        图拉普拉斯矩阵构建
        X_train_tensor_list: 列表，每个元素是单个样本的大矩阵 (from block_diagonal_fft)
        """
        l = len(X_train_tensor_list)
        # 获取维度信息，确保兼容 tensor 和 numpy
        sample_0 = X_train_tensor_list[0]
        if torch.is_tensor(sample_0):
            sample_0 = sample_0.numpy()
        
        dim = sample_0.shape[0]
        
        # 预分配大矩阵
        try:
            re = np.zeros((dim, dim), dtype=np.complex64)
            re1 = np.zeros((dim, dim), dtype=np.complex64)
        except MemoryError:
            print("错误：内存不足，无法构建图矩阵。请尝试减小数据量。")
            sys.exit(1)

        for i in range(l):
            yi = X_train_tensor_list[i]
            if torch.is_tensor(yi): yi = yi.numpy()
            
            # Term 1: Yi * Yi^H * D[i,i]
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
        
        l, m, n = X.shape  # l=d(特征维度), m=N(样本数), n=w^2(频率/窗口大小)

        # === 核心优化：将输入的大图矩阵转换为 Batch 形式 ===
        # 原矩阵大小 (d*n, d*n)，转换为 (n, d, d) 以便并行计算
        def big_to_batch(big_mat, n_freq):
            if not torch.is_tensor(big_mat):
                big_mat = torch.from_numpy(big_mat).to(self.device).to(torch.complex64)
            
            dim_big = big_mat.shape[0]
            dim_small = dim_big // n_freq
            batch_mat = torch.zeros((n_freq, dim_small, dim_small), dtype=torch.complex64, device=self.device)
            # 提取对角块
            for i in range(n_freq):
                batch_mat[i] = big_mat[i*dim_small:(i+1)*dim_small, i*dim_small:(i+1)*dim_small]
            return batch_mat

        print("正在将图矩阵转换为 Batch 模式...")
        left_batch = big_to_batch(left_mat, n)
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
        # 初始化 P 为近似正交 (可选)
        # P[:, :, 0] = torch.eye(r, l) 
        Q = torch.zeros((r, l, n), device=self.device)
        W = torch.zeros((c, r, n), device=self.device)
        G = torch.zeros((c, l, n), device=self.device)
        
        W1 = torch.zeros((r, m, n), device=self.device)
        W2 = torch.zeros((l, m, n), device=self.device)
        W3 = torch.zeros((c, l, n), device=self.device)
        W4 = torch.zeros((r, l, n), device=self.device)

        # 预计算单位阵 (Batch 模式)
        I_batch = torch.eye(l, device=self.device).unsqueeze(0).expand(n, l, l).to(torch.complex64)

        print(f"ADMM 开始优化: 输入形状 X={X.shape}, 类别数={c}")

        for iters in range(max_iters):
            # -----------------------------------------------------------------
            # 1. 更新 M (t-SVD)
            # -----------------------------------------------------------------
            M_new = self.SVDShrink(L + W2/mu, 1/mu)

            # -----------------------------------------------------------------
            # 2. 更新 L (最小二乘，Batch 频域求解)
            # -----------------------------------------------------------------
            # 全部转为频域 Batch 格式: (n, d1, d2)
            X_f = torch.fft.fft(X, dim=2).permute(2,0,1)
            P_f = torch.fft.fft(P, dim=2).permute(2,0,1)
            M_f = torch.fft.fft(M_new, dim=2).permute(2,0,1)
            E_f = torch.fft.fft(E, dim=2).permute(2,0,1)
            W1_f = torch.fft.fft(W1, dim=2).permute(2,0,1)
            W2_f = torch.fft.fft(W2, dim=2).permute(2,0,1)
            
            Ph = P_f.conj().transpose(1, 2)
            
            # 计算 RHS
            Term1 = torch.matmul(Ph, torch.matmul(P_f, X_f) - E_f + W1_f/mu)
            Term2 = M_f - W2_f/mu
            
            # 求解线性方程 (P^H P + I) L = RHS
            # 注意: P^H P 并不完全是 I (在迭代过程中), 所以显式求逆更稳健
            LHS_inv = torch.linalg.inv(torch.matmul(Ph, P_f) + I_batch)
            L_new_f = torch.matmul(LHS_inv, Term1 + Term2)
            
            # IFFT 回到时域
            L_new = torch.fft.ifft(L_new_f.permute(1,2,0), dim=2).real

            # -----------------------------------------------------------------
            # 3. 更新 E (软阈值)
            # -----------------------------------------------------------------
            PX = self.T_product(P, X)
            PL = self.T_product(P, L_new)
            E_new = self.SoftShrink(PX - PL + W1/mu, lamb/mu)

            # -----------------------------------------------------------------
            # 4. 更新 W (判别投影 H) 和 G
            # -----------------------------------------------------------------
            G_f = torch.fft.fft(G, dim=2).permute(2,0,1)
            W3_f = torch.fft.fft(W3, dim=2).permute(2,0,1)
            P_f = torch.fft.fft(P, dim=2).permute(2,0,1) # 重新获取 P_f (可能未变)
            
            # 更新 W: Procrustes 问题 max Tr(W^T M) -> W = UV^T
            # Target = (G - W3/mu) P^H
            Target_W = torch.matmul(G_f - W3_f/mu, P_f.conj().transpose(1, 2))
            Uw, _, Vhw = torch.linalg.svd(Target_W, full_matrices=False)
            W_new_f = torch.matmul(Uw, Vhw)
            W_new = torch.fft.ifft(W_new_f.permute(1,2,0), dim=2).real
            
            # 更新 G:
            Y_f = torch.fft.fft(Y, dim=2).permute(2,0,1)
            Xh = X_f.conj().transpose(1, 2)
            # Target = beta * Y X^H + mu/2 * (W P + W3/mu)
            Target_G = beita * torch.matmul(Y_f, Xh) + (mu/2)*(torch.matmul(W_new_f, P_f) + W3_f/mu)
            Ug, _, Vhg = torch.linalg.svd(Target_G, full_matrices=False)
            G_new = torch.fft.ifft(torch.matmul(Ug, Vhg).permute(1,2,0), dim=2).real

            # -----------------------------------------------------------------
            # 5. 更新 P (Batch Procrustes)
            # -----------------------------------------------------------------
            # 准备新的变量 (Batch 频域)
            Eb_f = torch.fft.fft(E_new, dim=2).permute(2,0,1)
            Lb_f = torch.fft.fft(L_new, dim=2).permute(2,0,1) # L_new_f already computed but just to be safe
            Qb_f = torch.fft.fft(Q, dim=2).permute(2,0,1)
            Gb_f = torch.fft.fft(G_new, dim=2).permute(2,0,1)
            W4_f = torch.fft.fft(W4, dim=2).permute(2,0,1)
            
            # M1 = (E - W1/mu) * (X - L)^H
            Diff_XL = X_f - Lb_f
            M1 = torch.matmul(Eb_f - W1_f/mu, Diff_XL.conj().transpose(1,2))
            
            # M2 = W^H * (G - W3/mu)
            Wh_f = W_new_f.conj().transpose(1, 2)
            M2 = torch.matmul(Wh_f, Gb_f - W3_f/mu)
            
            # M3 = Q - W4/mu
            M3 = Qb_f - W4_f/mu
            
            # 汇总求解
            M_total = M1 + M2 + M3
            Up, _, Vhp = torch.linalg.svd(M_total, full_matrices=False)
            P_new_f = torch.matmul(Up, Vhp)
            P_new = torch.fft.ifft(P_new_f.permute(1,2,0), dim=2).real

            # -----------------------------------------------------------------
            # 6. 更新 Q (Batch 梯度下降求解流形正则项)
            # -----------------------------------------------------------------
            Pb_f = P_new_f 
            W4b_f = W4_f
            Q_curr = torch.fft.fft(Q, dim=2).permute(2,0,1) # 初始值
            
            q_lr = 0.01
            q_iters = 5 # 内部迭代次数
            lambda_val = 1.0 # 约束乘子
            
            # 在频域并行更新所有切片
            for _ in range(q_iters):
                # 计算梯度 (Batch)
                # QL = Q * Left
                QL = torch.matmul(Q_curr, left_batch)
                # Diff = P - Q + ...
                Diff = Pb_f - Q_curr + W4b_f/mu
                # QD = Q * Right
                QD = torch.matmul(Q_curr, right_batch)
                
                # Grad = 2*gama*Q*L - mu*Diff + 2*lambda*Q*D
                Grad = 2 * gama * QL - mu * Diff + 2 * lambda_val * QD
                
                # 更新 Q
                Q_curr = Q_curr - q_lr * Grad
                
                # 更新 lambda (满足约束 Trace(Q D Q^H) = 1)
                QDQ = torch.matmul(Q_curr, torch.matmul(right_batch, Q_curr.conj().transpose(1,2)))
                # 计算 trace (sum of diag) 并求实部
                tr_val = torch.real(torch.vmap(torch.trace)(QDQ)).sum()
                
                # 假设归一化目标为样本数或其他常数，这里设为1
                grad_lambda = tr_val - 1
                lambda_val = lambda_val + q_lr * grad_lambda
            
            Q_new = torch.fft.ifft(Q_curr.permute(1,2,0), dim=2).real

            # -----------------------------------------------------------------
            # 7. 更新乘子
            # -----------------------------------------------------------------
            term1 = self.T_product(P_new, X) - self.T_product(P_new, L_new) - E_new
            W1 = W1 + mu * term1
            
            W2 = W2 + mu * (L_new - M_new)
            
            term3 = self.T_product(W_new, P_new) - G_new
            W3 = W3 + mu * term3
            
            term4 = P_new - Q_new
            W4 = W4 + mu * term4
            
            mu = min(rho * mu, mu_max)

            # 更新迭代变量
            M, L, E, P, W, G, Q = M_new, L_new, E_new, P_new, W_new, G_new, Q_new

            # 简易收敛检查
            if iters % 10 == 0:
                err = torch.norm(term1)
                print(f"Iter {iters:3d}: P*X Recon Err={err:.4f}, mu={mu:.2e}")

        return M_new, L_new, E_new, P_new, W_new, G_new, Q

def nn_unique(x_train, train_label, x_test, test_label):
    '''
    简易 KNN 分类评估
    '''
    # 展平数据用于计算距离: (d, N, w^2) -> (N, d*w^2)
    N_train = x_train.shape[1]
    N_test = x_test.shape[1]
    
    xt_flat = x_train.permute(1, 0, 2).reshape(N_train, -1).detach().cpu().numpy()
    xe_flat = x_test.permute(1, 0, 2).reshape(N_test, -1).detach().cpu().numpy()
    
    # 距离矩阵计算
    from scipy.spatial.distance import cdist
    D = cdist(xe_flat, xt_flat, metric='euclidean')

    # 最近邻索引
    id = np.argsort(D, axis=1)
    # 预测标签
    pred_labels = np.array(train_label)[id[:, 0]]

    # 精度计算
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

    # Kappa 计算
    confusion = np.zeros((len(unique_classes)+1, len(unique_classes)+1)) # +1 防止索引越界
    # 建立标签映射
    lbl_map = {c: i for i, c in enumerate(unique_classes)}
    
    for t, p in zip(test_label, pred_labels):
        if t in lbl_map and p in lbl_map:
            confusion[lbl_map[t], lbl_map[p]] += 1
    
    total = np.sum(confusion)
    po = np.trace(confusion) / total
    pe = np.sum(np.sum(confusion, 0) * np.sum(confusion, 1)) / (total**2)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0

    return {
        'OA': precision_OA,
        'AA': precision_AA,
        'Kappa': kappa
    }

if __name__ == '__main__':
    # ================= 配置区 (Indian Pines) =================
    # 请确认数据文件在以下路径
    data_path = '/data/LOH/TSPLL_label/dataset/Indian_pines/' 
    mat_file = 'Indian_pines.mat'
    gt_file = 'Indian_pines_gt.mat'
    key_img = 'indian_pines_corrected' 
    key_gt = 'indian_pines_gt'
    
    # 索引文件路径 (Indian Pines 对应的索引)
    idx_file_path = '/data/LOH/TSPLL_label/random_idx/random_idx.npy'
    
    PATCH_SIZE = 9
    w_sq = PATCH_SIZE ** 2
    # =========================================================

    print(f"Loading data from {data_path}...")
    try:
        import os
        img_path = os.path.join(data_path, mat_file)
        gt_path = os.path.join(data_path, gt_file)
        
        image = loadmat(img_path)[key_img]
        label = loadmat(gt_path)[key_gt]
        print(f"Data loaded successfully. Shape: {image.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # 仅供测试的兜底数据
        image = np.random.rand(145, 145, 200)
        label = np.random.randint(0, 17, (145, 145))

    # 1. 归一化
    image = image.astype(float)
    for band in range(image.shape[2]):
        mi = np.min(image[:, :, band])
        ma = np.max(image[:, :, band])
        if ma != mi:
            image[:, :, band] = (image[:, :, band] - mi) / (ma - mi)

    # 2. 添加噪声 (保持原逻辑)
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

    # 3. 数据集划分 (恢复为您要求的原始逻辑)
    print("Preparing Datasets via train_test_tensor_fold...")
    try:
        # 加载索引文件 (加上 allow_pickle=True 以防报错)
        print(f"Loading index from: {idx_file_path}")
        random_idx = np.load(idx_file_path, allow_pickle=True)
        
        # 处理可能的字典包裹情况 (如果 .npy 保存的是 item/dict)
        if isinstance(random_idx, np.ndarray) and random_idx.dtype == object and random_idx.ndim == 0:
            random_idx = random_idx.item()
        
        # 如果是字典，展平成列表
        if isinstance(random_idx, dict):
            final_idx = []
            for k, v in random_idx.items():
                if isinstance(v, dict) and 'train_indices' in v:
                    final_idx.extend(v['train_indices'])
                elif isinstance(v, (list, np.ndarray)):
                    final_idx.extend(v)
            random_idx = final_idx
            
        # 调用 fold 函数
        # 返回值: X_train, train_label, X_test, test_label, train_label_list, test_label_list
        ret = train_test_tensor_fold(PATCH_SIZE, random_idx, image, label)
        x_train, train_label_tensor, x_test, test_label_tensor, train_label_list, test_label_list = ret
        
        print(f"数据集划分完成。训练样本数: {x_train.shape[1]}, 测试样本数: {x_test.shape[1]}")

    except Exception as e:
        print(f"!!! 数据集划分失败 !!! 错误: {e}")
        # 遇到错误时停止，方便您排查路径问题
        sys.exit(1)

    # 初始化模型
    ours = TRPCA()

    # 4. 构图 (计算 W, D)
    print("Constructing Graph (Using EMD)...")
    try:
        # 获取用于 EMD 的数据 (半张量)
        ret_half = train_test_tensor_half(PATCH_SIZE, random_idx, image, label)
        x_train_W = ret_half[0]
        
        # 计算协方差
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

        # 计算 EMD 距离
        # k_near = 10, t = 1000 (参考您的原始代码参数)
        w_mat, d_mat = dist_EMD(x_train_W, ci, 10, 1000)
        print("EMD Graph constructed.")
        
    except Exception as e:
        print(f"Graph construction failed: {e}. Using Identity matrix.")
        N_train = x_train.shape[1]
        w_mat = np.eye(N_train)
        d_mat = np.eye(N_train)

    # 准备频域大矩阵列表
    N_train = x_train.shape[1]
    X_train_fft_list = []
    for i in range(N_train):
        samp = x_train[:, i:i+1, :]
        samp_bd = ours.block_diagonal_fft(samp).numpy()
        X_train_fft_list.append(samp_bd)

    print("Calculating Laplacian Matrices...")
    left, right = ours.getyi_yj(X_train_fft_list, w_mat, d_mat)
    left = torch.from_numpy(left)
    right = torch.from_numpy(right)

    # 构造 Y (One-hot)
    # Indian Pines 16 类
    num_classes = 16
    if len(train_label_list) > 0:
        max_val = int(np.max(train_label_list))
        if max_val > num_classes: num_classes = max_val
            
    Y_tensor = torch.zeros(num_classes, N_train, w_sq)
    for i in range(N_train):
        l_idx = int(train_label_list[i]) - 1 
        if 0 <= l_idx < num_classes:
            Y_tensor[l_idx, i, :] = 1

    # 5. ADMM 优化
    print(">>> Starting ADMM Optimization <<<")
    M, L, E, P, W, G, Q = ours.ADMM(left, right, x_train, Y_tensor)
    print(">>> Finished <<<")
    
    # 6. 评估
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