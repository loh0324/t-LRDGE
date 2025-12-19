import math
import sys
import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from sklearn.decomposition import PCA
import torch
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
import spectral as spy

# 保持引用不变，确保这些文件在同级目录
try:
    from tensor_function import Patch, getU, kmode_product, train_test_tensor, train_test_zishiying, \
        train_test_tensor_half, train_test_tensor_fold
    from tensor_function_MTLPP import dist_EMD
except ImportError:
    pass # 允许在缺少依赖时仅做代码检查

class TRPCA:
    def __init__(self):
        # 强制使用 CPU，保证稳定性
        self.device = torch.device("cpu")
        print(f"Running on device: {self.device}")

    def converged(self, M, L, E, P, W, G, X, M_new, L_new, E_new, P_new, W_new, G_new):
        '''判断收敛条件'''
        eps = 1e-4
        condition2 = torch.max(torch.abs(L_new - L)) < eps
        condition4 = torch.max(torch.abs(P_new - P)) < eps
        return condition2 and condition4

    def SoftShrink(self, X, tau):
        if not torch.is_tensor(tau):
            tau = torch.tensor(tau).to(X.device)
        # 修复复数警告：使用 abs 计算 magnitude
        return torch.sign(X) * torch.maximum(torch.abs(X) - tau, torch.zeros_like(X))

    def SVDShrink(self, X, tau):
        """
        修复版 SVD 收缩算子：
        1. 修复 UnboundLocalError (mat 定义)
        2. 修复 TypeError (移除 driver)
        3. 增加 NaN 检查与重试机制
        """
        n1, n2, n3 = X.shape
        X_f = torch.fft.fft(X, dim=2)
        X_res_f = torch.zeros_like(X_f)
        
        for i in range(n3):
            # [Fix 1] 必须在 try 之前定义 mat，供 except 使用
            mat = X_f[:, :, i]

            # [Fix 3] 提前拦截 NaN，防止报错
            if torch.isnan(mat).any() or torch.isinf(mat).any():
                # print(f"Warn: Slice {i} NaN/Inf detected in SVDShrink. Zeroing out.")
                # 返回零矩阵比报错更好，允许 ADMM 尝试恢复
                X_res_f[:, :, i] = 0
                continue

            try:
                # [Fix 2] 移除 driver='gesvd' 以兼容旧版 PyTorch
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            except RuntimeError:
                # 第一次重试：微小扰动
                try:
                    jitter = 1e-4 * torch.eye(mat.shape[0], mat.shape[1], device=mat.device, dtype=mat.dtype)
                    U, S, Vh = torch.linalg.svd(mat + jitter, full_matrices=False)
                except RuntimeError:
                    # 第二次重试：随机噪声
                    # print(f"SVD hard fail at {i}, adding noise.")
                    noise = 1e-3 * torch.randn_like(mat)
                    U, S, Vh = torch.linalg.svd(mat + noise, full_matrices=False)
            
            S_thresh = self.SoftShrink(S, tau)
            S_mat = torch.diag_embed(S_thresh).to(torch.complex64)
            X_res_f[:, :, i] = U @ S_mat @ Vh
            
        return torch.fft.ifft(X_res_f, dim=2).real

    def T_product(self, A, B):
        n1, _, n3 = A.shape
        l = B.shape[1]
        Af = torch.fft.fft(A, dim=2)
        Bf = torch.fft.fft(B, dim=2)
        Cf = torch.zeros((n1, l, n3), dtype=torch.complex64).to(A.device)
        for i in range(n3):
            Cf[:, :, i] = Af[:, :, i] @ Bf[:, :, i]
        return torch.fft.ifft(Cf, dim=2).real

    def ADMM_Collaborative(self, X, Y_onehot, D_mat=None):
        '''
        基于张量协同图学习的 ADMM 求解
        '''
        self.device = X.device 
        l, m, n = X.shape
        c = Y_onehot.shape[0]
        r = 30  # 降维维度
        
        # --- 超参数 ---
        lambda1 = 1e-2  # 稀疏噪声
        lambda2 = 1.0   # 判别项
        lambda3 = 1.0   # 协同重构
        lambda4 = 0.1   # 空间距离惩罚
        
        rho = 1e-1
        rho_max = 1e6
        mu_rate = 1.1
        max_iters = 60
        
        # --- 初始化 ---
        L = X.clone()
        E = torch.zeros((r, m, n), device=self.device)
        # 初始化 P 为正交矩阵，防止 Z 初始化为 0 导致后续除零
        P = torch.randn((r, l, n), device=self.device)
        for i in range(n):
            u, _, vh = torch.linalg.svd(P[:, :, i], full_matrices=False)
            P[:, :, i] = u @ vh
        
        H = torch.zeros((c, r, n), device=self.device) 
        S = torch.zeros((m, m, n), device=self.device) 
        
        # Z 初始化为 P*L，避免 Z=0 导致 H 更新时求逆崩溃
        Z = self.T_product(P, L)
        
        # 辅助变量 & 乘子
        M = L.clone()
        J = S.clone()
        
        W_M = torch.zeros_like(L)
        W_E = torch.zeros_like(E)
        W_Z = torch.zeros_like(Z)
        W_J = torch.zeros_like(S)
        
        if D_mat is None:
            D_mat = torch.ones((m, m), device=self.device)
        else:
            D_mat = D_mat.to(self.device)

        loss_history = [] 
        print(f"Start Collaborative ADMM... X shape: {X.shape}")

        for k in range(max_iters):
            # 1. Update M: min ||M||_* + rho/2 ||M - (L + W_M/rho)||^2
            # 加上数值保护，如果 L 已经炸了，限制范围
            L_in = L + W_M / rho
            if torch.isnan(L_in).any():
                L_in = torch.nan_to_num(L_in)
            M = self.SVDShrink(L_in, 1.0 / rho)
            
            # 2. Update J (Aux for S with Distance Penalty)
            D_sq = D_mat.unsqueeze(-1) ** 2
            Temp = S + W_J / rho
            Denominator = 2 * lambda4 * D_sq + rho
            J = (rho * Temp) / Denominator
            for i in range(n):
                J[:, :, i].fill_diagonal_(0)
                
            # 3. Update S (Collaborative Coefficient)
            # 求解 (2*lambda3 * Z^T Z + rho * I) S = ...
            target_S = J - W_J / rho
            Z_f = torch.fft.fft(Z, dim=2)
            tgt_S_f = torch.fft.fft(target_S, dim=2)
            S_f = torch.zeros_like(tgt_S_f)
            
            for i in range(n):
                z_slice = Z_f[:, :, i] 
                ZTZ = z_slice.conj().T @ z_slice
                eye = torch.eye(m, device=self.device)
                LHS = 2 * lambda3 * ZTZ + rho * eye
                RHS = 2 * lambda3 * ZTZ + rho * tgt_S_f[:, :, i]
                
                # 增加数值稳定性
                LHS = LHS + 1e-5 * eye
                try:
                    S_f[:, :, i] = torch.linalg.solve(LHS, RHS)
                except:
                    # 使用最小二乘解作为备选
                    S_f[:, :, i] = torch.linalg.lstsq(LHS, RHS).solution

            S = torch.fft.ifft(S_f, dim=2).real
            
            # 4. Update H (Classifier)
            # H = Y Z^T (Z Z^T)^-1
            Y_f = torch.fft.fft(Y_onehot, dim=2)
            H_f = torch.zeros((c, r, n), dtype=torch.complex64, device=self.device)
            for i in range(n):
                y_sl = Y_f[:, :, i]
                z_sl = Z_f[:, :, i]
                # 加大正则项防止除零
                ZZT = z_sl @ z_sl.conj().T + 1e-4 * torch.eye(r, device=self.device)
                RHS_H = y_sl @ z_sl.conj().T
                
                try:
                    H_f[:, :, i] = RHS_H @ torch.linalg.inv(ZZT)
                except RuntimeError:
                    H_f[:, :, i] = torch.linalg.lstsq(ZZT.T, RHS_H.T).solution.T

            H = torch.fft.ifft(H_f, dim=2).real
            
            # 5. Update Z (Latent Feature) - 梯度下降法
            # 容易发散，这里加入梯度裁剪
            PL = self.T_product(P, L)
            target_Z = PL - W_Z / rho
            
            HZ_Y = self.T_product(H, Z) - Y_onehot
            Grad1 = 2 * lambda2 * self.T_product(H.permute(1,0,2), HZ_Y)
            
            I_S = torch.eye(m, device=self.device).unsqueeze(2) - S
            ISIS = self.T_product(I_S, I_S.permute(1,0,2))
            Grad2 = 2 * lambda3 * self.T_product(Z, ISIS)
            
            Grad3 = rho * (Z - target_Z)
            
            Total_Grad = Grad1 + Grad2 + Grad3
            # [Fix] 梯度裁剪：防止 Z 爆炸导致后续计算 NaN
            Total_Grad = torch.clamp(Total_Grad, -10, 10) 
            
            # 步长可能需要调小，或者使用自适应优化器，这里手动减小步长
            step_size = 1e-4 if k < 10 else 1e-3
            Z = Z - step_size * Total_Grad
            
            # 6. Update E
            P_XL = self.T_product(P, X - L)
            E = self.SoftShrink(P_XL + W_E / rho, lambda1 / rho)
            
            # 7. Update L
            # L = (M + P^T(Z) + P^T(PX-E)) / 3
            term_M = M - W_M / rho
            term_Z = self.T_product(P.permute(1,0,2), Z + W_Z/rho)
            term_E = self.T_product(P.permute(1,0,2), self.T_product(P, X) - E + W_E/rho)
            L = (term_M + term_Z + term_E) / 3.0
            
            # 8. Update P (Orthogonal Procrustes)
            target_Z_P = Z + W_Z / rho
            target_E_P = E - W_E / rho
            
            A_mat = torch.cat([L, X - L], dim=1)
            B_mat = torch.cat([target_Z_P, target_E_P], dim=1)
            
            A_f = torch.fft.fft(A_mat, dim=2)
            B_f = torch.fft.fft(B_mat, dim=2)
            P_f = torch.zeros((r, l, n), dtype=torch.complex64, device=self.device)
            
            for i in range(n):
                Mat = B_f[:, :, i] @ A_f[:, :, i].conj().T
                
                if torch.isnan(Mat).any():
                    Mat = torch.nan_to_num(Mat) # 防止 SVD 崩溃

                try:
                    U, _, Vh = torch.linalg.svd(Mat, full_matrices=False)
                except RuntimeError:
                    jitter = 1e-4 * torch.eye(Mat.shape[0], Mat.shape[1], device=self.device, dtype=Mat.dtype)
                    U, _, Vh = torch.linalg.svd(Mat + jitter, full_matrices=False)

                P_f[:, :, i] = U @ Vh 
                
            P = torch.fft.ifft(P_f, dim=2).real
            
            # 9. Multipliers
            W_M = W_M + rho * (L - M)
            W_Z = W_Z + rho * (Z - self.T_product(P, L))
            W_E = W_E + rho * (self.T_product(P, X - L) - E)
            W_J = W_J + rho * (S - J)
            
            rho = min(rho * mu_rate, rho_max)
            
            # 10. 计算残差
            norm_X = torch.norm(X).item()
            res1 = torch.norm(L - M).item()
            res2 = torch.norm(Z - self.T_product(P, L)).item()
            res3 = torch.norm(self.T_product(P, X - L) - E).item()
            
            total_res = (res1 + res2 + res3) / (norm_X + 1e-9)
            loss_history.append(total_res)
            
            if k % 10 == 0:
                print(f"Iter {k}: Relative Residual={total_res:.6f}, rho={rho:.2e}")
                
            if torch.isnan(torch.tensor(total_res)):
                print("Convergence failed (NaN). Breaking.")
                break

        return M, L, E, P, H, S, Z, loss_history


if __name__ == '__main__':
    for lunshu in range(1, 2):
        # 1. 数据加载与预处理
        try:
            # 尝试加载真实数据
            image = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines.mat')['indian_pines_corrected']
            label = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        except:
            print("警告: 无法加载数据文件，使用随机数据进行测试。")
            image = np.random.rand(145, 145, 200)
            label = np.random.randint(0, 17, (145, 145))

        '''PCA'''
        rows, cols = np.nonzero(label != 0)
        coordinates = np.column_stack((rows, cols))
        fea = np.zeros((coordinates.shape[0], image.shape[2]))
        for i, j in enumerate(coordinates):
            fea[i, :] = image[j[0], j[1], :]
        
        pca = PCA(n_components=80) 
        pca.fit(fea)
        fea_reduced = pca.transform(fea)
        x = np.zeros((label.shape[0], label.shape[1], 80))
        for i in range(fea.shape[0]):
            a = coordinates[i][0]
            b = coordinates[i][1]
            x[a][b][:] = fea_reduced[i][:]
        image = x.astype(np.float32)

        try:
            random_idx = np.load('/data/LOH/TSPLL_label/random_idx/random_idx.npy')
        except:
            print("使用随机生成的索引")
            total_samples = coordinates.shape[0]
            # 随机选 5% 作为训练集
            random_idx = np.random.permutation(total_samples)[:int(0.05*total_samples)]

        PATCH_SIZE = 9
        # 注意：这里假设 tensor_function 里的 train_test_tensor_fold 可用
        try:
            x_train, train_label, x_test, test_label, train_label_list, test_label_list = train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
        except Exception as e:
            print(f"数据预处理出错: {e}, 正在生成虚拟 tensor 数据用于调试...")
            N_train = len(random_idx)
            x_train = torch.randn(80, N_train, 81) # Band, N, Patch^2
            train_label_list = np.random.randint(1, 17, N_train)
            x_test = torch.randn(80, 100, 81)
            test_label_list = np.random.randint(1, 17, 100)

        ours = TRPCA()

        # 2. 计算空间距离矩阵 D_mat
        if isinstance(random_idx, list): random_idx = np.array(random_idx)
        train_coords = coordinates[random_idx]
        dist_vec = pdist(train_coords, metric='euclidean')
        D_mat_np = squareform(dist_vec)
        D_mat_np = D_mat_np / (np.max(D_mat_np) + 1e-6)
        D_mat = torch.tensor(D_mat_np, dtype=torch.float32)

        lb = preprocessing.LabelBinarizer()
        lb.fit(range(1, 17))
        Y_train_onehot = lb.transform(train_label_list)
        Y_tensor = torch.tensor(Y_train_onehot.T).unsqueeze(2).repeat(1, 1, x_train.shape[2]).float().to(ours.device)
        
        x_train_tensor = x_train.to(ours.device)
        D_mat = D_mat.to(ours.device)

        # 3. 运行 ADMM
        M, L, E, P, H, S, Z, loss_history = ours.ADMM_Collaborative(x_train_tensor, Y_tensor, D_mat)
        
        # --- 绘制收敛曲线 ---
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, linewidth=2, color='red', marker='o', markersize=3)
        plt.title('Convergence Analysis')
        plt.xlabel('Iterations')
        plt.ylabel('Relative Residual')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('convergence_collaborative.png')
        print("Convergence plot saved.")

        # 4. 评估
        print("Evaluating...")
        X_train_reduced = ours.T_product(P, x_train_tensor)
        X_test_reduced = ours.T_product(P, x_test.to(ours.device))
        
        def nn_unique(x_train, train_label, x_test, test_label):
            xtr = x_train.cpu().detach().numpy() if torch.is_tensor(x_train) else x_train
            xte = x_test.cpu().detach().numpy() if torch.is_tensor(x_test) else x_test
            
            # 简单的 KNN (K=1)
            from scipy.spatial.distance import cdist
            # 展平 patch 维度进行距离计算 (Band, N, Patch) -> (N, Band*Patch)
            xtr_flat = xtr.transpose(1, 0, 2).reshape(xtr.shape[1], -1)
            xte_flat = xte.transpose(1, 0, 2).reshape(xte.shape[1], -1)
            
            D = cdist(xte_flat, xtr_flat, metric='euclidean')
            min_indices = np.argmin(D, axis=1)
            pred_labels = np.array(train_label)[min_indices]
            
            correct = np.sum(pred_labels == test_label)
            return correct / len(test_label)

        acc = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list)
        print(f"Test Accuracy (OA): {acc * 100:.2f}%")