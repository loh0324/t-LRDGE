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
# 保持引用不变
from tensor_function import Patch, getU, kmode_product, train_test_tensor, train_test_zishiying, \
    train_test_tensor_half, train_test_tensor_fold
from tensor_function_MTLPP import dist_EMD

class TRPCA:
    def __init__(self):
        # 自动检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def converged(self, M, L, E, P, W, G, X, M_new, L_new, E_new, P_new, W_new, G_new):
        '''判断收敛条件'''
        eps = 1e-4
        condition2 = torch.max(torch.abs(L_new - L)) < eps
        condition4 = torch.max(torch.abs(P_new - P)) < eps
        return condition2 and condition4

    def SoftShrink(self, X, tau):
        if not torch.is_tensor(tau):
            tau = torch.tensor(tau).to(X.device)
        return torch.sign(X) * torch.maximum(torch.abs(X) - tau, torch.zeros_like(X))

    def SVDShrink(self, X, tau):
        n1, n2, n3 = X.shape
        X_f = torch.fft.fft(X, dim=2)
        X_res_f = torch.zeros_like(X_f)
        for i in range(n3):
            U, S, Vh = torch.linalg.svd(X_f[:, :, i], full_matrices=False)
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

    def block_diagonal_fft(self, X):
        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X, dim=2)
        Xf_block_diag = torch.zeros((n1 * n3, n2 * n3), dtype=torch.complex64).to(X.device)
        for i in range(n3):
            Xf_block_diag[i * n1:(i + 1) * n1, i * n2:(i + 1) * n2] = Xf[:, :, i]
        return Xf_block_diag

    def ADMM_Collaborative(self, X, Y_onehot, D_mat=None):
        '''
        创新点2：基于张量协同图学习的 ADMM 求解
        min ||L||_* + lam1 ||E||_1 
            + lam2 ||Y - H * (P*L)||_F^2 
            + lam3 ||Z - Z * S||_F^2 + lam4 ||D .* S||_F^2
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
        P = torch.zeros((r, l, n), device=self.device)
        for i in range(n):
            torch.nn.init.orthogonal_(P[:, :, i])
        
        H = torch.zeros((c, r, n), device=self.device) 
        S = torch.zeros((m, m, n), device=self.device) 
        Z = torch.zeros((r, m, n), device=self.device) 
        
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

        loss_history = [] # 记录收敛曲线
        print(f"Start Collaborative ADMM... X shape: {X.shape}")

        for k in range(max_iters):
            # 1. Update M
            M = self.SVDShrink(L + W_M / rho, 1.0 / rho)
            
            # 2. Update J (Aux for S with Distance Penalty)
            D_sq = D_mat.unsqueeze(-1) ** 2
            Temp = S + W_J / rho
            Denominator = 2 * lambda4 * D_sq + rho
            J = (rho * Temp) / Denominator
            for i in range(n):
                J[:, :, i].fill_diagonal_(0)
                
            # 3. Update S (Collaborative Coefficient)
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
                LHS = LHS + 1e-5 * eye
                S_f[:, :, i] = torch.linalg.solve(LHS, RHS)
            S = torch.fft.ifft(S_f, dim=2).real
            
            # 4. Update H (Classifier)
            Y_f = torch.fft.fft(Y_onehot, dim=2)
            H_f = torch.zeros((c, r, n), dtype=torch.complex64, device=self.device)
            for i in range(n):
                y_sl = Y_f[:, :, i]
                z_sl = Z_f[:, :, i]
                ZZT = z_sl @ z_sl.conj().T + 1e-4 * torch.eye(r, device=self.device)
                RHS = y_sl @ z_sl.conj().T
                H_f[:, :, i] = RHS @ torch.linalg.inv(ZZT)
            H = torch.fft.ifft(H_f, dim=2).real
            
            # 5. Update Z (Latent Feature)
            PL = self.T_product(P, L)
            target_Z = PL - W_Z / rho
            
            HZ_Y = self.T_product(H, Z) - Y_onehot
            Grad1 = 2 * lambda2 * self.T_product(H.permute(1,0,2), HZ_Y)
            
            I_S = torch.eye(m, device=self.device).unsqueeze(2) - S
            ISIS = self.T_product(I_S, I_S.permute(1,0,2))
            Grad2 = 2 * lambda3 * self.T_product(Z, ISIS)
            
            Grad3 = rho * (Z - target_Z)
            
            Z = Z - 1e-3 * (Grad1 + Grad2 + Grad3)
            
            # 6. Update E
            P_XL = self.T_product(P, X - L)
            E = self.SoftShrink(P_XL + W_E / rho, lambda1 / rho)
            
            # 7. Update L
            term_M = M - W_M / rho
            term_Z = self.T_product(P.permute(1,0,2), Z + W_Z/rho)
            term_E = self.T_product(P.permute(1,0,2), self.T_product(P, X) - E + W_E/rho)
            L = (term_M + term_Z + term_E) / 3.0
            
            # 8. Update P
            target_Z_P = Z + W_Z / rho
            target_E_P = E - W_E / rho
            
            A = torch.cat([L, X - L], dim=1)
            B = torch.cat([target_Z_P, target_E_P], dim=1)
            
            A_f = torch.fft.fft(A, dim=2)
            B_f = torch.fft.fft(B, dim=2)
            P_f = torch.zeros((r, l, n), dtype=torch.complex64, device=self.device)
            
            for i in range(n):
                Mat = B_f[:, :, i] @ A_f[:, :, i].conj().T
                U, _, Vh = torch.linalg.svd(Mat, full_matrices=False)
                P_f[:, :, i] = U @ Vh 
                
            P = torch.fft.ifft(P_f, dim=2).real
            
            # 9. Multipliers
            W_M = W_M + rho * (L - M)
            W_Z = W_Z + rho * (Z - self.T_product(P, L))
            W_E = W_E + rho * (self.T_product(P, X - L) - E)
            W_J = W_J + rho * (S - J)
            
            rho = min(rho * mu_rate, rho_max)
            
            # 10. 计算残差并记录 (用于画图)
            # 主要约束误差: L-M, Z-PL, PX-PL-E
            res1 = torch.norm(L - M)
            res2 = torch.norm(Z - PL)
            res3 = torch.norm(self.T_product(P, X - L) - E)
            total_res = (res1 + res2 + res3).item()
            loss_history.append(total_res)
            
            if k % 10 == 0:
                print(f"Iter {k}, Residual={total_res:.4f}, rho={rho:.2e}")

        # 返回值增加 loss_history
        return M, L, E, P, H, S, Z, loss_history


if __name__ == '__main__':
    for lunshu in range(1, 2):
        # 1. 数据加载与预处理
        try:
            image = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines.mat')['indian_pines_corrected']
            label = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        except:
            print("警告: 使用随机数据测试")
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

        '''Add Noise'''
        image = image.astype(float)
        for band in range(image.shape[2]):
            mi, ma = np.min(image[:, :, band]), np.max(image[:, :, band])
            if ma != mi:
                image[:, :, band] = (image[:, :, band] - mi) / (ma - mi)
        
        np.random.seed(42)
        noisy_image = np.copy(image).astype(np.float64)
        selected_bands = np.random.choice(image.shape[2], size=60, replace=False)
        for i in selected_bands:
            variance = np.random.uniform(0, 0.5)
            noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
            noisy_image[..., i] += noise
            salt = np.random.rand(*image[..., i].shape) < 0.05
            pepper = np.random.rand(*image[..., i].shape) < 0.05
            noisy_image[salt, i] = np.max(image[..., i])
            noisy_image[pepper, i] = np.min(image[..., i])
        image = noisy_image

        try:
            random_idx = np.load('/data/LOH/TSPLL_label/random_idx/random_idx.npy')
        except:
            total_samples = coordinates.shape[0]
            random_idx = np.random.permutation(total_samples)[:int(0.05*total_samples)]

        PATCH_SIZE = 9
        x_train, train_label, x_test, test_label, train_label_list, test_label_list = train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
        ours = TRPCA()

        # 2. 计算空间距离矩阵 D_mat
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

        # 3. 运行 ADMM 并获取 loss_history
        M, L, E, P, H, S, Z, loss_history = ours.ADMM_Collaborative(x_train_tensor, Y_tensor, D_mat)
        
        # --- 新增: 绘制收敛曲线 ---
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, linewidth=2, color='red', marker='o', markersize=3)
        plt.title('Convergence Analysis (Collaborative Graph Learning)')
        plt.xlabel('Iterations')
        plt.ylabel('Total Residual (Log scale)')
        plt.yscale('log') # 使用对数坐标看收敛更清晰
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('convergence_collaborative.png', dpi=300)
        print("收敛曲线已保存为 'convergence_collaborative.png'")
        # plt.show() # 如果在服务器运行请注释此行

        # 4. 评估函数
        X_train_reduced = ours.T_product(P, x_train_tensor)
        X_test_reduced = ours.T_product(P, x_test.to(ours.device))
        X_train_reduced1 = ours.T_product(P, L)
        
        def nn_unique(x_train, train_label, x_test, test_label, random, label):
            computedClass = []
            
            xtr = x_train.cpu().detach().numpy() if torch.is_tensor(x_train) else x_train
            xte = x_test.cpu().detach().numpy() if torch.is_tensor(x_test) else x_test
            
            D = np.zeros((xte.shape[1], xtr.shape[1]))
            for i in range(xte.shape[1]):
                current_block = xte[:, i, :]
                for j in range(xtr.shape[1]):
                    neighbor_block = xtr[:, j, :]
                    w = current_block - neighbor_block
                    d = np.linalg.norm(w)
                    D[i, j] = d

            id = np.argsort(D, axis=1)
            computedClass.append(np.array(train_label)[id[:, 0]])

            updated_test_label = np.copy(test_label)
            updated_test_label[:] = computedClass[0]

            rows, cols = np.nonzero(label != 0)
            coordinates = np.column_stack((rows, cols))
            label_matrix = np.copy(label)
            idx_test = np.setdiff1d(range(len(coordinates)), random)

            for test_idx, coord in enumerate(idx_test):
                i, j = coordinates[coord]
                label_matrix[i, j] = updated_test_label[test_idx]

            total_correct = np.sum(updated_test_label == test_label)
            precision_OA = total_correct / xte.shape[1]

            unique_classes = np.unique(train_label)
            class_precision = {}

            for cls in unique_classes:
                test_cls_indices = np.where(test_label == cls)[0]
                if len(test_cls_indices) == 0:
                    continue
                correct_count = 0
                for idx in test_cls_indices:
                    if updated_test_label[idx] == cls:
                        correct_count += 1
                precision = correct_count / len(test_cls_indices)
                class_precision[cls] = precision

            precision_AA = np.mean(list(class_precision.values()))

            confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
            class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

            for i in range(len(test_label)):
                actual_idx = class_to_index[test_label[i]]
                predicted_idx = class_to_index[updated_test_label[i]]
                confusion_matrix[actual_idx, predicted_idx] += 1

            total_samples = np.sum(confusion_matrix)
            P_o = np.trace(confusion_matrix) / total_samples
            P_e = np.sum(
                (np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples ** 2
            )
            precision_Kappa = (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

            precision = {
                'total_precision': precision_OA,
                'class_precision': class_precision,
                'average_precision': precision_AA,
                'kappa': precision_Kappa
            }
            return precision, label_matrix

        acc, tup = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list, random_idx, label)
        acc1, tup1 = nn_unique(X_train_reduced1, train_label_list, X_test_reduced, test_label_list, random_idx, label)
        
        print(f"Lunshu: {lunshu}")
        print("-" * 30)
        print("P*X Results (Collaborative):")
        print(f"Total Precision (OA): {acc['total_precision'] * 100:.4f}")
        print(f"Average Precision (AA): {acc['average_precision'] * 100:.4f}")
        print(f"Kappa: {acc['kappa'] * 100:.4f}")
        for cls, precision in acc['class_precision'].items():
            print(f"Class {cls} Precision: {precision * 100:.4f}")
            
        print("-" * 30)
        print("P*L Results (De-noised):")
        print(f"Total Precision (OA): {acc1['total_precision'] * 100:.4f}")
        print(f"Average Precision (AA): {acc1['average_precision'] * 100:.4f}")
        print(f"Kappa: {acc1['kappa'] * 100:.4f}")