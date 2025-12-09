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
# 请确保 tensor_function.py 和 tensor_function_MTLPP.py 在同一目录下
# =============================================================================
try:
    from tensor_function import Patch, getU, kmode_product, train_test_tensor, \
        train_test_zishiying, train_test_tensor_half, train_test_tensor_fold
    from tensor_function_MTLPP import dist_EMD
except ImportError:
    print("Error: Custom tensor modules not found.")
    print("Please ensure 'tensor_function.py' and 'tensor_function_MTLPP.py' are in the working directory.")
    sys.exit(1)


class TRPCA:
    def __init__(self):
        # 强制使用 CPU，利用 Batch 优化确保速度
        self.device = torch.device("cpu")

    def converged(self, M, L, E, P, W, G, X, M_new, L_new, E_new, P_new, W_new, G_new):
        eps = 1e-6
        c1 = torch.max(torch.abs(M_new - M)) < eps
        c2 = torch.max(torch.abs(L_new - L)) < eps
        c3 = torch.max(torch.abs(E_new - E)) < eps
        c4 = torch.max(torch.abs(P_new - P)) < eps
        c5 = torch.max(torch.abs(W_new - W)) < eps
        c6 = torch.max(torch.abs(G_new - G)) < eps

        # 检查约束 P*X = P*L + E (近似)
        res = self.T_product(P, L_new) + E_new - self.T_product(P, X)
        c7 = torch.max(torch.abs(res)) < eps

        return c1 and c2 and c3 and c4 and c5 and c6 and c7

    def SoftShrink(self, X, tau):
        # 软阈值算子
        z = torch.sgn(X) * torch.relu(torch.abs(X) - tau)
        return z

    def SVDShrink(self, X, tau):
        # 优化：频域 Batch SVD (CPU 并行)
        n1, n2, n3 = X.shape
        X_f = torch.fft.fft(X, dim=2)

        # 调整维度 (Batch, H, W) -> (n3, n1, n2)
        X_b = X_f.permute(2, 0, 1)

        # Batch SVD
        U, S, Vh = torch.linalg.svd(X_b, full_matrices=False)

        # 软阈值操作奇异值
        S_new = torch.relu(S - tau)

        # 重构: U @ diag(S) @ Vh
        S_mat = torch.zeros_like(X_b)
        idx = torch.arange(min(n1, n2), device=self.device)
        S_mat[:, idx, idx] = S_new

        W_b = torch.matmul(U, torch.matmul(S_mat, Vh))

        # 还原维度并 IFFT
        W_f = W_b.permute(1, 2, 0)
        return torch.fft.ifft(W_f, dim=2).real

    def getyi_yj(self, X_train_tensor_list, W, D):
        '''
        图拉普拉斯矩阵构建
        X_train_tensor_list: List of tensors/arrays (from block_diagonal_fft)
        '''
        l = len(X_train_tensor_list)
        # 获取维度信息
        sample_0 = X_train_tensor_list[0]
        if torch.is_tensor(sample_0):
            sample_0 = sample_0.numpy()

        dim = sample_0.shape[0]

        # 预分配大矩阵 (注意：样本量大时这里是内存瓶颈)
        try:
            re = np.zeros((dim, dim), dtype=np.complex64)
            re1 = np.zeros((dim, dim), dtype=np.complex64)
        except MemoryError:
            print("Error: Memory limit exceeded in graph construction.")
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

    def block_diagonal_fft(self, X):
        # 仅用于构图时的兼容函数
        # X: (d, 1, w^2) -> (d*w^2, w^2) sparse-like block diag in freq domain
        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X, dim=2)
        res = torch.zeros((n1 * n3, n3), dtype=torch.complex64)
        for i in range(n3):
            res[i * n1: (i + 1) * n1, i: i + 1] = Xf[:, :, i]
        return res

    def T_product(self, A, B):
        # Batch T-product 优化版
        n1, n2, n3 = A.shape
        Af = torch.fft.fft(A, dim=2)
        Bf = torch.fft.fft(B, dim=2)

        # Permute to (Batch, H, W)
        Ab = Af.permute(2, 0, 1)
        Bb = Bf.permute(2, 0, 1)

        # Batch matmul
        Cb = torch.matmul(Ab, Bb)

        Cf = Cb.permute(1, 2, 0)
        return torch.fft.ifft(Cf, dim=2).real

    def ADMM(self, left_mat, right_mat, X, Y):
        '''
        Optimized ADMM (CPU Batch Version)
        '''
        # 数据转换
        if not torch.is_tensor(X): X = torch.from_numpy(X).float()
        if not torch.is_tensor(Y): Y = torch.from_numpy(Y).float()
        X = X.to(self.device)
        Y = Y.to(self.device)

        if not torch.is_tensor(left_mat): left_mat = torch.from_numpy(left_mat).cfloat()
        if not torch.is_tensor(right_mat): right_mat = torch.from_numpy(right_mat).cfloat()
        left_mat = left_mat.to(self.device)
        right_mat = right_mat.to(self.device)

        l, m, n = X.shape
        c = Y.shape[0]
        r = 30  # 降维维度

        rho = 1.1
        mu = 1e-1
        mu_max = 1e10
        max_iters = 100
        lamb = 1
        beita = 1
        gama = 1  # lambda3

        # 初始化变量
        L = torch.zeros((l, m, n), device=self.device)
        M = torch.zeros((l, m, n), device=self.device)
        E = torch.zeros((r, m, n), device=self.device)
        P = torch.zeros((r, l, n), device=self.device)
        # 初始化 P 为正交阵 (可选)
        # P[:, :, 0] = torch.eye(r, l)
        Q = torch.zeros((r, l, n), device=self.device)
        W = torch.zeros((c, r, n), device=self.device)
        G = torch.zeros((c, l, n), device=self.device)

        # 拉格朗日乘子
        W1 = torch.zeros((r, m, n), device=self.device)
        W2 = torch.zeros((l, m, n), device=self.device)
        W3 = torch.zeros((c, l, n), device=self.device)
        W4 = torch.zeros((r, l, n), device=self.device)

        iters = 0
        print(f"ADMM Start: Shape X={X.shape}, Class={c}")

        while True:
            iters += 1

            # --- 1. Update M ---
            M_new = self.SVDShrink(L + W2 / mu, 1 / mu)

            # --- 2. Update L (Batch LS) ---
            Pf = torch.fft.fft(P, dim=2)
            Xf = torch.fft.fft(X, dim=2)
            Ef = torch.fft.fft(E, dim=2)
            Mf = torch.fft.fft(M_new, dim=2)
            W1f = torch.fft.fft(W1, dim=2)
            W2f = torch.fft.fft(W2, dim=2)

            Pb = Pf.permute(2, 0, 1)
            Xb = Xf.permute(2, 0, 1)
            Eb = Ef.permute(2, 0, 1)
            Mb = Mf.permute(2, 0, 1)
            W1b = W1f.permute(2, 0, 1)
            W2b = W2f.permute(2, 0, 1)

            Ph_b = Pb.conj().transpose(1, 2)

            Term1 = torch.matmul(Ph_b, torch.matmul(Pb, Xb) - Eb + W1b / mu)
            Term2 = Mb - W2b / mu
            Lb_new = 0.5 * (Term1 + Term2)

            L_new = torch.fft.ifft(Lb_new.permute(1, 2, 0), dim=2).real

            # --- 3. Update E ---
            PL = self.T_product(P, L_new)
            PX = self.T_product(P, X)
            E_new = self.SoftShrink(PX - PL + W1 / mu, lamb / mu)

            # --- 4. Update W (H) ---
            Gf = torch.fft.fft(G, dim=2)
            W3f = torch.fft.fft(W3, dim=2)
            Gb = Gf.permute(2, 0, 1)
            W3b = W3f.permute(2, 0, 1)

            Mb = torch.matmul(Gb - W3b / mu, Ph_b)

            Ub, Sb, Vhb = torch.linalg.svd(Mb, full_matrices=False)
            Wb_new = torch.matmul(Ub, Vhb)
            W_new = torch.fft.ifft(Wb_new.permute(1, 2, 0), dim=2).real

            # --- 5. Update G ---
            Yf = torch.fft.fft(Y, dim=2)
            Yb = Yf.permute(2, 0, 1)
            Xb_new = torch.fft.fft(X, dim=2).permute(2, 0, 1)
            Xh_b = Xb_new.conj().transpose(1, 2)

            T1 = beita * torch.matmul(Yb, Xh_b)
            Pb_curr = torch.fft.fft(P, dim=2).permute(2, 0, 1)
            T2 = (mu / 2) * (torch.matmul(Wb_new, Pb_curr) + W3b / mu)

            Ug, Sg, Vhg = torch.linalg.svd(T1 + T2, full_matrices=False)
            Gb_new = torch.matmul(Ug, Vhg)
            G_new = torch.fft.ifft(Gb_new.permute(1, 2, 0), dim=2).real

            # --- 6. Update P ---
            Eb_new = torch.fft.fft(E_new, dim=2).permute(2, 0, 1)
            Lb_new_f = torch.fft.fft(L_new, dim=2).permute(2, 0, 1)
            W1b = torch.fft.fft(W1, dim=2).permute(2, 0, 1)
            Qb = torch.fft.fft(Q, dim=2).permute(2, 0, 1)
            W4b = torch.fft.fft(W4, dim=2).permute(2, 0, 1)

            Diff_XL = (Xb_new - Lb_new_f).conj().transpose(1, 2)
            Part1 = torch.matmul(Eb_new - W1b / mu, Diff_XL)

            Wh_b = Wb_new.conj().transpose(1, 2)
            Part2 = torch.matmul(Wh_b, Gb_new - W3b / mu)

            Part3 = Qb - W4b / mu

            Up, Sp, Vhp = torch.linalg.svd(Part1 + Part2 + Part3, full_matrices=False)
            Pb_new = torch.matmul(Up, Vhp)
            P_new = torch.fft.ifft(Pb_new.permute(1, 2, 0), dim=2).real

            # --- 7. Update Q (Big Matrix Gradient Descent) ---
            # 辅助：Tensor -> Big Block Diag
            def tensor_to_big(T):
                Tf = torch.fft.fft(T, dim=2)
                r, l, n = T.shape
                big = torch.zeros((r * n, l * n), dtype=torch.complex64, device=self.device)
                for i in range(n):
                    big[i * r:(i + 1) * r, i * l:(i + 1) * l] = Tf[:, :, i]
                return big

            P_big = tensor_to_big(P_new)
            W4_big = tensor_to_big(W4)
            Q_big = tensor_to_big(Q)

            lambda_val = 1.0
            lr = 0.01
            q_iters = 10

            for _ in range(q_iters):
                QL = Q_big @ left_mat
                Diff = P_big - Q_big + W4_big / mu
                QD = Q_big @ right_mat

                grad = 2 * gama * QL - mu * Diff + 2 * lambda_val * QD
                Q_big = Q_big - lr * grad

                tr_val = torch.trace(Q_big @ right_mat @ Q_big.conj().T).real
                lambda_val += lr * (tr_val - 1)

            Qf_new = torch.zeros((r, l, n), dtype=torch.complex64, device=self.device)
            for i in range(n):
                Qf_new[:, :, i] = Q_big[i * r:(i + 1) * r, i * l:(i + 1) * l]
            Q_new = torch.fft.ifft(Qf_new, dim=2).real

            # --- 8. Update Multipliers ---
            term1 = self.T_product(P_new, X) - self.T_product(P_new, L_new) - E_new
            W1 += mu * term1
            W2 += mu * (L_new - M_new)
            term3 = self.T_product(W_new, P_new) - G_new
            W3 += mu * term3
            W4 += mu * (P_new - Q_new)

            mu = min(rho * mu, mu_max)

            # Check
            if self.converged(M, L, E, P, W, G, X, M_new, L_new, E_new, P_new, W_new, G_new) or iters >= max_iters:
                print(f"Converged at iteration {iters}")
                return M_new, L_new, E_new, P_new, W_new, G_new, Q_new
            else:
                M, L, E, P, W, G, Q = M_new, L_new, E_new, P_new, W_new, G_new, Q_new
                if iters % 10 == 0:
                    err = torch.max(torch.abs(term1))
                    print(f"Iter {iters}: RecErr={err:.6f}, mu={mu:.2f}")





if __name__ =='__main__':
    for lunshu in range(1, 2):
        # R = loadmat('./Indian_pines_16Classes.mat')['R']
        # image = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines.mat')['indian_pines_corrected']
        # label = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        image = loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        label = loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        # label = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_gt']
        # image = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_corrected']
        spy.imshow(classes=label)  # 确保 label 是 2D 整数数组
        plt.axis('off')
        plt.savefig('/data/LOH/TSPLL_label/longkou-gt.jpg', dpi=300, bbox_inches='tight',interpolation='none')
        plt.show()


        '''add noise(0,0.5)零均值高斯噪声    随机生成胡椒和盐噪声的比例(0,0.3)'''
        image = image.astype(float)
        for band in range(image.shape[2]):
            image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                        np.max(image[:, :, band]) - np.min(image[:, :, band]))
        np.random.seed(42)
        noisy_image = np.copy(image).astype(np.float64)
        selected_bands = np.random.choice(image.shape[2], size=30, replace=False)
        for i in selected_bands:
            variance = np.random.uniform(0, 0.5)
            noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
            noisy_image[..., i] += noise
            # 随机生成胡椒和盐噪声的比例(0,0.3)
            salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
            pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
            noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
            noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
        image = noisy_image


        # random_idx = np.load('/data/LOH/TSPLL_label/random_idx/random_idx.npy')
        random_idx_dict = np.load('/data/LOH/TSPLL_label/random_idx/random_idx_longkou_0.2%.npy', allow_pickle=True).item()
        random_idx = []
        for class_label, indices in random_idx_dict.items():
            random_idx.extend(indices['train_indices'])

        PATCH_SIZE = 9
        x_train, train_label, x_test, test_label ,train_label_list, test_label_list= train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
        ours = TRPCA()

        '''计算临近点'''
        x_train_W, _, _, _ = train_test_tensor_half(PATCH_SIZE,random_idx, image, label)
        l = len(x_train_W)
        ci = []
        b = x_train_W[0].shape[2]
        for i in range(l):
            c_matrix = torch.zeros((b, b))
            xt = x_train_W[i]
            ui = torch.mean(xt, dim=(0, 1), keepdim=True)
            ui1 = ui.reshape(b, -1)
            for m in range(9):
                for n in range(9):
                    xt1 = xt[m, n, :].reshape(b, -1)
                    c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
            c_matrix = c_matrix / 80
            ci.append(c_matrix)
        t = 1000
        k_near = 10
        w, d = dist_EMD(x_train_W, ci, k_near, t)
        # np.save('w-1000-indian', w)
        # np.save('d-1000-indian', d)
        X_train_fft_list = []
        for i in range(x_train.shape[1]):
            x = x_train[:, i, :].reshape(b, 1, 81)
            x =ours.block_diagonal_fft(x)
            X_train_fft_list.append(x)

        left ,right= ours.getyi_yj(X_train_fft_list, w, d)  # (9,9)
        left = torch.from_numpy(left)
        right = torch.from_numpy(right)

        M, L, E, P, W, G, Q= ours.ADMM(left,right,x_train,train_label)
        X_train_reduced = ours.T_product(P, x_train)
        X_train_reduced1 = ours.T_product(P, L)
        X_train_reduced_Q = ours.T_product(Q, x_train)
        X_train_reduced_Q1 = ours.T_product(Q, L)
        X_test_reduced = ours.T_product(P, x_test)
        X_test_reduced_Q = ours.T_product(Q, x_test)

        def nn_unique(x_train, train_label, x_test, test_label, random, label):
            computedClass = []
            D = np.zeros((x_test.shape[1], x_train.shape[1]))
            # 计算距离矩阵
            for i in range(x_test.shape[1]):
                current_block = x_test[:, i, :]
                for j in range(x_train.shape[1]):
                    neighbor_block = x_train[:, j, :]
                    w = current_block - neighbor_block
                    d = torch.linalg.norm(w)
                    D[i, j] = d

            # 获取每个测试样本的最小距离邻居
            id = np.argsort(D, axis=1)
            computedClass.append(np.array(train_label)[id[:, 0]])

            # 将预测类别加入到test_label中（更新标签）
            updated_test_label = np.copy(test_label)  # 复制原始标签以保留原始信息
            updated_test_label[:] = computedClass[0]  # 将计算得到的预测标签填入更新后的标签中

            rows, cols = np.nonzero(label != 0)
            coordinates = np.column_stack((rows, cols))

            label_matrix = np.copy(label)
            idx_test = np.setdiff1d(range(len(coordinates)), random)

            # Process testing patches 可视化分类结果
            for test_idx, coord in enumerate(idx_test):
                i, j = coordinates[coord]
                label_matrix[i, j] = updated_test_label[test_idx]

            # 计算总精度 OA
            total_correct = np.sum(updated_test_label == test_label)
            precision_OA = total_correct / x_test.shape[1]

            # 计算每个类别的精度
            unique_classes = np.unique(train_label)
            class_precision = {}

            for cls in unique_classes:
                # 获取该类别在测试集中的索引
                test_cls_indices = np.where(test_label == cls)[0]
                if len(test_cls_indices) == 0:
                    continue  # 如果没有该类别的测试样本，跳过

                correct_count = 0
                for idx in test_cls_indices:
                    # 比较预测标签和真实标签
                    if updated_test_label[idx] == cls:
                        correct_count += 1

                # 计算该类别的精度
                precision = correct_count / len(test_cls_indices)
                class_precision[cls] = precision

            # 计算平均精度 (AA)
            precision_AA = np.mean(list(class_precision.values()))

            # 计算 Cohen's Kappa
            # 混淆矩阵
            confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
            class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

            for i in range(len(test_label)):
                actual_idx = class_to_index[test_label[i]]
                predicted_idx = class_to_index[updated_test_label[i]]
                confusion_matrix[actual_idx, predicted_idx] += 1

            total_samples = np.sum(confusion_matrix)
            P_o = np.trace(confusion_matrix) / total_samples  # 观测一致性
            P_e = np.sum(
                (np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples ** 2
            )  # 随机一致性
            precision_Kappa = (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0  # 防止分母为 0

            # 返回总精度和每类精度，及更新后的标签矩阵
            precision = {
                'total_precision': precision_OA,
                'class_precision': class_precision,
                'average_precision': precision_AA,
                'kappa': precision_Kappa
            }
            return precision, label_matrix


        acc,tup = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list, random_idx, label)
        # view2 = spy.imshow(classes=tup, title="acc-gt")
        # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/P*X.jpg')

        acc1,tup1 = nn_unique(X_train_reduced1, train_label_list, X_test_reduced, test_label_list, random_idx, label)
        # view3 = spy.imshow(classes=tup1, title="acc1-gt")
        # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/P*L.jpg')

        accQ,tupQ = nn_unique(X_train_reduced_Q, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
        # view4 = spy.imshow(classes=tupQ, title="accQ-gt")
        # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/Q*X.jpg')

        accQ1,tupQ1 = nn_unique(X_train_reduced_Q1, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
        # view5 = spy.imshow(classes=tupQ1, title="accQ1-gt")
        # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/Q*L.jpg')
        # plt.pause(60)

        # print acc
        with open('/data/LOH/TSPLL_label/Classification maps/noise/longkou/result.txt', 'a', encoding='utf-8') as file:
            sys.stdout = file
            print(lunshu)
            print(f"P*X Total Precision: {acc['total_precision']* 100:.4f}")
            print(f"P*X average_precision: {acc['average_precision'] * 100:.4f}")
            print(f"P*X kappa: {acc['kappa'] * 100:.4f}")
            for cls, precision in acc['class_precision'].items():
                print(f"Class {cls} Precision: {precision * 100:.4f}")

            print(f"P*L Total Precision: {acc1['total_precision']* 100:.4f}")
            print(f"P*L average_precision: {acc1['average_precision'] * 100:.4f}")
            print(f"P*L kappa: {acc1['kappa'] * 100:.4f}")
            for cls, precision in acc1['class_precision'].items():
                print(f"Class {cls} Precision: {precision * 100:.4f}")

            print(f"Q*X Total Precision: {accQ['total_precision'] * 100:.4f}")
            print(f"Q*X average_precision: {accQ['average_precision'] * 100:.4f}")
            print(f"Q*X kappa: {accQ['kappa'] * 100:.4f}")
            for cls, precision in accQ['class_precision'].items():
                print(f"Class {cls} Precision: {precision * 100:.4f}")

            print(f"Q*L Total Precision: {accQ1['total_precision'] * 100:.4f}")
            print(f"Q*L average_precision: {accQ1['average_precision'] * 100:.4f}")
            print(f"Q*L kappa: {accQ1['kappa'] * 100:.4f}")
            for cls, precision in accQ1['class_precision'].items():
                print(f"Class {cls} Precision: {precision * 100:.4f}")
            sys.stdout = sys.__stdout__