import math
import sys
import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from sklearn.decomposition import PCA
import torch
from scipy.io import loadmat
from tensor_function import Patch,getU,kmode_product,train_test_tensor,train_test_zishiying,train_test_tensor_half,train_test_tensor_fold,train_test_tensor_half
from tensor_function_MTLPP import dist_EMD
import matplotlib.pyplot as plt
from scipy.io import loadmat
import spectral as spy

class TRPCA:

    def converged(self, M, L, E, P, W, G, X, M_new, L_new, E_new,P_new,W_new,G_new):
        '''M, L, E, P, X, M_new, L_new, E_new,P_new
        judge convered or not
        '''
        eps = 1e-8
        condition1 = torch.max(M_new - M) < eps
        condition2 = torch.max(L_new - L) < eps
        condition3 = torch.max(E_new - E) < eps
        condition4 = torch.max(P_new - P) < eps
        condition5 = torch.max(W_new - W) < eps
        condition6 = torch.max(G_new - G) < eps
        condition7 = torch.max(self.T_product(P, L) + E_new - self.T_product(P, X)) < eps
        return condition1 and condition2 and condition3 and condition4 and condition5 and condition6 and condition7

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = torch.sgn(X) * (torch.abs(X) - tau) * ((torch.abs(X) - tau) > 0)

        return z
    def SoftShrink1(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

        return z
    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        X = X.cpu().numpy()
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range(X.shape[2]):
            if i < X.shape[2]:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink1(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == X.shape[2]:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))

        return torch.from_numpy(np.fft.ifft(W_bar).real).to(torch.float32)

    def block_diagonal_fft(self,X):
        # 将三阶张量X转换为傅里叶域的块对角形式X~
        # X - n1 x n2 x n3 tensor
        # X~ - n1 x n2 x n3 tensor in the Fourier domain

        n1, n2, n3 = X.shape
        Xf = torch.fft.fft(X)
        Xf_block_diag = torch.zeros((n1 * n3, n2 * n3))

        for i in range(n3):
            Xf_block_diag[i * n1:(i + 1) * n1, i * n2:(i + 1) * n2] = Xf[:, :, i]

        # X_block_diag = np.fft.ifft(Xf_block_diag)

        return Xf_block_diag

    def getyi_yj(self,Y, W,D):
        l = len(Y)
        re = np.zeros(np.dot(Y[0], Y[0].T).shape)
        re1 = np.zeros(np.dot(Y[0], Y[0].T).shape)
        for i in range(l):
            re1 = re1 + np.dot(Y[i], Y[i].T) * D[i][i]
            for j in range(i + 1, l):
                if W[i][j] != 0:
                    re = re + np.dot((Y[i] - Y[j]), (Y[i] - Y[j]).T) * W[i][j] * 2
                    # re = re + np.dot((Y[i] - Y[j]), (Y[i] - Y[j]).T) * W[i][j]
        return re,re1

    # def getyy(self,Y, D):
    #     l = len(Y)
    #     re = np.zeros(np.dot(Y[0], Y[0].T).shape)
    #     for i in range(l):
    #         re = re + np.dot(Y[i], Y[i].T) * D[i][i]
    #     return re
    def T_product(self,A,B):
            # tensor-tensor product of two 3-order tensors: C = A * B
            # compute in the Fourier domain, efficiently
            # A - n1 x n2 x n3 tensor
            # B - n2 x l  x n3 tensor
            # C - n1 x l  x n3 tensor n1降维 n2原维

            n1, _, n3 = A.shape
            l = B.shape[1]
            Af = torch.fft.fft(A)
            Bf = torch.fft.fft(B)
            Cf = torch.zeros((n1, l, n3), dtype = torch.complex64)
            for i in range(n3):
                Cf[:, :, i] = Af[:, :, i] @ Bf[:, :, i]
            C = torch.fft.ifft(Cf).real
            return C
    def ADMM(self,left,right,X,Y):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        l, m, n = X.shape
        c = Y.shape[0]
        r = 30#reduce
        rho = 1.1
        mu = 1
        mu_max = 1e10
        max_iters = 100
        lamb = 1
        beita = 1
        gama = 1
        L = torch.zeros((l, m, n))
        W = torch.zeros((c, r, n))
        M = torch.zeros((l, m, n))
        P = torch.zeros((r, l, n))
        Q = torch.zeros((r, l, n))
        E = torch.zeros((r, m, n))
        G = torch.zeros((c, l, n))
        I = torch.zeros((l, l, n))
        I[:,:,0] = torch.eye(l)
        W1 = torch.zeros((r, m, n))
        W2 = torch.zeros((l, m, n))
        W3 = torch.zeros((c, l, n))
        W4 = torch.zeros((r, l, n))
        iters = 0
        loss = []
        print("ADMM Optimization Start...")
        while True:
            iters += 1
            # update M
            M_new = self.SVDShrink(L + (1 / mu) * W2, 1 / mu)
            # update L(recovered image)
            X1 = self.block_diagonal_fft(X)
            P1 = self.block_diagonal_fft(P)
            M1 = self.block_diagonal_fft(M_new)
            E1 = self.block_diagonal_fft(E)
            W11 = self.block_diagonal_fft(W1)
            W22 = self.block_diagonal_fft(W2)
            I1 = self.block_diagonal_fft(I)
            z = P1.T @ P1 + I1
            z = torch.inverse(z)
            L_new_diag = z @ (P1.T @ (P1 @ X1 - E1 + W11/mu)+(M1 - W22/mu))
            L_new = torch.zeros((l, m, n),dtype = torch.complex64)
            for i in range(n):
                L_new[:,:,i] = L_new_diag[i * l:(i + 1) * l, i * m:(i + 1) * m]
            L_new = torch.fft.ifft(L_new).real

            # update E(noise)
            a = self.T_product(P, X)
            b = self.T_product(P, L_new)
            E_new = self.SoftShrink(a - b + (1 / mu) * W1, lamb / mu)

            # update W
            P1 = self.block_diagonal_fft(P)
            G1 = self.block_diagonal_fft(G)
            W33 = self.block_diagonal_fft(W3)
            f = (G1-(1 / mu) * W33) @ P1.T
            f = (mu/2)*f
            U, _, V = torch.svd(f)
            W_new_diag = U @ V.T
            W_new = torch.zeros((c, r, n))
            for i in range(n):
                W_new[:, :, i] = W_new_diag[i * c:(i + 1) * c, i * r:(i + 1) * r]
            W_new = torch.fft.ifft(W_new).real

            # update G
            Y1 = self.block_diagonal_fft(Y)
            W_1 = self.block_diagonal_fft(W_new)
            P1 = self.block_diagonal_fft(P)
            W33 = self.block_diagonal_fft(W3)
            f=beita*(Y1@X1.T) + (mu/2)*(W_1@P1+(1 / mu) * W33)
            U, _, V = torch.svd(f)
            G_new_diag = U @ V.T
            G_new = torch.zeros((c, l, n))
            for i in range(n):
                G_new[:, :, i] = G_new_diag[i * c:(i + 1) * c, i * l:(i + 1) * l]
            G_new = torch.fft.ifft(G_new).real

            # update P
            W_1 = self.block_diagonal_fft(W_new)
            X1 = self.block_diagonal_fft(X)
            L1 = self.block_diagonal_fft(L_new)
            E1 = self.block_diagonal_fft(E_new)
            Q1 = self.block_diagonal_fft(Q)
            G1 = self.block_diagonal_fft(G_new)
            W11 = self.block_diagonal_fft(W1)
            W44 = self.block_diagonal_fft(W4)
            f = (E1 - W11 / mu) @ (X1 - L1).T + W_1.T @ (G1 - W33 / mu) + (Q1 - W44 / mu)
            # f = (E1 - W11 / mu) @ (X1 - L1).T + (G1 - W33 / mu).T @ W_1 + (Q1 - W44 / mu)
            f = (mu / 2) * f
            U, _, V = torch.svd(f)
            P_new_diag = U @ V.T
            P_new = torch.zeros((r, l, n))
            for i in range(n):
                P_new[:, :, i] = P_new_diag[i * r:(i + 1) * r, i * l:(i + 1) * l]
            P_new = torch.fft.ifft(P_new).real

            # update Q
            P1 = self.block_diagonal_fft(P_new)
            W44 = self.block_diagonal_fft(W4)
            Q1 = self.block_diagonal_fft(Q)
            I5 = torch.eye(left.shape[0])
            def optimize_Q(P, L, D, mu, W44, lambda3, Q, lambda_val, num_iterations, learning_rate):
                for i in range(num_iterations):
                    # 计算梯度
                    # grad_Q = lambda3 * 2 * Q @ L - mu * (P - Q + W44 / mu)
                    # grad_Q = lambda3 * 2 * Q @ L - mu * (P - Q + W44 / mu) + lambda_val * 2 * Q @ D
                    grad_Q = lambda3 * 2 * (Q @ L).detach() - mu * ((P - Q + W44 / mu).detach()) + lambda_val * 2 * (Q @ D).detach()
                    # 2. 对 lambda 的梯度
                    grad_lambda = torch.trace(Q @ D @ Q.T) - 1
                    # 更新 Q 和 lambda
                    Q.data = Q.data - learning_rate * grad_Q
                    lambda_val.data = lambda_val.data - learning_rate * grad_lambda
                    #正则化 Q 以确保约束条件
                    constraint = torch.trace(Q @ D @ Q.T)
                    if constraint != 1:
                        Q.data = Q.data * (1 / torch.sqrt(constraint))  # 重新缩放 Q 使得约束条件满
                    # 每 10 次迭代打印一次目标函数和约束条件
                    if i % 5 == 0:
                        term1 = lambda3 * torch.trace(Q @ L @ Q.T)
                        term2 = (mu / 2) * torch.norm(P - Q + W44 / mu) ** 2
                        constraint = torch.trace(Q @ D @ Q.T)
                        lagrangian = term1 + term2 + lambda_val * (constraint - 1)
                        print(torch.norm(P - Q + W44 / mu) ** 2, mu / 2)
                        print(f"Iteration {i}, Objective: {lagrangian.item()}")
                        print(f"Constraint: {constraint.item()}")
                return Q, lambda_val

            lambda_val = torch.tensor(1.0).float()
            left = left.to(torch.float32)
            right = right.to(torch.float32)
            Q_new_diag, lambda_optimized = optimize_Q(P1, left, right, mu, W44, gama, Q1, lambda_val,
                                                      num_iterations=20,
                                                      learning_rate=0.01)
            Q_new = torch.zeros((r, l, n))
            for i in range(n):
                Q_new[:, :, i] = Q_new_diag[i * r:(i + 1) * r, i * l:(i + 1) * l]
            Q_new = torch.fft.ifft(Q_new).real

            # update W1,W2,mu
            W1 += mu * (self.T_product(P_new,X) - self.T_product(P_new,L_new) - E_new)
            W2 += mu * (L_new - M_new)
            W3 += mu * (self.T_product(W_new,P_new) - G_new)
            W4 += mu * (P_new - Q_new)
            mu = min(rho * mu, mu_max)

            term1 = self.T_product(P_new, X) - self.T_product(P_new, L_new) - E_new
            term2 = L_new - M_new
            term3 = self.T_product(W_new, P_new) - G_new
            term4 = P_new - Q_new

            res1 = torch.norm(term1).item()
            res2 = torch.norm(term2).item()
            res3 = torch.norm(term3).item()
            res4 = torch.norm(term4).item()
            total_residual = res1 + res2 + res3 + res4
            loss.append(total_residual)
            print(f"Iter {iters}: Total Residual={total_residual:.6f}, mu={mu:.2e}")
            if self.converged(M, L, E, P, W, G, X, M_new, L_new, E_new,P_new,W_new,G_new) or iters >= max_iters:
                return M_new, L_new, E_new, P_new, W_new,G_new,Q_new,loss
            else:
                M, L, E, P, W, G, Q= M_new, L_new, E_new, P_new, W_new,G_new,Q_new
                torch.set_printoptions(precision=12)
                print(iters, torch.max(self.T_product(P,X) - self.T_product(P,L) - E))
                print(iters, torch.max(L - M))
                print(iters, torch.max(self.T_product(W,P) - G))
                print(iters, torch.max(P - Q))
                print(iters,mu)

if __name__ =='__main__':
    # for lunshu in range(1, 2):
    #     # R = loadmat('./Indian_pines_16Classes.mat')['R']
    #     # image = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines.mat')['indian_pines_corrected']
    #     # label = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
    #     image = loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
    #     label = loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
    #     # label = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_gt']
    #     # image = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_corrected']
    #     spy.imshow(classes=label)  # 确保 label 是 2D 整数数组
    #     plt.axis('off')
    #     plt.savefig('/data/LOH/TSPLL_label/longkou-gt.jpg', dpi=300, bbox_inches='tight',interpolation='none')
    #     plt.show()
    #     '''PCA'''
    #     # rows, cols = np.nonzero(label != 0)
    #     # coordinates = np.column_stack((rows, cols))
    #     # # fea = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['fea']
    #     # fea = np.zeros((coordinates.shape[0], image.shape[2]))
    #     # for i, j in enumerate(coordinates):
    #     #     fea[i, :] = image[j[0], j[1], :]
    #     # # 10249*200
    #     # pca = PCA(n_components=80)  # 降维到100
    #     # pca.fit(fea)
    #     # fea_reduced = pca.transform(fea)
    #     # x = np.zeros((label.shape[0], label.shape[1], 80))
    #     # for i in range(fea.shape[0]):
    #     #     a = coordinates[i][0]
    #     #     b = coordinates[i][1]
    #     #     x[a][b][:] = fea_reduced[i][:]
    #     # image = x.astype(np.float32)

    #     '''add noise(0,0.5)零均值高斯噪声    随机生成胡椒和盐噪声的比例(0,0.3)'''
    #     image = image.astype(float)
    #     for band in range(image.shape[2]):
    #         image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
    #                     np.max(image[:, :, band]) - np.min(image[:, :, band]))
    #     np.random.seed(42)
    #     noisy_image = np.copy(image).astype(np.float64)
    #     selected_bands = np.random.choice(image.shape[2], size=30, replace=False)
    #     for i in selected_bands:
    #         variance = np.random.uniform(0, 0.5)
    #         noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
    #         noisy_image[..., i] += noise
    #         # 随机生成胡椒和盐噪声的比例(0,0.3)
    #         salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
    #         pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
    #         noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
    #         noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
    #     image = noisy_image


    #     # random_idx = np.load('/data/LOH/TSPLL_label/random_idx/random_idx.npy')
    #     random_idx_dict = np.load('/data/LOH/TSPLL_label/random_idx/random_idx_longkou_0.2%.npy', allow_pickle=True).item()
    #     random_idx = []
    #     for class_label, indices in random_idx_dict.items():
    #         random_idx.extend(indices['train_indices'])

    #     PATCH_SIZE = 9
    #     x_train, train_label, x_test, test_label ,train_label_list, test_label_list= train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
    #     ours = TRPCA()

    #     '''计算临近点'''
    #     # x_train_W, _, _, _ = train_test_tensor_half(PATCH_SIZE,random_idx, image, label)
    #     # l = len(x_train_W)
    #     # ci = []
    #     # b = x_train_W[0].shape[2]
    #     # for i in range(l):
    #     #     c_matrix = torch.zeros((b, b))
    #     #     xt = x_train_W[i]
    #     #     ui = torch.mean(xt, dim=(0, 1), keepdim=True)
    #     #     ui1 = ui.reshape(b, -1)
    #     #     for m in range(9):
    #     #         for n in range(9):
    #     #             xt1 = xt[m, n, :].reshape(b, -1)
    #     #             c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
    #     #     c_matrix = c_matrix / 80
    #     #     ci.append(c_matrix)
    #     # t = 1000
    #     # k_near = 10
    #     # w, d = dist_EMD(x_train_W, ci, k_near, t)
    #     # # np.save('w-1000-indian', w)
    #     # # np.save('d-1000-indian', d)
    #     # X_train_fft_list = []
    #     # for i in range(x_train.shape[1]):
    #     #     x = x_train[:, i, :].reshape(b, 1, 81)
    #     #     x =ours.block_diagonal_fft(x)
    #     #     X_train_fft_list.append(x)

    #     # left ,right= ours.getyi_yj(X_train_fft_list, w, d)  # (9,9)
    #     # left = torch.from_numpy(left)
    #     # right = torch.from_numpy(right)

    #     #普通欧式
    #     N_train = x_train.shape[1]
    #     b = x_train.shape[0]  # 获取波段数，后面 reshape 要用
    #     x_flat = x_train.permute(1, 0, 2).reshape(N_train, -1).numpy()
    #     # 2. 计算欧氏距离矩阵
    #     from scipy.spatial.distance import pdist, squareform
    #     # 计算两两之间的欧氏距离
    #     dist_mat = squareform(pdist(x_flat, metric='euclidean'))
    #     # 3. 构建权重矩阵 W (使用高斯核函数)
    #     # sigma 使用距离的平均值自适应
    #     sigma = np.mean(dist_mat)
    #     w = np.exp(-dist_mat**2 / (2 * sigma**2))
    #     np.fill_diagonal(w, 0)  # 对角线置0（自己和自己无连接）

    #     # 4. KNN 稀疏化 (保留 K 个权重最大的邻居)
    #     k_near = 10
    #     for i in range(N_train):
    #         # argsort 从小到大排，取最后 k 个即为最大的权重
    #         topk_indices = np.argsort(w[i])[-k_near:]
    #         # 创建掩码，非 Top-K 的位置置为 0
    #         mask = np.zeros_like(w[i], dtype=bool)
    #         mask[topk_indices] = True
    #         w[i] = w[i] * mask

    #     # 保证图是对称的
    #     w = (w + w.T) / 2

    #     # 5. 计算度矩阵 D (对角阵)
    #     # D_ii = sum(W_ij)
    #     d = np.diag(np.sum(w, axis=1))

    #     # --- 以下保持您原本的后续逻辑不变 ---
    #     X_train_fft_list = []
    #     for i in range(x_train.shape[1]):
    #         x = x_train[:, i, :].reshape(b, 1, 81)
    #         x = ours.block_diagonal_fft(x)
    #         X_train_fft_list.append(x)
    #     left ,right= ours.getyi_yj(X_train_fft_list, w, d)  # (9,9)
    #     left = torch.from_numpy(left)
    #     right = torch.from_numpy(right)


    #     M, L, E, P, W, G, Q= ours.ADMM(left,right,x_train,train_label)
    #     X_train_reduced = ours.T_product(P, x_train)
    #     X_train_reduced1 = ours.T_product(P, L)
    #     X_train_reduced_Q = ours.T_product(Q, x_train)
    #     X_train_reduced_Q1 = ours.T_product(Q, L)
    #     X_test_reduced = ours.T_product(P, x_test)
    #     X_test_reduced_Q = ours.T_product(Q, x_test)

    #     def nn_unique(x_train, train_label, x_test, test_label, random, label):
    #         computedClass = []
    #         D = np.zeros((x_test.shape[1], x_train.shape[1]))
    #         # 计算距离矩阵
    #         for i in range(x_test.shape[1]):
    #             current_block = x_test[:, i, :]
    #             for j in range(x_train.shape[1]):
    #                 neighbor_block = x_train[:, j, :]
    #                 w = current_block - neighbor_block
    #                 d = torch.linalg.norm(w)
    #                 D[i, j] = d

    #         # 获取每个测试样本的最小距离邻居
    #         id = np.argsort(D, axis=1)
    #         computedClass.append(np.array(train_label)[id[:, 0]])

    #         # 将预测类别加入到test_label中（更新标签）
    #         updated_test_label = np.copy(test_label)  # 复制原始标签以保留原始信息
    #         updated_test_label[:] = computedClass[0]  # 将计算得到的预测标签填入更新后的标签中

    #         rows, cols = np.nonzero(label != 0)
    #         coordinates = np.column_stack((rows, cols))

    #         label_matrix = np.copy(label)
    #         idx_test = np.setdiff1d(range(len(coordinates)), random)

    #         # Process testing patches 可视化分类结果
    #         for test_idx, coord in enumerate(idx_test):
    #             i, j = coordinates[coord]
    #             label_matrix[i, j] = updated_test_label[test_idx]

    #         # 计算总精度 OA
    #         total_correct = np.sum(updated_test_label == test_label)
    #         precision_OA = total_correct / x_test.shape[1]

    #         # 计算每个类别的精度
    #         unique_classes = np.unique(train_label)
    #         class_precision = {}

    #         for cls in unique_classes:
    #             # 获取该类别在测试集中的索引
    #             test_cls_indices = np.where(test_label == cls)[0]
    #             if len(test_cls_indices) == 0:
    #                 continue  # 如果没有该类别的测试样本，跳过

    #             correct_count = 0
    #             for idx in test_cls_indices:
    #                 # 比较预测标签和真实标签
    #                 if updated_test_label[idx] == cls:
    #                     correct_count += 1

    #             # 计算该类别的精度
    #             precision = correct_count / len(test_cls_indices)
    #             class_precision[cls] = precision

    #         # 计算平均精度 (AA)
    #         precision_AA = np.mean(list(class_precision.values()))

    #         # 计算 Cohen's Kappa
    #         # 混淆矩阵
    #         confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    #         class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

    #         for i in range(len(test_label)):
    #             actual_idx = class_to_index[test_label[i]]
    #             predicted_idx = class_to_index[updated_test_label[i]]
    #             confusion_matrix[actual_idx, predicted_idx] += 1

    #         total_samples = np.sum(confusion_matrix)
    #         P_o = np.trace(confusion_matrix) / total_samples  # 观测一致性
    #         P_e = np.sum(
    #             (np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples ** 2
    #         )  # 随机一致性
    #         precision_Kappa = (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0  # 防止分母为 0

    #         # 返回总精度和每类精度，及更新后的标签矩阵
    #         precision = {
    #             'total_precision': precision_OA,
    #             'class_precision': class_precision,
    #             'average_precision': precision_AA,
    #             'kappa': precision_Kappa
    #         }
    #         return precision, label_matrix


    #     acc,tup = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list, random_idx, label)
    #     # view2 = spy.imshow(classes=tup, title="acc-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/P*X.jpg')

    #     acc1,tup1 = nn_unique(X_train_reduced1, train_label_list, X_test_reduced, test_label_list, random_idx, label)
    #     # view3 = spy.imshow(classes=tup1, title="acc1-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/P*L.jpg')

    #     accQ,tupQ = nn_unique(X_train_reduced_Q, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
    #     # view4 = spy.imshow(classes=tupQ, title="accQ-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/Q*X.jpg')

    #     accQ1,tupQ1 = nn_unique(X_train_reduced_Q1, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
    #     # view5 = spy.imshow(classes=tupQ1, title="accQ1-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/longkou/lambda=1,1,0 no_graph/Q*L.jpg')
    #     # plt.pause(60)

    #     # print acc
    #     with open('/data/LOH/TSPLL_label/Classification maps/noise/longkou/result.txt', 'a', encoding='utf-8') as file:
    #         sys.stdout = file
    #         print(lunshu)
    #         print(f"P*X Total Precision: {acc['total_precision']* 100:.4f}")
    #         print(f"P*X average_precision: {acc['average_precision'] * 100:.4f}")
    #         print(f"P*X kappa: {acc['kappa'] * 100:.4f}")
    #         for cls, precision in acc['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")

    #         print(f"P*L Total Precision: {acc1['total_precision']* 100:.4f}")
    #         print(f"P*L average_precision: {acc1['average_precision'] * 100:.4f}")
    #         print(f"P*L kappa: {acc1['kappa'] * 100:.4f}")
    #         for cls, precision in acc1['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")

    #         print(f"Q*X Total Precision: {accQ['total_precision'] * 100:.4f}")
    #         print(f"Q*X average_precision: {accQ['average_precision'] * 100:.4f}")
    #         print(f"Q*X kappa: {accQ['kappa'] * 100:.4f}")
    #         for cls, precision in accQ['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")

    #         print(f"Q*L Total Precision: {accQ1['total_precision'] * 100:.4f}")
    #         print(f"Q*L average_precision: {accQ1['average_precision'] * 100:.4f}")
    #         print(f"Q*L kappa: {accQ1['kappa'] * 100:.4f}")
    #         for cls, precision in accQ1['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")
    #         sys.stdout = sys.__stdout__



    # for lunshu in range(1, 2):
    #     label = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_gt']
    #     image = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['salinas_corrected']
    #
    #     '''PCA'''
    #     rows, cols = np.nonzero(label != 0)
    #     coordinates = np.column_stack((rows, cols))
    #     # fea = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['fea']
    #     fea = np.zeros((coordinates.shape[0], image.shape[2]))
    #     for i, j in enumerate(coordinates):
    #         fea[i, :] = image[j[0], j[1], :]
    #     # 10249*200
    #     pca = PCA(n_components=80)  # 降维到100
    #     pca.fit(fea)
    #     fea_reduced = pca.transform(fea)
    #     x = np.zeros((label.shape[0], label.shape[1], 80))
    #     for i in range(fea.shape[0]):
    #         a = coordinates[i][0]
    #         b = coordinates[i][1]
    #         x[a][b][:] = fea_reduced[i][:]
    #     image = x.astype(np.float32)
    #
    #     '''add noise(0,0.5)零均值高斯噪声    随机生成胡椒和盐噪声的比例(0,0.3)'''
    #     # image = image.astype(float)
    #     # for band in range(image.shape[2]):
    #     #     image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
    #     #                 np.max(image[:, :, band]) - np.min(image[:, :, band]))
    #     np.random.seed(42)
    #     noisy_image = np.copy(image).astype(np.float64)
    #     selected_bands = np.random.choice(image.shape[2], size=30, replace=False)
    #     for i in selected_bands:
    #         variance = np.random.uniform(0, 0.5)
    #         noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
    #         noisy_image[..., i] += noise
    #         # 随机生成胡椒和盐噪声的比例(0,0.3)
    #         salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
    #         pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
    #         noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
    #         noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
    #     image = noisy_image
    #
    #     random_idx_dict = np.load('/data/LOH/TSPLL_label/random_idx/random_idx_salinas_1%.npy',allow_pickle=True).item()
    #     random_idx = []
    #     for class_label, indices in random_idx_dict.items():
    #         random_idx.extend(indices['train_indices'])
    #
    #     PATCH_SIZE = 9
    #     x_train, train_label, x_test, test_label ,train_label_list, test_label_list= train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
    #     ours = TRPCA()
    #
    #     '''计算临近点'''
    #     x_train_W, _, _, _ = train_test_tensor_half(PATCH_SIZE, random_idx, image, label)
    #     l = len(x_train_W)
    #     ci = []
    #     b = x_train_W[0].shape[2]
    #     for i in range(l):
    #         c_matrix = torch.zeros((b, b))
    #         xt = x_train_W[i]
    #         ui = torch.mean(xt, dim=(0, 1), keepdim=True)
    #         ui1 = ui.reshape(b, -1)
    #         for m in range(9):
    #             for n in range(9):
    #                 xt1 = xt[m, n, :].reshape(b, -1)
    #                 c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
    #         c_matrix = c_matrix / 80
    #         ci.append(c_matrix)
    #     t = 1000
    #     k_near = 10
    #     w, d = dist_EMD(x_train_W, ci, k_near, t)
    #     # np.save('w-1000-indian', w)
    #     # np.save('d-1000-indian', d)
    #     X_train_fft_list = []
    #     for i in range(x_train.shape[1]):
    #         x = x_train[:, i, :].reshape(b, 1, 81)
    #         x =ours.block_diagonal_fft(x)
    #         X_train_fft_list.append(x)
    #
    #     left ,right= ours.getyi_yj(X_train_fft_list, w, d)  # (9,9)
    #     # np.save('left-1000-indian',left)
    #     # np.save('right-1000-indian',right)
    #     left = torch.from_numpy(left)
    #     right = torch.from_numpy(right)
    #
    #     M, L, E, P, W, G, Q= ours.ADMM(left,right,x_train,train_label)
    #     X_train_reduced = ours.T_product(P, x_train)
    #     X_train_reduced1 = ours.T_product(P, L)
    #     X_train_reduced_Q = ours.T_product(Q, x_train)
    #     X_train_reduced_Q1 = ours.T_product(Q, L)
    #     X_test_reduced = ours.T_product(P, x_test)
    #     X_test_reduced_Q = ours.T_product(Q, x_test)
    #
    #     def nn_unique(x_train, train_label, x_test, test_label, random, label):
    #         computedClass = []
    #         D = np.zeros((x_test.shape[1], x_train.shape[1]))
    #         # 计算距离矩阵
    #         for i in range(x_test.shape[1]):
    #             current_block = x_test[:, i, :]
    #             for j in range(x_train.shape[1]):
    #                 neighbor_block = x_train[:, j, :]
    #                 w = current_block - neighbor_block
    #                 d = torch.linalg.norm(w)
    #                 D[i, j] = d
    #
    #         # 获取每个测试样本的最小距离邻居
    #         id = np.argsort(D, axis=1)
    #         computedClass.append(np.array(train_label)[id[:, 0]])
    #
    #         # 将预测类别加入到test_label中（更新标签）
    #         updated_test_label = np.copy(test_label)  # 复制原始标签以保留原始信息
    #         updated_test_label[:] = computedClass[0]  # 将计算得到的预测标签填入更新后的标签中
    #
    #         rows, cols = np.nonzero(label != 0)
    #         coordinates = np.column_stack((rows, cols))
    #
    #         label_matrix = np.copy(label)
    #         idx_test = np.setdiff1d(range(len(coordinates)), random)
    #
    #         # Process testing patches 可视化分类结果
    #         for test_idx, coord in enumerate(idx_test):
    #             i, j = coordinates[coord]
    #             label_matrix[i, j] = updated_test_label[test_idx]
    #
    #         # 计算总精度 OA
    #         total_correct = np.sum(updated_test_label == test_label)
    #         precision_OA = total_correct / x_test.shape[1]
    #
    #         # 计算每个类别的精度
    #         unique_classes = np.unique(train_label)
    #         class_precision = {}
    #
    #         for cls in unique_classes:
    #             # 获取该类别在测试集中的索引
    #             test_cls_indices = np.where(test_label == cls)[0]
    #             if len(test_cls_indices) == 0:
    #                 continue  # 如果没有该类别的测试样本，跳过
    #
    #             correct_count = 0
    #             for idx in test_cls_indices:
    #                 # 比较预测标签和真实标签
    #                 if updated_test_label[idx] == cls:
    #                     correct_count += 1
    #
    #             # 计算该类别的精度
    #             precision = correct_count / len(test_cls_indices)
    #             class_precision[cls] = precision
    #
    #         # 计算平均精度 (AA)
    #         precision_AA = np.mean(list(class_precision.values()))
    #
    #         # 计算 Cohen's Kappa
    #         # 混淆矩阵
    #         confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    #         class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    #
    #         for i in range(len(test_label)):
    #             actual_idx = class_to_index[test_label[i]]
    #             predicted_idx = class_to_index[updated_test_label[i]]
    #             confusion_matrix[actual_idx, predicted_idx] += 1
    #
    #         total_samples = np.sum(confusion_matrix)
    #         P_o = np.trace(confusion_matrix) / total_samples  # 观测一致性
    #         P_e = np.sum(
    #             (np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples ** 2
    #         )  # 随机一致性
    #         precision_Kappa = (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0  # 防止分母为 0
    #
    #         # 返回总精度和每类精度，及更新后的标签矩阵
    #         precision = {
    #             'total_precision': precision_OA,
    #             'class_precision': class_precision,
    #             'average_precision': precision_AA,
    #             'kappa': precision_Kappa
    #         }
    #         return precision, label_matrix
    #
    #
    #     acc,tup = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list, random_idx, label)
    #     # view2 = spy.imshow(classes=tup, title="acc-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/Salinas/lambda=1,1,0 no_graph/P*X.jpg')
    #
    #     acc1,tup1 = nn_unique(X_train_reduced1, train_label_list, X_test_reduced, test_label_list, random_idx, label)
    #     # view3 = spy.imshow(classes=tup1, title="acc1-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/Salinas/lambda=1,1,0 no_graph/P*L.jpg')
    #
    #     accQ,tupQ = nn_unique(X_train_reduced_Q, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
    #     # view4 = spy.imshow(classes=tupQ, title="accQ-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/Salinas/lambda=1,1,0 no_graph/Q*X.jpg')
    #
    #     accQ1,tupQ1 = nn_unique(X_train_reduced_Q1, train_label_list, X_test_reduced_Q, test_label_list, random_idx, label)
    #     # view5 = spy.imshow(classes=tupQ1, title="accQ1-gt")
    #     # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/Salinas/lambda=1,1,0 no_graph/Q*L.jpg')
    #     # plt.pause(60)
    #
    #     # print acc
    #     with open('/data/LOH/TSPLL_label/Classification maps/noise/salinas/result.txt', 'a', encoding='utf-8') as file:
    #         sys.stdout = file
    #         print(lunshu)
    #         print(f"P*X Total Precision: {acc['total_precision']* 100:.4f}")
    #         print(f"P*X average_precision: {acc['average_precision'] * 100:.4f}")
    #         print(f"P*X kappa: {acc['kappa'] * 100:.4f}")
    #         for cls, precision in acc['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")
    #
    #         print(f"P*L Total Precision: {acc1['total_precision']* 100:.4f}")
    #         print(f"P*L average_precision: {acc1['average_precision'] * 100:.4f}")
    #         print(f"P*L kappa: {acc1['kappa'] * 100:.4f}")
    #         for cls, precision in acc1['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")
    #
    #         print(f"Q*X Total Precision: {accQ['total_precision'] * 100:.4f}")
    #         print(f"Q*X average_precision: {accQ['average_precision'] * 100:.4f}")
    #         print(f"Q*X kappa: {accQ['kappa'] * 100:.4f}")
    #         for cls, precision in accQ['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")
    #
    #         print(f"Q*L Total Precision: {accQ1['total_precision'] * 100:.4f}")
    #         print(f"Q*L average_precision: {accQ1['average_precision'] * 100:.4f}")
    #         print(f"Q*L kappa: {accQ1['kappa'] * 100:.4f}")
    #         for cls, precision in accQ1['class_precision'].items():
    #             print(f"Class {cls} Precision: {precision * 100:.4f}")
    #         sys.stdout = sys.__stdout__

    for lunshu in range(1, 2):
        image = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines.mat')['indian_pines_corrected']
        label = loadmat('/data/LOH/TSPLL_label/dataset/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        # view1 = spy.imshow(classes=label, title="gt")
        # plt.savefig('/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/indian_pines/lambda=1,1,0 no_graph/gt.jpg')
        '''PCA'''
        rows, cols = np.nonzero(label != 0)
        coordinates = np.column_stack((rows, cols))
        # fea = loadmat('/data/LOH/TSPLL_label/dataset/Salinas/Salinas.mat')['fea']
        fea = np.zeros((coordinates.shape[0], image.shape[2]))
        for i, j in enumerate(coordinates):
            fea[i, :] = image[j[0], j[1], :]
        # 10249*200
        pca = PCA(n_components=80)  # 降维到100
        pca.fit(fea)
        fea_reduced = pca.transform(fea)
        x = np.zeros((label.shape[0], label.shape[1], 80))
        for i in range(fea.shape[0]):
            a = coordinates[i][0]
            b = coordinates[i][1]
            x[a][b][:] = fea_reduced[i][:]
        image = x.astype(np.float32)
    
        '''add noise(0,0.5)零均值高斯噪声    随机生成胡椒和盐噪声的比例(0,0.3)'''
        # image = image.astype(float)
        # for band in range(image.shape[2]):
        #     image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
        #             np.max(image[:, :, band]) - np.min(image[:, :, band]))
        # np.random.seed(42)
        # noisy_image = np.copy(image).astype(np.float64)
        # selected_bands = np.random.choice(image.shape[2], size=60, replace=False)
        # for i in selected_bands:
        #     variance = np.random.uniform(0, 0.5)
        #     noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
        #     noisy_image[..., i] += noise
        #     # 随机生成胡椒和盐噪声的比例(0,0.3)
        #     salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
        #     pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
        #     noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
        #     noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
        # image = noisy_image
    
        random_idx = np.load('/data/LOH/TSPLL_label/random_idx/random_idx.npy')
        PATCH_SIZE = 9
        x_train, train_label, x_test, test_label, train_label_list, test_label_list = train_test_tensor_fold(PATCH_SIZE,random_idx, image, label)
        ours = TRPCA()
    
    # #欧式构图用于test
    #     print("正在使用欧氏距离构建 KNN 图 (快速模式)...")
    #     # 1. 准备数据：将 x_train_W (Patch列表) 转换为特征矩阵
    #     # x_train_W[i] 形状通常是 (Patch_H, Patch_W, Bands)
    #     # 我们将其展平作为该样本的特征向量
    #     num_samples = len(x_train_W)
    #     features_list = []
    #     for i in range(num_samples):
    #         # 将 patch 展平为一维向量
    #         flat_feat = x_train_W[i].flatten() 
    #         features_list.append(flat_feat)
    #     features = np.array(features_list) # 形状: (样本数, 特征维度)

    #     # 2. 计算两两欧氏距离矩阵
    #     from scipy.spatial.distance import pdist, squareform
    #     dist_mat = squareform(pdist(features, metric='euclidean'))
    #     k_near = 10 # 邻居数
    #     w = np.zeros((num_samples, num_samples))
    #     sigma = np.mean(dist_mat) 
    #     for i in range(num_samples):
    #         sorted_indices = np.argsort(dist_mat[i])
    #         neighbors = sorted_indices[1:k_near+1]
    #         for j in neighbors:
    #             dist = dist_mat[i, j]
    #             weight = np.exp(-(dist**2) / (2 * sigma**2))
    #             w[i, j] = weight
    #             w[j, i] = weight # 保持对称性
    #     d = np.diag(np.sum(w, axis=1))
    #     print("构图完成。")

        # '''计算临近点'''
        # x_train_W, _, _, _ = train_test_tensor_half(PATCH_SIZE,random_idx, image, label)
        # l = len(x_train_W)
        # ci = []
        # b = x_train_W[0].shape[2]
        # for i in range(l):
        #     c_matrix = torch.zeros((b, b))
        #     xt = x_train_W[i]
        #     ui = torch.mean(xt, dim=(0, 1), keepdim=True)
        #     ui1 = ui.reshape(b, -1)
        #     for m in range(9):
        #         for n in range(9):
        #             xt1 = xt[m, n, :].reshape(b, -1)
        #             c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
        #     c_matrix = c_matrix / 80
        #     ci.append(c_matrix)
        # t = 1000
        # k_near = 10
        # w, d = dist_EMD(x_train_W, ci, k_near, t)
        # # np.save('w-1000-indian', w)
        # # np.save('d-1000-indian', d)
        # X_train_fft_list = []
        # for i in range(x_train.shape[1]):
        #     x = x_train[:, i, :].reshape(b, 1, 81)
        #     x = ours.block_diagonal_fft(x)
        #     X_train_fft_list.append(x)
    
        # left, right = ours.getyi_yj(X_train_fft_list, w, d)  # (9,9)
        # np.save('left-1000-indian',left)
        # np.save('right-1000-indian',right)
        left = np.load('left-1000-indian.npy')
        right = np.load('right-1000-indian.npy')
        left = torch.from_numpy(left)
        right = torch.from_numpy(right)
    
        M, L, E, P, W, G, Q, loss_list = ours.ADMM(left, right, x_train, train_label)
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
    
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=2, color='red', marker='o', markersize=3)
        plt.xlabel('Iterations')
        plt.ylabel('Total Residual (Constraint Violation)')
        plt.title('Convergence Analysis of TSPLL (Fold Version)')
        plt.grid(True)
        # 保存图片到当前目录
        plt.savefig('/data/LOH/TSPLL_label/Classification maps/convergence/convergence_TSPLL_residuals.png')
        plt.show()
        print("收敛图已保存为 'convergence_TSPLL_residuals.png'")

        acc, tup = nn_unique(X_train_reduced, train_label_list, X_test_reduced, test_label_list, random_idx, label)
        # view2 = spy.imshow(classes=tup, title="acc-gt")
        # plt.savefig(
        #     '/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/indian_pines/lambda=1,1,0 no_graph/P*X.jpg')
    
        acc1, tup1 = nn_unique(X_train_reduced1, train_label_list, X_test_reduced, test_label_list, random_idx,label)
        # view3 = spy.imshow(classes=tup1, title="acc1-gt")
        # plt.savefig(
        #     '/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/indian_pines/lambda=1,1,0 no_graph/P*L.jpg')
    
        accQ, tupQ = nn_unique(X_train_reduced_Q, train_label_list, X_test_reduced_Q, test_label_list, random_idx,
                               label)
        # view4 = spy.imshow(classes=tupQ, title="accQ-gt")
        # plt.savefig(
        #     '/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/indian_pines/lambda=1,1,0 no_graph/Q*X.jpg')
    
        accQ1, tupQ1 = nn_unique(X_train_reduced_Q1, train_label_list, X_test_reduced_Q, test_label_list,
                                 random_idx, label)
        # view5 = spy.imshow(classes=tupQ1, title="accQ1-gt")
        # plt.savefig(
        #     '/data/LOH/TSPLL_label/Classification maps/Ablation Analysis/indian_pines/lambda=1,1,0 no_graph/Q*L.jpg')
        # plt.pause(60)
    
        # print acc
        with open(
                '/data/LOH/TSPLL_label/Classification maps/convergence/result.txt',
                'a', encoding='utf-8') as file:
            sys.stdout = file
            print(lunshu)
            print(f"P*X Total Precision: {acc['total_precision'] * 100:.4f}")
            print(f"P*X average_precision: {acc['average_precision'] * 100:.4f}")
            print(f"P*X kappa: {acc['kappa'] * 100:.4f}")
            for cls, precision in acc['class_precision'].items():
                print(f"Class {cls} Precision: {precision * 100:.4f}")
    
            print(f"P*L Total Precision: {acc1['total_precision'] * 100:.4f}")
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