import pickle
import numpy as np
import spectral
import torch
import math
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wasserstein_distance
def Patch(data,H,W,PATCH_SIZE):
    transpose_array=np.transpose(data,(2,0,1)) #(C,H,W)
    height_slice=slice(H,H+PATCH_SIZE)
    width_slice=slice(W,W+PATCH_SIZE)
    patch=transpose_array[:,height_slice,width_slice]
    return np.array(patch)

def dist_inter(c,x,k,t,Band):#x:train
    n=len(x)
    alpha_inter = 1
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))
    '''inner_similarity_mean'''
    x_inner_similarity_mean = []
    for ni in range(n):
        reshaped_x = x[ni].reshape(81, Band)
        x_inner_similarity = np.zeros((81,81))
        for i in range(reshaped_x.shape[0]):
            for j in range(i+1,reshaped_x.shape[0]):
                x_inner_similarity[i, j] = np.exp(-(math.pow(np.linalg.norm(reshaped_x[i]-reshaped_x[j]),2))/t)
        'block-mean'
        x_inner_similarity = x_inner_similarity + x_inner_similarity.T
        for i in range(81):
            x_inner_similarity[i,i] = 1
        x_inner_similarity_mean.append(np.mean(x_inner_similarity))

    for i in range(n):
        xi = np.array(c[i])
        si = x_inner_similarity_mean[i]
        for j in range(i+1,n):
            xj = np.array(c[j])
            sj = x_inner_similarity_mean[j]
            sij = np.exp(-(math.pow(np.linalg.norm(xi - xj), 2)) / t)
            s[i,j] = alpha_inter * sij + (1 - alpha_inter) * (si + sj) / 2
            # s[i,j]=np.exp(-(math.pow(np.linalg.norm(xi-xj),2))/t)
    '''knn'''
    s=s+s.T

    for i in range(n):
        s[i,i]=s[i,i]/2
        index_=np.argsort(s[i])[-(k):]
        w[i,index_]=s[i,index_]
        w[index_,i]=s[index_,i]

    '''D'''
    for i in range(n):
        d[i,i]=sum(w[i,:])
    return w,d
def dist(x,k,t):
    n=len(x)
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))

    # for i in range(n):
    #     features_i = x[i]
    #     for j in range(i+1,n):
    #             features_j = x[j]
    #             q = EMD(features_i, features_j)
    #             q = -(math.pow(q, 2)) / t# emd是计算EMD的函数
    #             s[i, j] = np.exp(q)
    for i in range(n):
        xi = np.array(x[i])
        #log_Ci=log_euclidean(xi)
        for j in range(i+1,n):
            xj = np.array(x[j])
            #log_Cj=log_euclidean(xj)
            s[i,j]=np.exp(-(math.pow(np.linalg.norm(xi-xj),2))/t)
    '''knn'''
    s=s+s.T

    for i in range(n):
        s[i,i]=s[i,i]/2
        index_=np.argsort(s[i])[-(k):]
        w[i,index_]=s[i,index_]
        w[index_,i]=s[index_,i]

    '''D'''
    for i in range(n):
        d[i,i]=sum(w[i,:])
    return w,d
def dist_EMD(x,c,k,t):
    n=len(x)
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))

    for i in range(n):
        xi = np.array(c[i])
        for j in range(i+1,n):
            xj = np.array(c[j])
            q = EMD(x[i], x[j])
            p = np.linalg.norm(xi - xj)
            p = -(math.pow(p, 2)) / t
            p = np.exp(p)
            s[i, j] = (p+(1/(1+q)))/2
            # s[i, j] = 1 / (1 + q)
            # q = -(math.pow(q, 2)) / t# emd是计算EMD的函数
            # s[i, j] = np.exp(q)

    # for i in range(n):
    #     xi = np.array(x[i])
    #     #log_Ci=log_euclidean(xi)
    #     for j in range(i+1,n):
    #         xj = np.array(x[j])
    #         #log_Cj=log_euclidean(xj)
    #         s[i,j]=np.exp(-(math.pow(np.linalg.norm(xi-xj),2))/t)
    '''knn'''
    s=s+s.T

    for i in range(n):
        s[i,i]=s[i,i]/2
        index_=np.argsort(s[i])[-(k):]
        w[i,index_]=s[i,index_]
        w[index_,i]=s[index_,i]

    '''D'''
    for i in range(n):
        d[i,i]=sum(w[i,:])
    return w,d
def fold(matrix, mode, shape):
    """ Fold a 2D array into a N-dimensional array."""
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = torch.from_numpy(np.moveaxis(np.reshape(matrix, full_shape), 0, mode))
    return tensor

def unfold(tensor,mode):
    """ Unfolds N-dimensional array into a 2D array."""
    t2=tensor.transpose(mode,0)
    matrix = t2.reshape(t2.shape[0], -1)
    return matrix

def kmode_product(tensor,matrix,mode):
    """ Mode-n product of a N-dimensional array with a matrix."""
    ori_shape=list(tensor.shape)
    new_shape=ori_shape
    new_shape[mode-1]=matrix.shape[0]
    result=fold(np.dot(matrix,unfold(tensor,mode-1)),mode-1,tuple(new_shape))
    return result
def getyi_yj(Y,W):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        for j in range(i+1,l):
            if W[i][j]!=0:
                re=re+np.dot((Y[i]-Y[j]),(Y[i]-Y[j]).T)*W[i][j]*2
    return re

def getyy(Y,D):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        re=re+np.dot(Y[i],Y[i].T)*D[i][i]
    return re
def getvalvec(left,right,n_dims):
    eig_val, eig_vec = sp.linalg.eig(left,right)#np.linalg.pinv(right),left))    
    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    # print("eig_val:", eig_val[:1])
    j = 0
    while eig_val[j] < 1e-6:
        j+=1
    # print("j: ", j)
    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    # print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_] 
    return eig_vec_picked

def getU(newshape,k_near,X_train,P,Band,t):
    '''MTLPP'''
    l = len(X_train)
    ci = []
    d = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train[i]
        ui = torch.mean(xt,dim=(0,1),keepdim=True)
        ui1 = ui.reshape(d, -1)
        for m in range(9):
            for n in range(9):
                xt1 = xt[m,n,:].reshape(d,-1)
                c_matrix = c_matrix + torch.matmul(xt1-ui1,(xt1-ui1).T)
        c_matrix = c_matrix / 80
        ci.append(c_matrix)
    #
    w,d=dist(ci,k_near,t)#计算临近点


    U1,U2,U3=np.eye(newshape[0],P),np.eye(newshape[1],P),np.eye(newshape[2],Band)
    U1=U1.astype(np.float64)
    U2=U2.astype(np.float64)
    U3=U3.astype(np.float64)
    t_max=5
    l=len(X_train)
    for t in range(t_max):
        y1,y2,y3=[],[],[]
        for i in range(l):
            y=kmode_product(X_train[i],U2,2)
            y=kmode_product(y,U3,3)
            y1.append(unfold(y,0))
        left=getyi_yj(y1,w)  #(9,9)
        right=getyy(y1,d)
        newu1=getvalvec(left,right,newshape[0])
        # print(newu1.dtype)
        
        lie=newu1.shape[1]
        for i in range(lie):
            newu1[:,i]=newu1[:,i]/np.linalg.norm(newu1[:,i])
        U1=newu1.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U3,3)
            y2.append(unfold(y,1))
        left=getyi_yj(y2,w)  #(9,9)
        right=getyy(y2,d)
        newu2=getvalvec(left,right,newshape[1])
        lie=newu2.shape[1]
        for i in range(lie):
            newu2[:,i]=newu2[:,i]/np.linalg.norm(newu2[:,i])
        U2=newu2.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U2,2)
            y3.append(unfold(y,2))
        left=getyi_yj(y3, w)
        right=getyy(y3,d)
        newu3=getvalvec(left,right,newshape[2])
        lie=newu3.shape[1]
        for i in range(lie):
            newu3[:,i]=newu3[:,i]/np.linalg.norm(newu3[:,i])
        U3=newu3.real.T
    return U1,U2,U3


def getU_EMD(newshape, k_near, X_train, P, Band, t):
    '''MTLPP'''
    l = len(X_train)
    ci = []
    d = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train[i]
        ui = torch.mean(xt,dim=(0,1),keepdim=True)
        ui1 = ui.reshape(d, -1)
        for m in range(9):
            for n in range(9):
                xt1 = xt[m,n,:].reshape(d,-1)
                c_matrix = c_matrix + torch.matmul(xt1-ui1,(xt1-ui1).T)
        c_matrix = c_matrix / 80
        ci.append(c_matrix)
    #
    w, d = dist_EMD(X_train,ci, k_near, t)  # 计算临近点

    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        # print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def getU_multiscalel(newshape, k_near, X_train,X_train1,X_train2, P, Band,t):
    '''MTLPP   9*9*200'''
    l = len(X_train)
    ci = []
    b = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((b, b))
        xt = X_train[i]
        ui = torch.mean(xt, dim=(0, 1), keepdim=True)
        ui1 = ui.reshape(b, -1)
        for m in range(9):
            for n in range(9):
                xt1 = xt[m, n, :].reshape(b, -1)
                c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
        c_matrix = c_matrix / 80
        ci.append(c_matrix)
    '''MTLPP    5*5*200'''
    l = len(X_train1)
    ci1 = []
    b = X_train1[0].shape[2]
    for i in range(l):
        c_matrix1 = torch.zeros((b, b))
        xt = X_train1[i]
        ui = torch.mean(xt, dim=(0, 1), keepdim=True)
        ui1 = ui.reshape(b, -1)
        for m in range(5):
            for n in range(5):
                xt1 = xt[m, n, :].reshape(b, -1)
                c_matrix1 = c_matrix1 + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
        c_matrix1 = c_matrix1 / 24
        ci1.append(c_matrix1)

    '''MTLPP    11*11*200'''
    l = len(X_train2)
    ci2 = []
    b = X_train2[0].shape[2]
    for i in range(l):
        c_matrix2 = torch.zeros((b, b))
        xt = X_train2[i]
        ui = torch.mean(xt, dim=(0, 1), keepdim=True)
        ui1 = ui.reshape(b, -1)
        for m in range(11):
            for n in range(11):
                xt1 = xt[m, n, :].reshape(b, -1)
                c_matrix2 = c_matrix2 + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
        c_matrix2 = c_matrix2 / 120
        ci2.append(c_matrix2)
    # with open("c-mul.pkl", "wb") as file:  # ci1 5*5
    #     pickle.dump(c, file)
    # # with open("ci1-2.pkl", "wb") as file:#ci1 5*5
    # #     pickle.dump(ci1, file)
    # # with open("ci2-2.pkl", "wb") as file:#ci1 11*11
    # #     pickle.dump(ci2, file)
    # # with open("ci-2.pkl", "rb") as file:
    # #     ci = pickle.load(file)
    # # with open("ci1-2.pkl", "rb") as file:
    # #     ci1 = pickle.load(file)
    # # with open("ci2-2.pkl", "rb") as file:
    # #     ci2 = pickle.load(file)
    c=[]
    for i in range(l):
        c.append((ci[i]+ci1[i]+ci2[i])/3)
    # with open("c-mul-paviau.pkl", "wb") as file:
    #     pickle.dump(c, file)

    w, d = dist(c, k_near, t) # 计算临近点
    np.save('d-multiscale-ZY-HHK-6.npy', d)
    np.save('w-multiscale-ZY-HHK-6.npy', w)
    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        # print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def getU_inter(newshape, k_near, X_train, X_train1,X_train2,P, Band,t):
    '''weight   9*9*200'''
    l = len(X_train)
    ci = []
    d = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train[i]
        xt = xt.reshape(81, d)
        we = np.zeros((81, 81))
        for m in range(81):
            for n in range(m,81):
                we[m,n] = np.exp(-(math.pow(np.linalg.norm(xt[m] - xt[n]), 2)) / 2)
        we = we+we.T
        np.fill_diagonal(we, 1)
        for m in range(81):
            for n in range(81):
                x_i = xt[m].reshape(d, -1)
                x_j = xt[n].reshape(d, -1)
                c_matrix = c_matrix + (torch.matmul(x_i - x_j, (x_i - x_j).T))*we[m,n]
                # c_matrix = c_matrix + torch.matmul(x_i - x_j, (x_i - x_j).T)
        c_matrix = c_matrix / (2*81*81)
        ci.append(c_matrix)

    '''weight   5*5*200'''
    l = len(X_train1)
    ci1 = []
    d = X_train1[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train1[i]
        xt = xt.reshape(25, d)
        we = np.zeros((25, 25))
        for m in range(25):
            for n in range(m,25):
                we[m,n] = np.exp(-(math.pow(np.linalg.norm(xt[m] - xt[n]), 2)) / 2)
        we = we+we.T
        np.fill_diagonal(we, 1)
        for m in range(25):
            for n in range(25):
                x_i = xt[m].reshape(d, -1)
                x_j = xt[n].reshape(d, -1)
                c_matrix = c_matrix + (torch.matmul(x_i - x_j, (x_i - x_j).T))*we[m,n]
                # c_matrix = c_matrix + torch.matmul(x_i - x_j, (x_i - x_j).T)
        c_matrix = c_matrix / (2*25*25)
        ci1.append(c_matrix)

    '''weight  11*11*200'''
    l = len(X_train2)
    ci2 = []
    d = X_train2[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train2[i]
        xt = xt.reshape(121, d)
        we = np.zeros((121, 121))
        for m in range(121):
            for n in range(m,121):
                we[m,n] = np.exp(-(math.pow(np.linalg.norm(xt[m] - xt[n]), 2)) / 2)
        we = we+we.T
        np.fill_diagonal(we, 1)
        for m in range(121):
            for n in range(121):
                x_i = xt[m].reshape(d, -1)
                x_j = xt[n].reshape(d, -1)
                c_matrix = c_matrix + (torch.matmul(x_i - x_j, (x_i - x_j).T))*we[m,n]
                # c_matrix = c_matrix + torch.matmul(x_i - x_j, (x_i - x_j).T)
        c_matrix = c_matrix / (2*121*121)
        ci2.append(c_matrix)

    #save weight-c
    # with open("ci2-0.pkl", "wb") as file:#ci1 11*11
    #     pickle.dump(ci2, file)
    # with open("ci-0.pkl", "rb") as file:
    #     ci = pickle.load(file)

    c=[]
    for i in range(l):
        c.append((ci[i]+ci1[i]+ci2[i])/3)
    w, d = dist_inter(c, X_train, k_near, t,Band) # 计算临近点
    np.save('w-weight-mult-ZY-HHK-6.npy', w)
    np.save('d-weight-mult-ZY-HHK-6.npy', d)
    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        # print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def getU_weight(newshape, k_near, X_train, X_train1,X_train2,P, Band,t):
    '''weight  9*9*200'''
    l = len(X_train)
    ci = []
    b = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((b, b))
        xt = X_train[i]
        xt = xt.reshape(81, b)
        we = np.zeros((81, 81))
        for m in range(81):
            for n in range(m,81):
                we[m,n] = np.exp(-(math.pow(np.linalg.norm(xt[m] - xt[n]), 2)) / 2)
        we = we+we.T
        np.fill_diagonal(we, 1)
        for m in range(81):
            for n in range(81):
                x_i = xt[m].reshape(b, -1)
                x_j = xt[n].reshape(b, -1)
                c_matrix = c_matrix + (torch.matmul(x_i - x_j, (x_i - x_j).T))*we[m,n]
                # c_matrix = c_matrix + torch.matmul(x_i - x_j, (x_i - x_j).T)
        c_matrix = c_matrix / (2*81*81)
        ci.append(c_matrix)
    # #
    # with open("ci-paviau.pkl", "wb") as file:#ci1 5*5
    #     pickle.dump(ci, file)
    # with open("ci-0.pkl", "rb") as file:
    #     ci = pickle.load(file)

    w, d = dist_inter(ci, X_train, k_near, t,Band) # 计算临近点
    np.save('w-weight-ZY-HHK-6.npy', w)
    np.save('d-weight-ZY-HHK-6.npy', d)
    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        # print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def getU_new_mult(newshape, k_near, X_train, X_train1, X_train2,P, Band, t):
    '''MTLPP   9*9*200'''
    # l = len(X_train)
    # ci = []
    # d = X_train[0].shape[2]
    # for i in range(l):
    #     c_matrix = torch.zeros((d, d))
    #     xt = X_train[i]
    #     ui = torch.mean(xt, dim=(0, 1), keepdim=True)
    #     ui1 = ui.reshape(200, -1)
    #     for m in range(9):
    #         for n in range(9):
    #             xt1 = xt[m, n, :].reshape(200, -1)
    #             c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
    #     c_matrix = c_matrix / 80
    #     ci.append(c_matrix)
    #
    # '''MTLPP    5*5*200'''
    # l = len(X_train1)
    # ci1 = []
    # d = X_train1[0].shape[2]
    # for i in range(l):
    #     c_matrix1 = torch.zeros((d, d))
    #     xt = X_train1[i]
    #     ui = torch.mean(xt, dim=(0, 1), keepdim=True)
    #     ui1 = ui.reshape(200, -1)
    #     for m in range(5):
    #         for n in range(5):
    #             xt1 = xt[m, n, :].reshape(200, -1)
    #             c_matrix1 = c_matrix1 + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
    #     c_matrix1 = c_matrix1 / 24
    #     ci1.append(c_matrix1)
    #
    # '''MTLPP    11*11*200'''
    # ci2 = []
    # d = X_train2[0].shape[2]
    # for i in range(l):
    #     c_matrix2 = torch.zeros((d, d))
    #     xt = X_train2[i]
    #     ui = torch.mean(xt, dim=(0, 1), keepdim=True)
    #     ui1 = ui.reshape(200, -1)
    #     for m in range(11):
    #         for n in range(11):
    #             xt1 = xt[m, n, :].reshape(200, -1)
    #             c_matrix2 = c_matrix2 + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
    #     c_matrix2 = c_matrix2 / 120
    #     ci2.append(c_matrix2)
    #
    # cr = []
    # for i in range(l):
    #     cr.append((ci[i] + ci1[i] + ci2[i]) / 3)

    '''weight--c  9*9*200 '''
    # l = len(X_train)
    # c = []
    # d = X_train[0].shape[2]
    # for i in range(l):
    #     c_matrix_all = torch.zeros((d, d))
    #     xt = X_train[i]
    #     xt = xt.reshape(81, 200)
    #     # we = np.zeros((81, 81))
    #     # for m in range(81):
    #     #     for n in range(m,81):
    #     #         we[m,n] = np.exp(-(math.pow(np.linalg.norm(xt[m] - xt[n]), 2)) / 2)
    #     # we = we+we.T
    #     # np.fill_diagonal(we, 1)
    #     for m in range(81):
    #         for n in range(81):
    #             x_i = xt[m].reshape(200, -1)
    #             x_j = xt[n].reshape(200, -1)
    #             # c_matrix = c_matrix + (torch.matmul(x_i - x_j, (x_i - x_j).T))*we[m,n]
    #             c_matrix_all = c_matrix_all + torch.matmul(x_i - x_j, (x_i - x_j).T)
    #     c_matrix_all = c_matrix_all / (2*81*81)
    #     c_matrix_all = c_matrix_all * cr[i]
    #     c.append(c_matrix_all)


    # with open("c-new.pkl", "wb") as file:#ci1 5*5
    #     pickle.dump(c, file)
    # with open("ci2.pkl", "wb") as file:#ci1 11*11
    #     pickle.dump(ci2, file)


    with open("c-new.pkl", "rb") as file:
        c = pickle.load(file)
    # with open("ci1.pkl", "rb") as file:
    #     ci1 = pickle.load(file)
    # with open("ci2.pkl", "rb") as file:
    #     ci2 = pickle.load(file)

    w, d = dist_inter(c, X_train, k_near, t) # 计算临近点
    # np.save('w-weight-mult.npy', w)
    # np.save('d-weight-mult.npy', d)
    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def log_euclidean(matrix):
    # 计算特征分解
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 对对角矩阵 Sigma 中的每个元素取对数
    Sigma = np.log(eigenvalues)
    log_Sigma = np.diag(Sigma)
    # 计算 log(C)
    log_C = eigenvectors @ log_Sigma @ np.linalg.inv(eigenvectors)
    return log_C



def EMD(tensor1,tensor2):
    def tensor_block_to_histogram(tensor_block, num_bins=128):
        histograms = []
        for band in range(tensor_block.shape[2]):
            # 计算每个波段的直方图
            hist, _ = np.histogram(tensor_block[:, :, band], bins=num_bins, range=(0, 1), density=True)
            histograms.append(hist)

        # 拼接所有波段的直方图
        combined_histogram = np.concatenate(histograms)
        return combined_histogram

    def normalize_histogram(histogram):
        return histogram / np.sum(histogram)

    hist1 = tensor_block_to_histogram(tensor1)
    hist2 = tensor_block_to_histogram(tensor2)

    # 步骤 2: 归一化直方图
    normalized_hist1 = normalize_histogram(hist1)
    normalized_hist2 = normalize_histogram(hist2)

    # 步骤 3: 计算EMD
    emd_distance = wasserstein_distance(normalized_hist1, normalized_hist2)
    return emd_distance

def EMD_c(tensor1,tensor2):
    def covariance_to_histogram(cov_matrix, num_bins=1000):
        # 将协方差矩阵展平并归一化
        flat_cov = cov_matrix.flatten()
        hist, _ = np.histogram(flat_cov, bins=num_bins, range=(0, 1), density=True)
        return hist / np.sum(hist)  # 归一化为概率分布

    hist1 = covariance_to_histogram(tensor1)
    hist2 = covariance_to_histogram(tensor2)

    # 步骤 3: 计算EMD
    emd_distance = wasserstein_distance(hist1, hist2)
    return emd_distance