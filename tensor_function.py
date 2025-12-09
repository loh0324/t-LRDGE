import numpy as np
import spectral
import torch
import math
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier

def Patch(data,H,W,PATCH_SIZE):
    transpose_array=np.transpose(data,(2,0,1)) #(C,H,W)
    height_slice=slice(H,H+PATCH_SIZE)
    width_slice=slice(W,W+PATCH_SIZE)
    patch=transpose_array[:,height_slice,width_slice]
    return np.array(patch)

def dist(x,k,t):
    n=len(x)
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        #log_Ci=log_euclidean(x[i])
        for j in range(i+1,n):
            #log_Cj=log_euclidean(x[j])
            s[i,j]=np.exp(-(math.pow(np.linalg.norm(x[i]-x[j]),2))/t)
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
    """ Unfolds N-dimensional array into a 2D array. (tensor,0)(tensor,2)"""
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
    print("eig_val:", eig_val[:1])
    j = 0
    while eig_val[j] < 1e-6:
        j+=1
    print("j: ", j)
    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_] 
    return eig_vec_picked

def getU(newshape,k_near,X_train,P,Band):
    '''MTLPP'''
    l = len(X_train)
    ci = []
    ci1 = 0
    d = X_train[0].shape[2]
    for i in range(l):
        c_matrix = torch.zeros((d, d))
        xt = X_train[i]
        ui = torch.mean(xt,dim=(0,1),keepdim=True)
        ui1 = ui.reshape(200, -1)
        for m in range(9):
            for n in range(9):
                xt1 = xt[m,n,:].reshape(200,-1)
                c_matrix = c_matrix + torch.matmul(xt1-ui1,(xt1-ui1).T)
        c_matrix = c_matrix / 80
        ci.append(c_matrix)

    w,d=dist(ci,k_near,2)#计算临近点
    U1,U2,U3=np.eye(newshape[0],P),np.eye(newshape[1],P),np.eye(newshape[2],Band)
##    U=[0,U1,U2,U3]
##    U=[0.0,U1.astype(np.float64),U2.astype(np.float64),U3.astype(np.float64)]
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
        print(newu1.dtype)
        
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
        left=getyi_yj(y3,w)  
        right=getyy(y3,d)
        newu3=getvalvec(left,right,newshape[2])
        lie=newu3.shape[1]
        for i in range(lie):
            newu3[:,i]=newu3[:,i]/np.linalg.norm(newu3[:,i])
        U3=newu3.real.T
    return U1,U2,U3
def log_euclidean(matrix):
    # 计算特征分解
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 对对角矩阵 Sigma 中的每个元素取对数
    Sigma = np.log(eigenvalues)
    log_Sigma = np.diag(Sigma)
    # 计算 log(C)
    log_C = eigenvectors @ log_Sigma @ eigenvectors.T
    return log_C

def train_test_tensor_fold(PATCH_SIZE,random_idx,image,label):
    num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
    Height, Width, Band = image.shape
    image = image.astype(float)
    '''guiyi'''
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image
    # PATCH_SIZE = 9

    '''Divide the training set and test set and randomly select 10% as the training set'''
    image_pad = np.zeros((Height + (PATCH_SIZE - 1), Width + (PATCH_SIZE - 1), Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    '''fenkuai----mode3-fold----'''
    data_patch_list = []
    label_patch_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                  PATCH_SIZE)  # 没问题
                cut_patch = torch.from_numpy(cut_patch.transpose(1, 2, 0))
                cut_patch = unfold(cut_patch,2)
                data_patch_list.append(cut_patch)
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])
    # random_idx = np.random.choice(len(data_patch_list), int(0.01 * len(data_patch_list)), replace=False)

    X_train = torch.zeros((Band,len(random_idx),PATCH_SIZE*PATCH_SIZE))
    train_label = torch.zeros((num_Class,len(random_idx),PATCH_SIZE*PATCH_SIZE))
    train_label_list, test_label_list = [], []
    i = 0
    for m in random_idx:
        train_label_list.append(label_patch_list[m])
        twist_X = data_patch_list[m].reshape(Band,1,PATCH_SIZE*PATCH_SIZE)
        X_train[:,i,:] = twist_X[:,0,:]
        Yi = torch.zeros((num_Class,PATCH_SIZE*PATCH_SIZE))
        if label_patch_list[m] != 0:
            Yi[label_patch_list[m]-1,:] = 1
        twist_Y = Yi.reshape(num_Class,1,PATCH_SIZE*PATCH_SIZE)
        train_label[:,i,:] = twist_Y[:,0,:]
        i+=1
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    X_test = torch.zeros((Band,len(idx_test),PATCH_SIZE*PATCH_SIZE))
    test_label = torch.zeros((num_Class,len(idx_test),PATCH_SIZE*PATCH_SIZE))
    j = 0
    for m in idx_test:
        test_label_list.append(label_patch_list[m])
        twist_X = data_patch_list[m].reshape(Band,1,PATCH_SIZE*PATCH_SIZE)        
        X_test[:,j,:] = twist_X[:,0,:]
        Yi = torch.zeros((num_Class,PATCH_SIZE*PATCH_SIZE))
        if label_patch_list[m] != 0:
            Yi[label_patch_list[m]-1,:] = 1
        twist_Y = Yi.reshape(num_Class,1,PATCH_SIZE*PATCH_SIZE)
        test_label[:,j,:] = twist_Y[:,0,:] 
        j+=1

    return X_train,train_label,X_test,test_label,train_label_list, test_label_list
def train_test_tensor(image,label):
    num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
    Height, Width, Band = image.shape
    image = image.astype(float)
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image
    PATCH_SIZE = 9

    '''Divide the training set and test set and randomly select 10% as the training set'''
    [Height, Width, Band] = data.shape
    image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    data_patch_list = []
    label_patch_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] - 1 + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] - 1 + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                  PATCH_SIZE)  # 没问题
                data_patch_list.append(torch.from_numpy(cut_patch.transpose(1, 2, 0)))
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])
    random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train, train_label, X_test, test_label = [], [], [], []
    for m in random_idx:
        X_train.append(data_patch_list[m])
        train_label.append(label_patch_list[m])
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    for m in idx_test:
        X_test.append(data_patch_list[m])
        test_label.append(label_patch_list[m])

    # x_test = []
    # for i in range(len(X_test)):
    #     q = X_test[i]
    #     x_test.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_test = torch.tensor([item.detach().numpy() for item in x_test])
    # x_train = []
    # for i in range(len(X_train)):
    #     q = X_train[i]
    #     x_train.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_train = torch.tensor([item.detach().numpy() for item in x_train])
    return X_train,train_label,X_test,test_label
def train_test_tensor_half(PATCH_SIZE,random_idx,image, label):
    Height, Width, Band = image.shape
    image = image.astype(float)
    '''guiyi'''
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image
    # PATCH_SIZE = 9

    '''Divide the training set and test set and randomly select 10% as the training set'''
    [Height, Width, Band] = data.shape
    image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    data_patch_list = []
    label_patch_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                  PATCH_SIZE)  # 没问题
                data_patch_list.append(torch.from_numpy(cut_patch.transpose(1, 2, 0)))
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])
    #random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train, train_label, X_test, test_label = [], [], [], []
    for m in random_idx:
        X_train.append(data_patch_list[m])
        train_label.append(label_patch_list[m])
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    for m in idx_test:
        X_test.append(data_patch_list[m])
        test_label.append(label_patch_list[m])

    # x_test = []
    # for i in range(len(X_test)):
    #     q = X_test[i]
    #     x_test.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_test = torch.tensor([item.detach().numpy() for item in x_test])
    # x_train = []
    # for i in range(len(X_train)):
    #     q = X_train[i]
    #     x_train.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_train = torch.tensor([item.detach().numpy() for item in x_train])
    return X_train,train_label,X_test,test_label
def train_test_2dim_half(random_idx,image, label):
    Height, Width, Band = image.shape
    image = image.astype(float)
    '''guiyi'''
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image


    '''Divide the training set and test set and randomly select 10% as the training set'''
    [Height, Width, Band] = data.shape

    data_patch_list = []
    label_patch_list = []
    for i in range(0, Height):
        for j in range(0, Width):
            if label[i][j] != 0:
                cut_patch = data[i][j]
                data_patch_list.append(torch.from_numpy(cut_patch))
                label_patch_list.append(label[i][j])
    #random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train, train_label, X_test, test_label = [], [], [], []
    for m in random_idx:
        X_train.append(data_patch_list[m])
        train_label.append(label_patch_list[m])
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    for m in idx_test:
        X_test.append(data_patch_list[m])
        test_label.append(label_patch_list[m])

    # x_test = []
    # for i in range(len(X_test)):
    #     q = X_test[i]
    #     x_test.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_test = torch.tensor([item.detach().numpy() for item in x_test])
    # x_train = []
    # for i in range(len(X_train)):
    #     q = X_train[i]
    #     x_train.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_train = torch.tensor([item.detach().numpy() for item in x_train])
    return X_train,train_label,X_test,test_label
def train_test_zishiying(image, label):
    num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
    Height, Width, Band = image.shape
    image = image.astype(float)
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image  # 归一化
    PATCH_SIZE = 11
    [Height, Width, Band] = data.shape
    # 填充 153*153
    image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    #装每个9*9的列表
    l = 49 #7*7
    data_patch_list = []
    label_patch_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] - 1 + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] - 1 + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),PATCH_SIZE)  # 没问题
                cut_patch = cut_patch.transpose(1, 2, 0)
                center = image_pad[i,j,:]
                flat = np.reshape(cut_patch,(PATCH_SIZE*PATCH_SIZE,Band))
                distance = []
                for vector in flat:
                    dis = np.linalg.norm(center-vector)
                    distance.append(dis)
                distance = np.array(distance)
                ids = np.argsort(distance)[:81]
                ids = np.reshape(ids,(9,9))
                data_patch = flat[ids,:]
                data_patch_list.append(torch.from_numpy(data_patch))
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])

    random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train, train_label, X_test, test_label = [], [], [], []
    for m in random_idx:
        X_train.append(data_patch_list[m])
        train_label.append(label_patch_list[m])
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    for m in idx_test:
        X_test.append(data_patch_list[m])
        test_label.append(label_patch_list[m])
    return X_train,train_label,X_test,test_label
def train_test_fold_zishiying(random_idx,image,label):
    num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
    Height, Width, Band = image.shape
    image = image.astype(float)
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image  # 归一化
    PATCH_SIZE = 11
    [Height, Width, Band] = data.shape
    # 填充 153*153
    image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    # 装每个9*9的列表
    l = 9  # 7*7
    data_patch_list = []
    label_patch_list = []
    data_patch_W_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] - 1 + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] - 1 + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                  PATCH_SIZE)  # 没问题
                cut_patch = cut_patch.transpose(1, 2, 0)
                center = image_pad[i, j, :]
                flat = np.reshape(cut_patch, (PATCH_SIZE * PATCH_SIZE, Band))
                distance = []
                for vector in flat:
                    dis = np.linalg.norm(center - vector)
                    distance.append(dis)
                distance = np.array(distance)
                ids = np.argsort(distance)[:81]
                ids1 = np.reshape(ids, (9, 9))
                data_patch = flat[ids, :]
                data_patch_W = flat[ids1, :]
                data_patch_list.append(torch.from_numpy(data_patch.transpose(1,0)))
                data_patch_W_list.append(torch.from_numpy(data_patch_W))
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])
    x_train_W = []
    # random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train = torch.zeros((Band, len(random_idx), l * l))
    train_label = torch.zeros((num_Class, len(random_idx), l * l))
    train_label_list, test_label_list = [], []
    i = 0
    for m in random_idx:
        train_label_list.append(label_patch_list[m])
        x_train_W.append(data_patch_W_list[m])
        twist_X = data_patch_list[m].reshape(Band, 1, l * l)
        X_train[:, i, :] = twist_X[:, 0, :]
        Yi = torch.zeros((num_Class, l * l))
        if label_patch_list[m] != 0:
            Yi[label_patch_list[m] - 1, :] = 1
        twist_Y = Yi.reshape(num_Class, 1, l * l)
        train_label[:, i, :] = twist_Y[:, 0, :]
        i += 1
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    X_test = torch.zeros((Band, len(idx_test), l * l))
    test_label = torch.zeros((num_Class, len(idx_test), l * l))
    j = 0
    for m in idx_test:
        test_label_list.append(label_patch_list[m])
        twist_X = data_patch_list[m].reshape(Band, 1, l * l)
        X_test[:, j, :] = twist_X[:, 0, :]
        Yi = torch.zeros((num_Class, l * l))
        if label_patch_list[m] != 0:
            Yi[label_patch_list[m] - 1, :] = 1
        twist_Y = Yi.reshape(num_Class, 1, l * l)
        test_label[:, j, :] = twist_Y[:, 0, :]
        j += 1

    return X_train, train_label, X_test, test_label, train_label_list, test_label_list,x_train_W

def nn_3dim(x_train, train_label,x_test, test_label):
    computedClass = []
    D = np.zeros((len(x_test),len(x_train)))
    for i in range(len(x_test)):
        current_block = x_test[i]
        for j in range(len(x_train)):
            neighbor_block = x_train[j]
            w = current_block-neighbor_block
            d = np.linalg.norm(w)
            D[i,j] = d
    id = np.argsort(D, axis=1)
    count = 0

    computedClass.append(np.array(train_label)[id[:, 0]])
    for w in range(len(x_test)):
        if computedClass[0][w]==test_label[w]:
            count = count+1
    recogRate = count / len(x_test)

    return recogRate
def nn_2dim(x_train, train_label,x_test, test_label):
    x_train = np.array(x_train)
    train_label = np.array(train_label)
    x_test= np.array(x_test)
    test_label= np.array(test_label)
    computedClass = []
    D = np.zeros((x_test.shape[0],x_train.shape[0]))
    for i in range(x_test.shape[0]):
        current_block = x_test[i][:]
        for j in range(x_train.shape[0]):
            neighbor_block = x_train[j][:]
            w = current_block-neighbor_block
            d = np.linalg.norm(w)
            D[i,j] = d
    id = np.argsort(D, axis=1)
    count = 0

    computedClass.append(train_label[id[:, 0]][:])
    for w in range(x_test.shape[0]):
        if computedClass[0][w]==test_label[w]:
            count = count+1
    recogRate = count / x_test.shape[0]

    return recogRate


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


def train_test_2dim(data, gnd):
    data = np.array(data)
    gnd = np.array(gnd)
    sample, band = data.shape
    data = data.astype(float)
    for sample in range(sample):
        data[sample, :] = (data[sample, :] - np.min(data[sample, :])) / (
                np.max(data[sample, :]) - np.min(data[sample, :]))

    label = np.unique(gnd)
    x_train = []
    train_label = []
    x_test = []
    test_label = []
    for i in range(len(label)):
        ind = np.where(gnd == label[i])[0]
        nl = len(ind)
        index = np.random.permutation(nl)
        ktrain = round(nl * 0.1)

        Trin = ind[index[:ktrain]]
        Tein = ind[index[ktrain:]]

        tmpTr = data[Trin, :]
        x_train.extend(tmpTr)
        tmpGTr = gnd[Trin]
        train_label.extend(tmpGTr)
        # ========测试样本===========
        tmpTe = data[Tein, :]
        x_test.extend(tmpTe)
        tmpGTe = gnd[Tein]
        test_label.extend(tmpGTe)

    return x_train, train_label, x_test, test_label
