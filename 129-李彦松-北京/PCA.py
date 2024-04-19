# #!/usr/bin/env python
# # encoding=gbk

# PCA降维算法实现
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
x, y = load_iris(return_X_y=True)

# 使用sklearn库中的PCA实现
pca_sklearn = PCA(n_components=2)
reduced_x_sklearn = pca_sklearn.fit_transform(x)
print("sklearn PCA 处理:\n", reduced_x_sklearn)

# 定义CPCA类
class CPCA:
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X       #样本矩阵X
        self.K = K       #K阶降维矩阵的K值
        self.centrX = [] #矩阵X的中心化
        self.C = []      #样本集的协方差矩阵C
        self.U = []      #样本矩阵X的降维转换矩阵
        self.Z = []      #样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU求得

    def _centralized(self):
        '''矩阵X的中心化'''
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  #样本集的特征均值
        centrX = self.X - mean  ##样本集的中心化
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        #样本集的样例总数
        ns = np.shape(self.centrX)[0]
        #样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b
        #给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)
        #构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        return Z

# 使用自定义的CPCA类
cpca = CPCA(x, 2)
reduced_x_cpca = cpca._Z()
print("CPCA 处理:\n", reduced_x_cpca)

# 定义PCA类
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        X = X - X.mean(axis=0)
        covariance = np.dot(X.T, X) / X.shape[0]
        eig_vals, eig_vectors = np.linalg.eig(covariance)
        idx = np.argsort(-eig_vals)
        components_ = eig_vectors[:, idx[:self.n_components]]
        return np.dot(X, components_)

# 使用自定义的PCA类
pca = PCA(n_components=2)
reduced_x_pca = pca.fit_transform(x)
print("PCA 处理后的数据:\n", reduced_x_pca)


reduced_x=pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()