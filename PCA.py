# #!/usr/bin/env python
# # encoding=gbk

# PCA��ά�㷨ʵ��
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# �����β�����ݼ�
x, y = load_iris(return_X_y=True)

# ʹ��sklearn���е�PCAʵ��
pca_sklearn = PCA(n_components=2)
reduced_x_sklearn = pca_sklearn.fit_transform(x)
print("sklearn PCA ����:\n", reduced_x_sklearn)

# ����CPCA��
class CPCA:
    '''��PCA����������X��K�׽�ά����Z
    Note:�뱣֤�������������X shape=(m, n)��m��������n������
    '''
    def __init__(self, X, K):
        '''
        :param X,��������X
        :param K,X�Ľ�ά����Ľ�������XҪ������ά��k��
        '''
        self.X = X       #��������X
        self.K = K       #K�׽�ά�����Kֵ
        self.centrX = [] #����X�����Ļ�
        self.C = []      #��������Э�������C
        self.U = []      #��������X�Ľ�άת������
        self.Z = []      #��������X�Ľ�ά����Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU���

    def _centralized(self):
        '''����X�����Ļ�'''
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  #��������������ֵ
        centrX = self.X - mean  ##�����������Ļ�
        return centrX

    def _cov(self):
        '''����������X��Э�������C'''
        #����������������
        ns = np.shape(self.centrX)[0]
        #���������Э�������C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        return C

    def _U(self):
        '''��X�Ľ�άת������U, shape=(n,k), n��X������ά��������k�ǽ�ά���������ά��'''
        #����X��Э�������C������ֵ����������
        a,b = np.linalg.eig(self.C) #����ֵ��ֵ��a����Ӧ����������ֵ��b
        #��������ֵ�����topK����������
        ind = np.argsort(-1*a)
        #����K�׽�ά�Ľ�άת������U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        '''����Z=XU��ά����Z, shape=(m,k), n������������k�ǽ�ά����������ά������'''
        Z = np.dot(self.X, self.U)
        return Z

# ʹ���Զ����CPCA��
cpca = CPCA(x, 2)
reduced_x_cpca = cpca._Z()
print("CPCA ����:\n", reduced_x_cpca)

# ����PCA��
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

# ʹ���Զ����PCA��
pca = PCA(n_components=2)
reduced_x_pca = pca.fit_transform(x)
print("PCA ����������:\n", reduced_x_pca)


reduced_x=pca.fit_transform(x) #��ԭʼ���ݽ��н�ά��������reduced_x��

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
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