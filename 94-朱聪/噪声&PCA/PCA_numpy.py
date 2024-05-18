import numpy as np


"""
一般步骤是这样的：

1、对原始数据零均值化（中心化）

2、求协方差矩阵

3、对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
"""

# 10样本3特征的样本集, 行为样例，列为特征维度
X = np.array([
    [10, 15, 29],
    [15, 46, 13],
    [23, 21, 30],
    [11, 9,  35],
    [42, 45, 11],
    [9,  48, 5],
    [11, 21, 14],
    [8,  5,  15],
    [11, 12, 21],
    [21, 20, 25]
])

K = np.shape(X)[1] - 1 # shape得到数据的行列元组（行数，列数） 取[1]就是列，也就是维度

# Ax = λx  A是初始矩阵，x和λ是要求的

# 1、去中心化
centrX = []
mean = np.array([np.mean(attr) for attr in X.T]) #样本集的特征均值
print('样本集的特征均值:\n',mean)
centrX = X - mean ##样本集的中心化
print('样本矩阵X的中心化centrX:\n', centrX)


# 2、求协方差矩阵
ns = np.shape(X)[0] # 样本数量
C = X.T.dot(X) / (ns - 1)

# 3、求特征值等
a,b = np.linalg.eig(C) # 特征值赋值给a，对应特征向量赋值给b

#给出特征值降序的topK的索引序列
ind = np.argsort(-1*a)

UT = [b[:,ind[i]] for i in range(K)]

# U = (UT).T # # 这里没用.T是因为UT是list，可以先np.array用T
U = np.transpose(UT)

Z = np.dot(X, U)



# 直接调API
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                 #输出降维后的数据