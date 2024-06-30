"""
手动实现PCA算法
"""

import  numpy as np

n_components=3
def PCA(X):
    X=X-X.mean(0) #列维度，减去均值
    c=np.dot(X.T,X)/X.shape[0] # 协方差矩阵
    # 求协方差矩阵的特征值和特征向量
    eig_vals, eig_vectors =    np.linalg.eig(c)
    # 获得降序排列特征值的序号
    idx = np.argsort(-eig_vals)
    # 降维矩阵
    c_ = eig_vectors[:, idx[:n_components]]
    # 对X进行降维
    return np.dot(X, c_)

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX=PCA(X)
print(newX)