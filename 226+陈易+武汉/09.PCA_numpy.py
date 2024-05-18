# coding=utf-8
import numpy as np

class PCA():
    def __init__(self,n_componeents):
        self.n_componeents = n_componeents

    def fit_tansform(self,X):
        self.n_features = X.shape[1]   # 样本矩阵X的列数（宽度）
        # 求协方差矩阵
        X = X - X.mean(axis=0)    # 中心化均值
        self.covarance = np.dot(X.T, X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.covarance)
        # 获取降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵  参数维度个数的列向量组成的特征向量
        self.componeents = eig_vectors[:,idx[:self.n_componeents]]    # n_componeents 为维度数
        # 对样本矩阵X进行降维
        return  np.dot(X,self.componeents)

# 面向对象  n_componeents 参数为维度数
pca = PCA(n_componeents=2)
# 样本集D
D = [[-1,2,66,-1],
      [-2,6,58,-1],
      [-3,8,45,-2],
      [1,9,36,1],
      [2,10,62,1],
      [3,5,83,2]]
# 样本矩阵X，维度为4
X = np.array(D)
print("导入数据X：\n",X)
# 调用降维函数  对象.函数 ：调用函数
new_X = pca.fit_tansform(X)
print("降维后的样本矩阵new_X:\n",new_X)