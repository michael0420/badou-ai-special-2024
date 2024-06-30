
import numpy as np
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit_transform(self, src):
           # 矩阵样本中心
           src = src - src.mean(axis=0) 
           
           # 求协方差矩阵
           self.covariance = np.dot(src.T, src) / src.shape[0]
           
           # 求协方差矩阵的特征值和特征向量
           eig_vals, eig_vectors = np.linalg.eig(self.covariance)
           
           # 获得降序排列特征值的序号
           idx = np.argsort(-eig_vals)[:self.n_components]
           
           # 降维矩阵
           self.components_ = eig_vectors[:,idx]
           
           # 对X进行降维
           return np.dot(src, self.components_)
           
# 调用

pca = PCA(n_components=2)
#导入数据，维度为4
src = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  

dst = pca.fit_transform(src)

print(dst)