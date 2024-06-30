# incoding = UTF-8
import numpy as np
from sklearn.decomposition import PCA   # 导入scikit-learn模块
# 样本集D
D = [[-1,2,66,-1],
     [-2,6,58,-1],
     [-3,8,45,-2],
     [1,9,36,1],
     [2,10,62,1],
     [3,5,83,2]]
print("原样本集D:\n",D)
# 样本矩阵X
X = np.array(D)
print("样本矩阵X:\n",X)
pca = PCA(n_components=2)     # 降为2维，n_components维度参数
pca.fit(X)                    # 执行：对pca这个对象进行训练
new_X = pca.fit_transform(X)  # 用X来训练PCA模型，同时返回降维后的数据
print("降维后的样本矩阵new_X:\n",new_X)