import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29],  # 10行3列
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
pca = PCA(n_components=2)  # 降至2维
pca.fit(X)  # 训练计算出主成分分析的结果，即计算出特征向量和特征值，存储在PCA模型对象中
newX = pca.fit_transform(X)  # 实际对输入数据做降维操作
print(newX)
