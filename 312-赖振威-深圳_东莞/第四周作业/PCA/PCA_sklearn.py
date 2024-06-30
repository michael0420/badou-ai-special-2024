#coding=utf-8

import numpy as np  # 导入NumPy库
from sklearn.decomposition import PCA  # 导入PCA模块

# 创建导入一个包含样本数据的NumPy数组，维度为4
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])

pca = PCA(n_components=2)  # 创建一个PCA对象，将数据降到2维
pca.fit(X)  # 对数据进行PCA降维模型的训练
newX = pca.fit_transform(X)  # 对原始数据进行降维操作，得到降维后的数据

# 输出PCA模型的贡献率，即每个主成分所占的方差比例
print('PCA模型的贡献率:\n',pca.explained_variance_ratio_)
'''
pca.explained_variance_ratio_ 是一个数组，其中每个元素表示对应主成分所解释的方差比例。
在主成分分析中，主成分的选择是按照它们解释数据中的方差的能力来排序的。
因此，pca.explained_variance_ratio_ 中的每个值都告诉你每个主成分解释的方差的比例。

具体地说，对于一个有 n 个特征的数据集，PCA 会生成 n 个主成分，其中第一个主成分解释的方差比例最大，第二个主成分解释的方差比例次之，依此类推。
这个属性的输出告诉你每个主成分解释了多大比例的总方差。

这里是默认设置好的嘛？
是的，PCA 类的 explained_variance_ratio_ 属性默认返回主成分解释的方差比例。
在 PCA 类的初始化中，没有指定 svd_solver 参数时，默认使用的是 'auto' 模式，该模式会根据输入数据的大小和特征数量自动选择合适的求解器。
在大多数情况下，explained_variance_ratio_ 返回的是按照方差解释比例递减的顺序排列的主成分。
'''

# 输出降维后的数据
print('输出降维后的数据:\n',newX)