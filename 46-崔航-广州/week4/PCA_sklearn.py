#coding=utf-8

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,6,-1], [-2,6,8,-1], [-3,8,4,-2], [1,9,6,1], [2,10,2,1], [3,5,3,2]])  #导入数据，维度为4
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据
