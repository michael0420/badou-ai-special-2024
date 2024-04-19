import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2) #降到2维
# pca.fit(X)
newX = pca.fit_transform(X) #执行降维  fit_transform 先fit后transform
print(newX)