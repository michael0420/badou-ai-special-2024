import  numpy as np
from sklearn.decomposition import PCA

#测试用数据
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
#降维
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(newX)