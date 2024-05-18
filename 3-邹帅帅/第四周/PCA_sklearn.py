
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-10,21,63,-1], [-5,6,78,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)
newX = pca.fit_transform(X)
print(newX)