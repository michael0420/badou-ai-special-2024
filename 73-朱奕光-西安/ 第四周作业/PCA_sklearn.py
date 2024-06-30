import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
pca = PCA(n_components=2)
pca.fit(X)
X_r = pca.transform(X)
print(X_r)
