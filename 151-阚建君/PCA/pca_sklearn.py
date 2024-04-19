import numpy as np
from sklearn.decomposition import PCA

X = np.array([[6, 25, 18],
              [16, 33, 55],
              [26, 43, 24],
              [15, 9, 46],
              [52, 41, 21],
              [9, 48, 5],
              [8, 9, 13],
              [81, 56, 55],
              [13, 31, 45],
              [40, 41, 23]])
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(newX)