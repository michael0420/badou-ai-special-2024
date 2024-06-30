import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
    X = np.array([
        [47, 8, 7],
        [11, 16, 32],
        [12, 5, 1],
        [5, 21, 15],
        [47, 47, 2],
        [8, 22, 44],
        [6, 14, 20],
        [7, 9, 3],
        [48, 2, 1],
        [37, 14, 23],
    ])
    pca = PCA(n_components=2)
    pca.fit(X)
    newX = pca.fit_transform(X)
    print(newX)
