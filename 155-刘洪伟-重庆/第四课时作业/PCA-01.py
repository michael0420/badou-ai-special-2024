# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 17:00
# @Author: iris
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.uniform(1, 10, (50, 5))
    pca = PCA(n_components=3)
    pca.fit(X)
    dst = pca.transform(X)
    print(dst)
    print(pca.explained_variance_ratio_)
