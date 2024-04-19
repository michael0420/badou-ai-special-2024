# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 17:05
# @Author: iris
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    target = iris.target
    print(data, target)
    pca = PCA(n_components=2)
    pca.fit(data)
    dst = pca.transform(data)
    print(dst)

    # 图形化显示
    plt.scatter(dst[target == 0, 0], dst[target == 0, 1], c='red', label=iris.target_names[0])
    plt.scatter(dst[target == 1, 0], dst[target == 1, 1], c='green', label=iris.target_names[1])
    plt.scatter(dst[target == 2, 0], dst[target == 2, 1], c='blue', label=iris.target_names[2])
    plt.title("iris dataset")
    plt.show()
