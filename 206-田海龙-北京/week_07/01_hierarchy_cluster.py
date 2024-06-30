from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


def hierarchy_cluster(data, method="ward"):

    # 分层聚类
    Z = linkage(data, method)

    # 聚类分割，显示的是分组标签
    labels = fcluster(Z, 3, "distance")
    print(labels)

    fig = plt.figure(figsize=(5, 3))
    # 显示层次聚类图
    # dn = dendrogram(Z)
    dendrogram(Z)
    print(Z)
    plt.show()


data = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
hierarchy_cluster(data)
