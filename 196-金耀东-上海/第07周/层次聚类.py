import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.datasets import load_iris

if __name__ == "__main__":
    # 生成测试数据
    src_data, _ = load_iris(return_X_y=True)

    # 截取数据前2维
    src_data = src_data[:, :2]

    # 进行层次聚类
    Z = linkage(src_data, method="ward")

    # 根据聚类结果画分层图
    dendrogram(Z)
    plt.show()

    # 距离阈值设置6进行聚类
    labels = fcluster(Z, 7, "distance")

    # 画散点图展示聚类结果
    plt.scatter(src_data[:, 0], src_data[:, 1], c=labels)
    plt.show()
