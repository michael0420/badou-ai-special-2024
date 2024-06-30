import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris

if __name__ == "__main__":
    # 生成测试数据
    src_data, _ = load_iris(return_X_y=True)

    # 截取数据前2维
    src_data = src_data[:, :2]

    # 加载密度聚类算法
    dbscan = DBSCAN(eps=0.35, min_samples=1)

    # 进行密度聚类
    dbscan.fit(src_data)

    # 获取标签
    labels = dbscan.labels_

    # 画散点图展示聚类结果
    plt.scatter(src_data[:, 0], src_data[:, 1], c=labels)
    plt.show()

