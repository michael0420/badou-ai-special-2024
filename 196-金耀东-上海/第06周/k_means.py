import numpy as np
import matplotlib.pyplot as plt

class K_Means():

    def __init__(self):
        self.num_clusters = None

    def fit(self, src_data, num_clusters, max_iter=10, tol=1e-10):
        # 设置聚类数量
        self.num_clusters = num_clusters

        # 初始化聚类中心
        centroids = np.array(src_data[:self.num_clusters], dtype=float)

        for i in range(max_iter):
            # 将样本分配到最近中心点的类中
            labels = np.zeros(src_data.shape[0]).astype("uint8")
            for j in range(src_data.shape[0]):
                distance = np.linalg.norm(centroids-src_data[j], ord=2, axis=1)
                labels[j] = distance.argmin()

            # 保存原来中心点
            pre_centroids = centroids.copy()

            # 更新中心点
            for j in range(self.num_clusters):
                cluster =  src_data[labels==j]
                if len(cluster) == 0:
                    continue
                centroids[j] = np.mean(cluster, axis=0)

            # 如果中心点无变化，则退出循环
            if np.all(np.linalg.norm(centroids-pre_centroids, ord=2, axis=1) < tol ):
                break

        return centroids, labels

if __name__ == "__main__":
    # 初始化测试数据
    src_data = np.array([[0.0888, 0.5885],
                 [0.1399, 0.8291],
                 [0.0747, 0.4974],
                 [0.0983, 0.5772],
                 [0.1276, 0.5703],
                 [0.1671, 0.5835],
                 [0.1306, 0.5276],
                 [0.1061, 0.5523],
                 [0.2446, 0.4007],
                 [0.1670, 0.4770],
                 [0.2485, 0.4313],
                 [0.1227, 0.4909],
                 [0.1240, 0.5668],
                 [0.1461, 0.5113],
                 [0.2315, 0.3788],
                 [0.0494, 0.5590],
                 [0.1107, 0.4799],
                 [0.1121, 0.5735],
                 [0.1007, 0.6318],
                 [0.2567, 0.4326],
                 [0.1956, 0.4280]
                 ])

    # 设置聚类数
    num_clusters = 3

    # 加载k_means算法
    my_k_means = K_Means()

    # 运行kmeans聚类算法
    centroids, labels = my_k_means.fit(src_data, num_clusters)

    # 画散点图展示聚类结果
    plt.scatter(src_data[:,0], src_data[:,1], c=labels)
    plt.show()