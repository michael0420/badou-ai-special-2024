import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

data = np.array(
    [[1.5, 'Red'], [3.1, 'Blue'], [5.2, 'Green'], [1.9, 'Yellow'], [3.5, 'Purple'], [7.2, 'Green'], [4.5, 'Red'],
     [3.8, 'Blue'], [1.2, 'Yellow'], [3.4, 'Yellow'], [3.1, 'Blue'], [6.2, 'Purple']])

# 将类别数据转换为数值型数据
categories = np.unique(data[:, 1])
category_mapping = {category: i for i, category in enumerate(categories)}
data[:, 1] = [category_mapping[category] for category in data[:, 1]]

# linkage进行层次聚类操作，返回数组Z，
# 含聚类迭代次数，如y=[[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]], Z=[1 3 4 1 2]，意味index0与index3在第1次迭代中完成聚合，之后第二次聚合与index5发生
# def linkage(y: ndarray, 若样本y维度为n，返回数组Z的维度为n-1
#             method: Optional[str] = 'single', 度量簇的相似性的算法
#             metric: Union[str, function, None] = 'euclidean', 距离算法，默认欧式距离
#             optimal_ordering: Optional[bool] = False)
Z = linkage(data, method='ward')

# fcluster的使用必须前置linkage，因为linkage的输出是fcluster的输入
max_clust_num = 2  # 设置输出簇的个数
clusters = fcluster(Z, max_clust_num, criterion='maxclust')
max_distance = 3  # 组成簇的最小距离
f = fcluster(Z, max_distance, criterion='distance')

# 打印簇分配结果
print("Cluster assignments:", clusters)


# 绘制聚类结果
def plot_clusters(data, clusters):
    plt.figure(figsize=(8, 5))
    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        cluster_data = data[clusters == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Hierarchical Clustering Results')
    plt.show()


plot_clusters(data, clusters)


# 绘制层次聚类树
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


plot_dendrogram(Z)
