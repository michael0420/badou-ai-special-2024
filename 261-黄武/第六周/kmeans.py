import numpy as np
import matplotlib.pyplot as plt

# 随机生成20个二维数据点
data = np.random.rand(20, 2)  # 生成20个随机二维点

# 选择初始质心，随机选择两个数据点作为初始质心
initial_centroids = data[np.random.choice(len(data), 2, replace=False)]



def euclidean_distance(data_points, centroid):
    return np.sqrt(np.sum((data_points - centroid) ** 2, axis=1))

def kmeans(data, initial_centroids, max_iters=100):
    num_clusters = len(initial_centroids)
    centroids = initial_centroids.copy()

    for _ in range(max_iters):
        # 将每个数据点分配到最近的质心
        distances = np.array([euclidean_distance(data, centroid) for centroid in centroids])
        closest_centroids = np.argmin(distances, axis=0)

        # 更新质心为分配给它的点的平均值
        new_centroids = np.array([data[closest_centroids == k].mean(axis=0) for k in range(num_clusters)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, closest_centroids


# 调用 kmeans 函数
centroids, closest_centroids = kmeans(data, initial_centroids, max_iters=10)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=closest_centroids, cmap='viridis', label='Data points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Num 1')
plt.ylabel('Num 2')

plt.legend()
plt.show()
