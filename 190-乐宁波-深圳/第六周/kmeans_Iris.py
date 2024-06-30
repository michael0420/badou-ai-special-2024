# 导入必要的库
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y_true = iris.target  # 真实类别

'''
鸢尾花（Iris）数据集中，通常包含以下四个特征：
花萼长度（Sepal Length）
花萼宽度（Sepal Width）
花瓣长度（Petal Length）
花瓣宽度（Petal Width）
'''
print(X)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 输出聚类结果
print("聚类结果:", y_kmeans)

# 可视化聚类结果
plt.figure(figsize=(12, 6))

# 为每个聚类结果绘制散点图
for i in range(kmeans.n_clusters):
    plt.scatter(X[y_kmeans == i, 2], X[y_kmeans == i, 3], label=f'Cluster {i + 1}',
                marker='o', alpha=0.6)  # 为了方便可视化，只选了第3/4个特征进行可视化

# 标记聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 2], centers[:, 3], c='red', s=200, alpha=0.75, marker='*', label='Centroids')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# 显示图表
plt.show()
