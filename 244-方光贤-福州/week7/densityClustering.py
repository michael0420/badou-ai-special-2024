import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA  # 导入PCA用于降维
from sklearn.metrics import silhouette_score  # 导入轮廓系数用于评估

iris = datasets.load_iris()
X = iris.data

# 使用PCA将特征降到二维以便于可视化 因为老师原始数据有4个维度特征但是只用了两个所以考虑到用pca降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)  # 尝试不同的参数
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制聚类结果
plt.scatter(X_pca[label_pred == 0, 0], X_pca[label_pred == 0, 1], c="red", marker='o', label='Cluster 0')
plt.scatter(X_pca[label_pred == 1, 0], X_pca[label_pred == 1, 1], c="green", marker='*', label='Cluster 1')
plt.scatter(X_pca[label_pred == 2, 0], X_pca[label_pred == 2, 1], c="blue", marker='+', label='Cluster 2')
# 如果有噪声点 使用不同的标记或颜色来表示
plt.scatter(X_pca[label_pred == -1, 0], X_pca[label_pred == -1, 1], c="black", marker='x', label='Noise')

plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(loc=2)
plt.title('DBSCAN Clustering of Iris Dataset (PCA-reduced)')
plt.show()

# 评估聚类结果
silhouette_avg = silhouette_score(X, label_pred)
print("Silhouette Coefficient: ", silhouette_avg)