import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 
from  sklearn.cluster import DBSCAN
from matplotlib import font_manager as fm
 
iris = datasets.load_iris() ##加载数据
X = iris.data[:, :4]  ##表示我们只取特征空间中的4个维度
print(X.shape)
# 绘制数据分布图

plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()


dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_ ## label_pred就是我们的分类结果
 
# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)
plt.show()  


# 尝试加载中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体字体路径，确保路径正确
font_prop = fm.FontProperties(fname=font_path)

# 设置matplotlib全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(dataset, point_index, eps):
    neighbors = []
    for index in range(len(dataset)):
        if index != point_index and euclidean_distance(dataset[point_index], dataset[index]) < eps:
            neighbors.append(index)
    return neighbors

def dbscan(dataset, eps, min_samples):
    labels = [0] * len(dataset)  # 0 表示未分类
    cluster_id = 0  # 类别编号

    for point_index in range(len(dataset)):
        if labels[point_index] != 0:  # 如果点已经被分类，则跳过
            continue

        neighbors = get_neighbors(dataset, point_index, eps)  # 获取邻域

        if len(neighbors) < min_samples:  # 如果邻域内点数少于最小样本数，则标记为噪声
            labels[point_index] = -1
        else:
            cluster_id += 1  # 启动一个新类
            labels[point_index] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_index = neighbors[i]
                if labels[neighbor_index] == -1:  # 如果是噪声点
                    labels[neighbor_index] = cluster_id
                elif labels[neighbor_index] == 0:  # 如果未分类
                    labels[neighbor_index] = cluster_id
                    point_neighbors = get_neighbors(dataset, neighbor_index, eps)
                    if len(point_neighbors) >= min_samples:  # 如果满足密度要求
                        neighbors += point_neighbors  # 将邻域加入到当前邻域中
                i += 1

    return labels

iris = datasets.load_iris() ##加载数据
X = iris.data[:, :4]  ##表示我们只取特征空间中的4个维度

# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

# 运行自定义的DBSCAN算法
eps = 0.4
min_samples = 9
labels = dbscan(X, eps, min_samples)

# 绘制聚类结果
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # 噪声点用黑色表示
    class_member_mask = (np.array(labels) == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors='k', s=50, label='类别 ' + str(k))

plt.title('估计的簇数量: %d' % len(unique_labels), fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()