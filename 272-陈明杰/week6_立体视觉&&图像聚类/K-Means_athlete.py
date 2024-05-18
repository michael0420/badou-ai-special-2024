from sklearn.cluster import KMeans

# 1、数据集
# X表示二维矩阵数据，篮球运动员比赛数据
# 总共n行，每行两列数据
# 第一列表示球员每分钟助攻数：assists_per_minute
# 第二列表示球员每分钟得分数：points_per_minute

X = [[0.0888, 0.5885],
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
     ]

# 2、初始化KMeans聚类
clf = KMeans(n_clusters=3)
# 进行聚类并将聚类结果放到clu中
clu = clf.fit_predict(X)
# print(clf)
# print(clu)

# 3、进行可视化显示
import matplotlib.pyplot as plt

x = [n[0] for n in X]
y = [n[1] for n in X]
# print(x)
# print(y)

# 绘制散点图
plt.scatter(x, y, c=clu, marker='o')

# 绘制标题
plt.title("Kmeans-Basketball Data")
# 绘制x、y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

import numpy as np
import matplotlib.patches as patches

unique_labels = np.unique(clu)
# print(unique_label)
colors = plt.cm.get_cmap('viridis', len(unique_labels))
patches_list = []

for i, label in enumerate(unique_labels):
    # 循环遍历每个唯一的聚类标签，并为每个标签创建一个代理点（Patch 对象）。
    # 代理点的颜色基于之前创建的颜色映射的索引，并且标签被设置为 'Cluster X'
    # 的形式，其中 X 是聚类标签。这些代理点稍后将用于在图例中显示。
    patch = patches.Patch(color=colors(i), label=f'Cluster {label}')
    patches_list.append(patch)

plt.legend(handles=patches_list, loc='best')
plt.show()


X = [[0.0888, 0.5885],
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
     ]

clf = KMeans(n_clusters=3)
labels = clf.fit_predict(X)
print(labels)

# 画图
data_x = [e[0] for e in X]
data_y = [e[1] for e in X]
plt.scatter(data_x, data_y, c=labels, marker='o')
plt.xlabel('score')
plt.ylabel('assist')
plt.title('athlete data')
# plt.legend(handles=scatter.legend_elements()[0], labels=["A", "B", "C"])
# scatter1 = plt.scatter(X[labels == 0, 0], X[labels == 0, 1], label='A', color='blue')
# scatter2 = plt.scatter(X[labels == 1, 0], X[labels == 1, 1], label='B', color='green')
# scatter3 = plt.scatter(X[labels == 2, 0], X[labels == 2, 1], label='C', color='red')
scatter = plt.scatter(x, y, c=labels, marker='x')
#设置右上角图例
plt.legend(handles=scatter.legend_elements()[0], labels=["A","B","C"])  # matplotlib3.1.2可用
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 假设 x, y 是您的数据，labels 是类别标签
# np.random.seed(0)
# x = np.random.rand(100)
# y = np.random.rand(100)
# labels = np.random.randint(0, 3, 100)  # 假设有三个类别：0, 1, 2
#
# # 为每个类别创建一个散点图，并指定 label
# for i in range(3):
#      plt.scatter(x[labels == i], y[labels == i], label=chr(65 + i), marker='x')  # 使用 A, B, C 作为标签
#
# # 创建图例
# plt.legend()
#
# # 显示图形
# plt.show()