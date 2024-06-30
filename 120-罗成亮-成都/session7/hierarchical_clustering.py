import random

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = []
for i in range(0, 5):
    X.append([random.randint(1, 10), random.randint(1, 10)])
print(X)
x, y = zip(*X)

# 创建一个新的图表
plt.figure()

# 绘制散点图
plt.scatter(x, y, color='red')  # 点的颜色设为红色

# 添加标题和坐标轴标签
plt.title('Points Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示网格
plt.grid(True)

Z = linkage(X, 'ward')
f = fcluster(Z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
