from sklearn.cluster import AgglomerativeClustering
import numpy as np

# 生成示例数据
np.random.seed(0)
data = np.random.rand(10, 2)




# 创建聚类器实例
agg_cluster = AgglomerativeClustering(n_clusters=3)

# 训练聚类模型
agg_cluster.fit(data)

# 预测数据的聚类标签
labels = agg_cluster.labels_

print(data)



# 打印聚类中心
print("聚类中心:", agg_cluster.n_clusters_)

# 打印聚类标签
print("数据点的聚类标签:", labels)



"""
可视化绘图
"""

import matplotlib.pyplot as plt

# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in data]
print(x)
y = [n[1] for n in data]
print(y)

''' 
绘制散点图  
'''
plt.scatter(x, y,  marker='x')

# 绘制标题
plt.title(" Implement hierarchical clustering ")

# 绘制x轴和y轴坐标
plt.xlabel("X-axis")
plt.ylabel("Y-axis")



# 显示图形
plt.show()










