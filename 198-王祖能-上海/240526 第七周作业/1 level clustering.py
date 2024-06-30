'''
hierarchical clustering层次聚类；dendrogram聚类画系统树图；linkage链条聚类计算；fcluster聚类过程表格结果
'''
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # 聚类/列表/绘图
from matplotlib import pyplot as plt
import numpy as np

X = np.array([[1, 2],
              [3, 2],
              [4, 4],
              [7, 12],
              [9, 7],
              [1, 2],
              [1, 3]])
Y = np.array([[1, 2, 4],
              [3, 2, 1],
              [4, 4, 3],
              [6, 2, 2],
              [1, 3, 9],
              [7, 7, 9]])
Z1 = linkage(X, method='single', metric='euclidean')  # 类内有多个点时，排列组合寻找两个集合内最近的点
Z2 = linkage(Y, method='single', metric='euclidean')  # 1,2距离是3，2,3距离也是3，可以直接归为一类，不用管1,3距离
print(Z1, type(Z1))
print(Z2, type(Z2))
'''
linkage(y, method='single', metric='euclidean', optimal_ordering=False)
y是1维压缩向量（距离向量)或者是2维观测向量（坐标矩阵）。1维压缩向量必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
n个点样本，一定做n-1次聚类，输出结果为4*(n-1)
method计算类间距离的方法:single 最小距离;complete 最大距离;average 平均距离;weighted 加权距离;centroid 中心距离;median 和中心距离相似;ward 基于Ward方差最小化算法
metric:euclidean	欧氏距离 ∥u−v∥ 2
'''
f1 = fcluster(Z1, 2, 'distance')  # ’distance’：类间距离的阈值，每个簇的距离不超过t=3，小于阈值的样本才能算一类
f2 = fcluster(Z2, 4, 'distance')
print(f1, type(f1))  # 簇间t=2，超过的不为一簇，可以结合链式图查看
print(f2, type(f2))
'''
fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
压平树状图Z为4*(n-1)矩阵
'''
fig1 = plt.figure(num=1, figsize=(5, 5))
dn1 = dendrogram(Z1)  # 不考虑簇间距离，合并簇到最后1簇
print(dn1)
plt.show()
fig2 = plt.figure(num=2, figsize=(5, 5))
dn2 = dendrogram(Z2)  # 绘图纵轴表示ward方差，single方法则为最近欧氏距离
print(dn2)  # 结果什么意思？？？
plt.show()
'''
num:图像编号或名称，数字为编号 ，字符串为名称。不指定调用figure时就会默认从1开始。
figsize:指定figure的宽和高，单位为英寸
dpi参数指定绘图对象的分辨率，即每英寸多少个像素
facecolor:背景颜色
edgecolor:边框颜色
frameon:是否显示边框
'''
