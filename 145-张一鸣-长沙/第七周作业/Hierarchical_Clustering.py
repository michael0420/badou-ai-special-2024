# coding = utf-8

'''
        实现层次聚类
'''

import cv2
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

# 定义一组数据
X = [[1, 3], [2, 4], [3, 5], [6, 8], [4, 5], [8, 8], [0, 1], [2, 3]]
# linkage函数生成链接矩阵
# Ward方差最小化方法尝试最小化聚类内方差的总和
# 在每一步中合并两个聚类，这两个聚类的合并会导致所有聚类的方差总和增加得最少
link = linkage(X, 'ward')
print(type(link), link)
# 设置停止迭代阈值为3，fcluster函数返回聚类标签数组，distance基于距离来定义聚类
fc = fcluster(link, 3, 'distance')
print(type(fc), fc)
# 绘制层次聚类树状图
plt.figure(figsize=(8, 8))
dd = dendrogram(link)

plt.show()
