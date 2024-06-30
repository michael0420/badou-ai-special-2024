# -*- coding: utf-8 -*-
"""
@File    :   层次聚类.py
@Time    :   2024/06/02 13:56:22
@Author  :   廖红洋 
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 这里的矩阵可以是任意维的，只要能够计算欧式距离就可以聚类，所以聚类不只是二维三维这种可以可视化的数据能聚类，四维五维一样可以聚类
matrix = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [6, 8, 6],
    [-1, 0, 2],
    [3, 5, 8],
    [7, 7, 6],
    [0, 0, 2],
    [2, 1, -3],
    [1, 0, 0],
    [9, 12, 5],
    [3, 3, 5],
]
dis = linkage(matrix)  # dis是聚类图，包含距离和聚类过程
result = fcluster(
    dis, 2, "distance"
)  # result保存聚类结果；2是聚类阈值，超过2就增加一类；'distance'是聚类规则，表示按距离聚类
pic = plt.figure(figsize=(6, 3))  # 设定图像尺寸
dn = dendrogram(dis)  # 生成聚类树图
print(result)
plt.show()
