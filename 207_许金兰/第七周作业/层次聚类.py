"""
@author: 207-xujinlan
层次聚类算法
"""

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

data = [[3,1],[5,8],[2,4],[9,7],[5,6]]
# 1.linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)，这个方法的使用在层次聚类之前。
Z = linkage(data, 'ward')
# 2.从给定的连接矩阵定义中的层次聚类中形成平面聚类。
f = fcluster(Z,4,'distance')
# 3.将层次聚类绘制成树状图。
plt.figure()
dn = dendrogram(Z)

