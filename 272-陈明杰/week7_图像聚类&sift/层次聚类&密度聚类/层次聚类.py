from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt

# 创建一些初始化数据
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# 进行图像聚类，计算点与点之间的距离时使用“ward”，最短距离法
Z=linkage(X,method='ward')
# 用于在层次聚类（hierarchical clustering）中将链接矩阵（linkage matrix）转换
# 为扁平聚类（flat clustering），1表示阈值，距离小于1的就是同一类
f=fcluster(Z,1,'distance')
print(f)
# 画层次聚类的树状图
dendrogram(Z)
plt.show()
