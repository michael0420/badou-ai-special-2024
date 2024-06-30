from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# 使用linkage进行层次聚类
Z = linkage(X, 'ward')
print(Z)
# 是用clusters划分不同的簇
clusters = fcluster(Z, 3, criterion='distance')
print(clusters)
fig = plt.figure(figsize=(5, 3))
# 树状图绘图
dn = dendrogram(Z)
plt.show()
