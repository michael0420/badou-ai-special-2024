# cluster
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt


X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

# 层次聚类
Z = linkage(X,'ward')
W = linkage(X,'single')
print('Z:\n',Z)
print("W:\n",W)

# 聚类的结果
f = fcluster(Z,4,'distance')

print(f)
# 树状图
zn = dendrogram(Z)
plt.show()
# 树状图
wn = dendrogram(W)
plt.show()