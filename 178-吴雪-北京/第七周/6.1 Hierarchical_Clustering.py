"""
层次聚类算法(Hierarchical Clustering)
"""
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, method='ward')
f = fcluster(Z, 4, criterion='distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
