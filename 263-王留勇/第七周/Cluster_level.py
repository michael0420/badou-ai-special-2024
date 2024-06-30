"""
层次聚类
"""

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
import matplotlib.pyplot as plt

X = [[1, 3], [2, 4], [5, 6], [3, 9], [5, 9], [4, 7]]
Z = linkage(X, 'ward')
F = fcluster(Z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
