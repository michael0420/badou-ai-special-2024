from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


X = [[1, 3], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, "ward")
print(Z)
F = fcluster(Z, 4, "distance")
fig = plt.figure(figsize=(10, 12))
dn = dendrogram(Z)
plt.show()