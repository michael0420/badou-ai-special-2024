from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

X =[[58, 93], [27, 84], [65, 3], [92, 11], [78, 50], [4, 72], [15, 89], [46, 26], [31, 5], [77, 94]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(10, 7))
dn = dendrogram(Z)
print(Z)
plt.show()
