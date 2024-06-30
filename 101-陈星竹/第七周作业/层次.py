from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

X = [[1,2],[3,2],[4,4],[2,4],[1,3],[3,1],[4,1],[2,2]]
Z = linkage(X,'single')
print(Z)
f = fcluster(Z,5,'distance')
print(f)
fig = plt.figure(figsize=(10,10))
dn = dendrogram(Z)
plt.show()