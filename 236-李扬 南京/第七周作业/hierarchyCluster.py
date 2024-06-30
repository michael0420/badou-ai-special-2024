from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')          #进行层次聚类
f = fcluster(Z, 4, 'distance')  #得到聚类结果
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
print(f)
print(dn)
plt.show()
