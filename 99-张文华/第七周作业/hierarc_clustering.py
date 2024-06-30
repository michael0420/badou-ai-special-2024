'''
实现层次聚类
'''

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


date = [[1,3], [5,6], [7,8], [2,5], [3,3]]
Z = linkage(date, 'ward')
print(Z)

f = fcluster(Z, 4, 'distance')
print(f)

fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(dn)

plt.show()